#!/usr/bin/env python3
"""
M2 - The harness.

Loads episodes, runs one arm over them, logs everything, writes
predictions in the format m1_score.py expects.

The agents do the thinking; this does the bookkeeping: rate limiting,
retries, timing, token accounting, and failure handling.

ARM A0 - single agent, no delegation
    One agent answers every requested label in one call. Sampled k times
    at a non-zero temperature; the modal answer wins and the proportion
    agreeing becomes the confidence. This is self-consistency, which is
    better calibrated than asking a model to rate its own confidence, and
    it is the only confidence signal available to an architecture with
    one agent. Arms B and C will instead derive confidence from agreement
    between agents, which is a property of orchestration.

Usage
-----
    python m2_runner.py --arm A0 --split dev
    python m2_runner.py --arm A0 --split dev --limit 3 --k 2   # quick probe
    python m1_score.py --predictions runs/A0_dev.jsonl --split dev --db ...
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy import create_engine, text

load_dotenv()   # before DEFAULT_DB reads the environment, not after

# Put DATABASE_URL in .env so no command needs the connection string.
DEFAULT_DB = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://homas:homas@localhost:5432/homas")

# Pinned and recorded with every run. Changing the model mid-study would
# invalidate earlier runs, so the value is written into the runs table.
DEFAULT_MODEL = "models/gemma-4-31b-it"

# Per-model free-tier ceilings, read from the AI Studio rate-limit console.
LIMITS = {
    "models/gemma-4-31b-it":        (30, 14_400),
    "models/gemma-4-26b-a4b-it":    (30, 14_400),
    "models/gemini-3.5-flash-lite": (15, 500),
    "models/gemini-3.1-flash-lite": (15, 500),
}

TEMPERATURE = 0.7          # non-zero: self-consistency needs variation
MAX_OUTPUT_TOKENS = 800
# Gemma reasons internally before answering, which dominated latency at
# roughly 150 s per call. Zero asks for no reasoning; models that ignore
# the setting simply behave as before.
THINKING_BUDGET = 0
TIMEOUT_MS = 120_000
K_SAMPLES = 5
MAX_RETRIES = 4
WORKERS = 8                # calls in flight; the limiter still governs rate


# ----------------------------------------------------------- rate limiting

class RateLimiter:
    """Requests per minute and per day, enforced locally.

    Cheaper to wait than to be refused: a 429 still consumes latency and
    the daily quota resets only at midnight Pacific.
    """

    def __init__(self, rpm, rpd):
        self.rpm, self.rpd = rpm, rpd
        self.minute, self.day = [], 0
        self.started = time.time()
        self._lock = threading.Lock()   # several calls are in flight at once

    def wait(self):
        with self._lock:
            self._wait_locked()

    def _wait_locked(self):
        if self.day >= self.rpd:
            raise SystemExit(
                f"Daily quota exhausted ({self.rpd}). Resets midnight Pacific.")
        now = time.time()
        self.minute = [t for t in self.minute if now - t < 60]
        if len(self.minute) >= self.rpm:
            sleep = 60 - (now - self.minute[0]) + 0.5
            if sleep > 0:
                print(f"    [rate limit: sleeping {sleep:.0f}s]", flush=True)
                time.sleep(sleep)
                now = time.time()
                self.minute = [t for t in self.minute if now - t < 60]
        self.minute.append(time.time())
        self.day += 1


# -------------------------------------------------------------- model call

def _config(temperature, no_thinking=True):
    kw = dict(temperature=temperature,
              max_output_tokens=MAX_OUTPUT_TOKENS,
              http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    if no_thinking:
        try:
            kw["thinking_config"] = types.ThinkingConfig(
                thinking_budget=THINKING_BUDGET)
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


def call_model(client, prompt, limiter, temperature, model):
    """One call, with backoff. Returns (text, usage, seconds, error)."""
    no_thinking = True
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=model,
                contents=prompt,
                config=_config(temperature, no_thinking),
            )
            u = r.usage_metadata
            usage = {
                "tokens_in": getattr(u, "prompt_token_count", 0) or 0,
                "tokens_out": getattr(u, "candidates_token_count", 0) or 0,
                # Gemma reasons internally before answering; these tokens
                # dominate the bill and must be counted separately or the
                # resource-efficiency metric is badly wrong.
                "tokens_thinking": getattr(u, "thoughts_token_count", 0) or 0,
            }
            txt = r.text or ""
            if not txt.strip():
                print(f"    [EMPTY response from {model}; "
                      f"usage={usage}]", flush=True)
            return txt, usage, time.time() - t0, None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            # Models differ on whether a zero thinking budget is legal: the
            # Gemini 2.5 family rejects it with a generic 400 rather than a
            # message naming the field. Drop the setting and retry rather
            # than failing the episode.
            if no_thinking and ("thinking" in str(e).lower()
                                or "INVALID_ARGUMENT" in str(e)
                                or "400" in str(e)):
                print(f"    [{model} rejected thinking_budget; "
                      f"retrying without it]", flush=True)
                no_thinking = False
                continue
            transient = any(s in str(e) for s in
                            ("429", "500", "503", "504", "timeout", "UNAVAILABLE"))
            if attempt == MAX_RETRIES - 1 or not transient:
                # A silent failure looks identical to a fast success in the
                # summary line, so say what happened.
                print(f"    [FAILED {model}: {msg[:200]}]", flush=True)
                return "", {}, time.time() - t0, msg
            backoff = 2 ** attempt + random.random()
            print(f"    [retry {attempt + 1}: {msg[:70]} - {backoff:.1f}s]",
                  flush=True)
            time.sleep(backoff)
    return "", {}, 0.0, "exhausted retries"


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(raw):
    """Models wrap JSON in fences, prefaces and trailing commentary.
    Strip what we can; a failure here is a real failure and is counted."""
    if not raw:
        return None
    s = raw.strip()
    for fence in ("```json", "```"):
        s = s.removeprefix(fence)
    s = s.removesuffix("```").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ prompt

SYSTEM = (
    "You are an experienced QA engineer triaging incoming bug reports for "
    "Sourcetree, a desktop Git client for Windows. You assign each report "
    "the labels the triage team would record.\n"
)


def build_prompt(ep):
    i, asked = ep["input"], ep["labels_requested"]
    parts = [SYSTEM]

    parts.append("\nPreviously triaged reports from this project, with the "
                 "labels the team assigned:\n")
    for r in i["reference_examples"]:
        parts.append(f"- \"{r['summary']}\" -> severity: {r['severity']}, "
                     f"component: {r['component']}, "
                     f"outcome: {r['resolution_class']}\n")

    parts.append("\n--- REPORT TO TRIAGE ---\n")
    parts.append(f"Summary: {i['summary']}\n")
    if i.get("description"):
        parts.append(f"Description: {i['description']}\n")
    parts.append(f"Type: {i['issue_type']}\n")
    parts.append(f"Reported: {i['created']}\n")
    parts.append(f"Attachment included: {'yes' if i['has_attachment'] else 'no'}\n")

    parts.append("\n--- YOUR TASK ---\n")
    fields = []
    if "severity" in asked:
        parts.append(f"severity, one of: {', '.join(i['valid_severities'])}\n")
        fields.append('"severity": "..."')
    # component is always answered: the Router's domain derives from it
    parts.append(f"component, one of: {', '.join(i['valid_components'])}\n")
    fields.append('"component": "..."')
    if "resolution_class" in asked:
        parts.append("the outcome this report will eventually receive, one of: "
                     f"{', '.join(i['valid_resolution_classes'])}\n")
        fields.append('"resolution_class": "..."')
    fields.append('"reasoning": "one short sentence"')

    parts.append("\nRespond with ONLY a JSON object. No markdown fences, no "
                 "commentary before or after:\n")
    parts.append("{" + ", ".join(fields) + "}")
    return "".join(parts)


DOMAIN_MAP = {
    "Git": "Core VCS", "branch": "Core VCS", "remote": "Core VCS",
    "Stash": "Core VCS", "Mercurial": "Core VCS", "hg": "Core VCS",
    "Bitbucket": "Integrations", "REST": "Integrations", "Web": "Integrations",
    "Askpass": "Integrations", "MercurialCredentialManager": "Integrations",
    "UX": "Client and UX", "ChangeSetViewManager": "Client and UX",
    "Beta": "Client and UX", "mac": "Client and UX",
    "General": "General",
}


# -------------------------------------------------------------------- arms

def run_a0(client, ep, limiter, k, log, model, pool):
    """Single agent, sampled k times. Mode wins; agreement is confidence.

    The k samples are independent, so they are issued concurrently. The
    rate limiter still governs how fast requests leave, so this reduces
    wall-clock time without exceeding any quota.
    """
    prompt = build_prompt(ep)
    asked = set(ep["labels_requested"]) | {"component"}

    futures = [pool.submit(call_model, client, prompt, limiter,
                           TEMPERATURE, model) for _ in range(k)]

    samples, usage_tot, secs_tot, failures = [], Counter(), 0.0, 0
    for s, fut in enumerate(futures):
        raw, usage, secs, err = fut.result()
        secs_tot += secs
        usage_tot.update(usage)
        parsed = parse_json(raw)
        log.append({
            "episode_id": ep["episode_id"], "agent": "single-agent",
            "sample": s + 1, "ok": parsed is not None,
            "error": err, "raw": (raw or "")[:500], **usage,
        })
        if parsed:
            samples.append(parsed)
        else:
            failures += 1

    predictions, confidence = {}, {}
    for label in asked:
        votes = [str(s.get(label)) for s in samples if s.get(label)]
        if votes:
            top, n = Counter(votes).most_common(1)[0]
            predictions[label] = top
            confidence[label] = n / len(votes)

    if predictions.get("component"):
        predictions["domain"] = DOMAIN_MAP.get(predictions["component"])
        confidence["domain"] = confidence.get("component", 0.0)

    return {
        "predictions": predictions,
        "confidence": confidence,
        "meta": {
            "tokens_in": usage_tot["tokens_in"],
            "tokens_out": usage_tot["tokens_out"],
            "tokens_thinking": usage_tot["tokens_thinking"],
            "seconds": round(secs_tot, 2),   # summed API time, not wall clock
            "samples_ok": len(samples),
            "samples_failed": failures,
            "calls": k,
        },
    }


ARMS = {"A0": run_a0}


# -------------------------------------------------------------------- main

def ensure_tables(engine):
    with engine.begin() as con:
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS runs (
              run_id text PRIMARY KEY, arm text, split text, model text,
              temperature double precision, k_samples int,
              n_episodes int, started timestamptz, finished timestamptz,
              tokens_in bigint, tokens_out bigint, tokens_thinking bigint,
              seconds double precision, failures int)"""))
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_calls (
              run_id text, episode_id text, agent text, sample int,
              ok boolean, error text, raw text,
              tokens_in int, tokens_out int, tokens_thinking int)"""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A0", choices=sorted(ARMS))
    ap.add_argument("--split", default="dev", choices=["dev", "held_out"])
    ap.add_argument("--episodes", default="episodes/episodes.jsonl")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--k", type=int, default=K_SAMPLES)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N episodes (for a quick probe)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    if args.split == "held_out":
        print("\n!! HELD-OUT SET. Run this once, when the system is finished.")
        if input("   Type 'yes' to proceed: ").strip().lower() != "yes":
            raise SystemExit("aborted")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rpm, rpd = LIMITS.get(args.model, (15, 500))
    limiter = RateLimiter(rpm, rpd)

    episodes = [json.loads(l) for l in open(args.episodes, encoding="utf-8")]
    episodes = [e for e in episodes if e["split"] == args.split]
    if args.limit:
        episodes = episodes[:args.limit]

    run_id = f"{args.arm}_{args.split}_r{args.rep}_" \
             f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    print(f"\nrun {run_id}\n  model {args.model}  k={args.k}  "
          f"{rpm} rpm / {rpd} rpd  {args.workers} workers\n  "
          f"{len(episodes)} episodes  ~{len(episodes) * args.k} calls\n")

    started = datetime.now(timezone.utc)
    rows, log = [], []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for n, ep in enumerate(episodes, 1):
        t0 = time.time()
        out = ARMS[args.arm](client, ep, limiter, args.k, log,
                             args.model, pool)
        rows.append({"episode_id": ep["episode_id"], "arm": args.arm,
                     "rep": args.rep, **out})
        p = out["predictions"]
        ok = out["meta"]["samples_ok"]
        print(f"  [{n:>3}/{len(episodes)}] {ep['episode_id']}  "
              f"sev={p.get('severity', '-'):<9} comp={p.get('component', '-'):<14} "
              f"ok={ok}/{args.k}  {time.time() - t0:.0f}s", flush=True)
    finished = datetime.now(timezone.utc)
    pool.shutdown(wait=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.arm}_{args.split}_r{args.rep}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot = Counter()
    for r in rows:
        tot.update({k: v for k, v in r["meta"].items() if isinstance(v, (int, float))})

    engine = create_engine(args.db)
    ensure_tables(engine)
    with engine.begin() as con:
        con.execute(text("""
            INSERT INTO runs VALUES (:run_id,:arm,:split,:model,:temperature,
              :k,:n,:started,:finished,:ti,:to,:tt,:secs,:fail)"""), {
            "run_id": run_id, "arm": args.arm, "split": args.split,
            "model": args.model, "temperature": TEMPERATURE, "k": args.k,
            "n": len(rows), "started": started, "finished": finished,
            "ti": tot["tokens_in"], "to": tot["tokens_out"],
            "tt": tot["tokens_thinking"], "secs": tot["seconds"],
            "fail": tot["samples_failed"]})
    pd.DataFrame([{**c, "run_id": run_id} for c in log]).to_sql(
        "agent_calls", engine, if_exists="append", index=False)

    print(f"\nwritten: {path}")
    print(f"  {tot['tokens_in']:,} in + {tot['tokens_thinking']:,} thinking "
          f"+ {tot['tokens_out']:,} out   {tot['seconds']:.0f}s   "
          f"{tot['samples_failed']} failed samples")
    print(f"\nScore it:\n  python m1_score.py --predictions {path} "
          f"--split {args.split} --db \"{args.db}\"")


if __name__ == "__main__":
    main()