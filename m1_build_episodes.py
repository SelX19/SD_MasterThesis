#!/usr/bin/env python3
"""
M1 - Build the frozen episode set.  (v2)

An EPISODE is one triage question with a known answer.

    SHOWN     summary, description, type, reporter, date, attachment flag;
              the valid label sets; and N reference examples drawn only
              from issues RESOLVED BEFORE this issue was CREATED.

    ASKED     severity + component always.
              resolution_class additionally, where the issue has one.
              (domain is derived from component and is what the Router is
               scored against - it is never shown to any agent.)

    HIDDEN    the values the human triage team actually recorded.

Changes from v1
-----------------------
* 60 held-out + 15 dev (was 40 + 10).
* Variable label sets: resolution is optional, so issues that were never
  resolved still make usable episodes. Candidate pool ~250 -> ~830.
* Description optional; summary alone is enough. Vague and incomplete
  tickets stay in - they are the realism argument, and they are where the
  arms are expected to separate.
* Each component maps to one of four DOMAINS, for Router scoring.
* severity_source flags whether the label is triager-validated (issue was
  resolved, 96.7% filled) or reporter-assigned (never resolved, 80.8%).
  Reported as a breakdown rather than used as a filter.
* priority is never asked: 88.6% "Low" makes it unscoreable.

Usage
-----
    python m1_build_episodes.py --db "postgresql+psycopg://you@localhost:5432/homas" --dry-run
    python m1_build_episodes.py --db "..."            # writes
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB = "postgresql+psycopg://homas:homas@localhost:5432/homas"
SEED = 20261117

N_DEV = 15
N_HELD_OUT = 60
N_REFERENCE = 20
MIN_REFERENCE = 10

SEVERITIES = ["Critical", "Major", "Minor"]

RESOLUTION_MAP = {
    "Fixed": "Fixed", "Done": "Fixed",
    "Duplicate": "Duplicate",
    "Cannot Reproduce": "Cannot Reproduce", "Incomplete": "Cannot Reproduce",
    "Spam": "Invalid", "Not a bug": "Invalid", "Invalid": "Invalid",
    "Incorrectly Filed": "Invalid", "Answered": "Invalid",
    "Resolved Locally": "Invalid",
}
RESOLUTION_CLASSES = ["Fixed", "Duplicate", "Cannot Reproduce", "Invalid"]

# The Router predicts a domain from issue text; these are the four it
# chooses between. Derived from component, never shown.
DOMAIN_MAP = {
    "Git": "Core VCS", "branch": "Core VCS", "remote": "Core VCS",
    "Stash": "Core VCS", "Mercurial": "Core VCS", "hg": "Core VCS",
    "Bitbucket": "Integrations", "REST": "Integrations", "Web": "Integrations",
    "Askpass": "Integrations", "MercurialCredentialManager": "Integrations",
    "UX": "Client and UX", "ChangeSetViewManager": "Client and UX",
    "Beta": "Client and UX", "mac": "Client and UX",
    "General": "General",
}
DOMAINS = ["Core VCS", "Integrations", "Client and UX", "General"]


def load_corpus(engine):
    with engine.connect() as con:
        df = pd.read_sql(text("""
            SELECT issue_key, summary, description, issue_type, status,
                   reporter, created, resolved, resolution, severity,
                   component, votes, has_attachment, desc_len
            FROM issues ORDER BY created
        """), con)
        components = [r[0] for r in con.execute(text("""
            SELECT component FROM issues WHERE component IS NOT NULL
            GROUP BY 1 ORDER BY COUNT(*) DESC
        """))]
    df["resolution_class"] = df["resolution"].map(RESOLUTION_MAP)
    df["domain"] = df["component"].map(DOMAIN_MAP)
    return df, components


def eligible(df):
    """Targets need a severity, a mappable component, and a summary."""
    m = (df["severity"].notna() & df["component"].notna()
         & df["domain"].notna() & df["summary"].notna()
         & df["created"].notna())
    unmapped = set(df.loc[df["component"].notna() & df["domain"].isna(),
                          "component"])
    if unmapped:
        print(f"  !! components missing from DOMAIN_MAP: {sorted(unmapped)}")
    return df[m].copy()


def reference_pool(df):
    """Examples must be fully decided: labelled AND resolved."""
    m = (df["resolution_class"].notna() & df["severity"].notna()
         & df["component"].notna() & df["summary"].notna()
         & df["resolved"].notna())
    return df[m].sort_values("resolved").copy()


def build_references(pool, target_created, n):
    """The n issues most recently RESOLVED before target_created.

    Strict truncation: an example must have been decided before the target
    issue arrived, so nothing from after the decision point can leak in.
    """
    prior = pool[pool["resolved"] < target_created]
    if len(prior) < MIN_REFERENCE:
        return None
    return [{"summary": r.summary, "severity": r.severity,
             "component": r.component, "resolution_class": r.resolution_class}
            for r in prior.tail(n).itertuples()]


def make_episode(row, refs, components):
    asked = ["severity", "component"]
    if pd.notna(row.resolution_class):
        asked.append("resolution_class")

    desc = row.description if pd.notna(row.description) else None

    return {
        "issue_key": row.issue_key,
        "labels_requested": asked,
        "input": {
            "summary": row.summary,
            "description": desc,
            "issue_type": row.issue_type,
            "reporter": row.reporter,
            "created": row.created.strftime("%Y-%m-%d"),
            "has_attachment": bool(row.has_attachment),
            "description_length": int(row.desc_len or 0),
            "valid_severities": SEVERITIES,
            "valid_components": components,
            "valid_domains": DOMAINS,
            "valid_resolution_classes": RESOLUTION_CLASSES,
            "reference_examples": refs,
        },
        "truth": {
            "severity": row.severity,
            "component": row.component,
            "domain": row.domain,
            "resolution_class": (row.resolution_class
                                 if pd.notna(row.resolution_class) else None),
        },
        "meta": {
            # Resolved issues passed through a triager, who fills or
            # corrects severity; never-resolved ones carry the reporter's
            # own assessment.
            "severity_source": ("triager_validated" if pd.notna(row.resolved)
                                else "reporter_assigned"),
            "has_description": desc is not None,
        },
    }


def stratified_sample(cands, n, rng):
    """Spread over severity x domain x resolution_class.

    Stratifying on the resolution CLASS rather than merely on whether one
    exists keeps rare outcomes (Cannot Reproduce has only 45 instances in
    the whole corpus) from being crowded out by Fixed. Many of the 60
    cells are empty; the round-robin below simply skips those, so the
    draw is as even as the data allows.
    """
    cells = defaultdict(list)
    for e in cands:
        cells[(e["truth"]["severity"], e["truth"]["domain"],
               e["truth"]["resolution_class"])].append(e)
    for v in cells.values():
        rng.shuffle(v)

    keys = sorted(cells, key=lambda k: (k[0], k[1], k[2] or ""))
    out = []
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


def trivial_baselines(episodes, corpus):
    rows = []
    for label in ("severity", "component", "domain", "resolution_class"):
        majority = corpus[label].dropna().value_counts().idxmax()
        truths = [e["truth"][label] for e in episodes
                  if e["truth"][label] is not None]
        if not truths:
            continue
        hits = sum(1 for t in truths if t == majority)
        rows.append({"label": label, "strategy": f"always '{majority}'",
                     "n_episodes": len(truths),
                     "accuracy": hits / len(truths)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="episodes/episodes.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    engine = create_engine(args.db)

    print("loading corpus ...")
    corpus, components = load_corpus(engine)
    print(f"  {len(corpus):,} issues, {len(components)} components, "
          f"{len(DOMAINS)} domains")

    cands_df = eligible(corpus)
    pool = reference_pool(corpus)
    print(f"  {len(cands_df):,} eligible targets, {len(pool):,} reference pool")

    candidates, thin = [], 0
    for row in cands_df.itertuples():
        refs = build_references(pool, row.created, N_REFERENCE)
        if refs is None:
            thin += 1
            continue
        candidates.append(make_episode(row, refs, components))
    print(f"  {len(candidates):,} candidates "
          f"({thin} skipped: fewer than {MIN_REFERENCE} prior examples)")

    need = N_DEV + N_HELD_OUT
    if len(candidates) < need:
        raise SystemExit(f"Only {len(candidates)} candidates, need {need}.")

    picked = stratified_sample(candidates, need, rng)
    rng.shuffle(picked)
    dev, held = picked[:N_DEV], picked[N_DEV:need]
    for i, e in enumerate(dev, 1):
        e["episode_id"], e["split"] = f"dev_{i:03d}", "dev"
    for i, e in enumerate(held, 1):
        e["episode_id"], e["split"] = f"held_{i:03d}", "held_out"
    episodes = dev + held

    clash = {e["issue_key"] for e in dev} & {e["issue_key"] for e in held}
    assert not clash, f"dev and held-out share issues: {clash}"

    # ---- report ---------------------------------------------------------
    print(f"\n{len(dev)} dev + {len(held)} held-out\n")
    for label in ("severity", "domain", "resolution_class"):
        c = Counter(e["truth"][label] for e in held
                    if e["truth"][label] is not None)
        print(f"  held-out {label:18s} " +
              " | ".join(f"{k} {v}" for k, v in c.most_common()))
    srcs = Counter(e["meta"]["severity_source"] for e in held)
    withdesc = sum(1 for e in held if e["meta"]["has_description"])
    withres = sum(1 for e in held if "resolution_class" in e["labels_requested"])
    print(f"\n  severity source: {dict(srcs)}")
    print(f"  with description: {withdesc}/{len(held)}   "
          f"resolution asked: {withres}/{len(held)}")

    print("\n=== TRIVIAL BASELINES, held-out set ===\n")
    base = trivial_baselines(held, corpus)
    print(f"{'label':20s}{'strategy':24s}{'n':>5s}{'accuracy':>11s}")
    for b in base:
        print(f"{b['label']:20s}{b['strategy']:24s}"
              f"{b['n_episodes']:>5}{b['accuracy']:>10.1%}")
    print("\n> No arm that fails to beat these has told you anything.")

    if args.dry_run:
        print("\ndry run - nothing written\n")
        s = json.loads(json.dumps(episodes[0]))
        if s["input"]["description"]:
            s["input"]["description"] = s["input"]["description"][:250] + " ..."
        s["input"]["reference_examples"] = \
            s["input"]["reference_examples"][:2] + ["... 18 more"]
        s["input"]["valid_components"] = s["input"]["valid_components"][:4] + ["..."]
        print(json.dumps(s, indent=2, ensure_ascii=False)[:2200])
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nwritten: {out} ({out.stat().st_size/1024:.0f} KB)")

    flat = pd.DataFrame([{
        "episode_id": e["episode_id"], "split": e["split"],
        "issue_key": e["issue_key"],
        "labels_requested": ",".join(e["labels_requested"]),
        "truth_severity": e["truth"]["severity"],
        "truth_component": e["truth"]["component"],
        "truth_domain": e["truth"]["domain"],
        "truth_resolution_class": e["truth"]["resolution_class"],
        "severity_source": e["meta"]["severity_source"],
        "has_description": e["meta"]["has_description"],
        "n_reference_examples": len(e["input"]["reference_examples"]),
    } for e in episodes])

    flat.to_sql("episodes", engine, if_exists="replace", index=False)
    pd.DataFrame(base).to_sql("baseline_scores", engine,
                              if_exists="replace", index=False)
    with engine.begin() as con:
        con.execute(text("ALTER TABLE episodes ADD PRIMARY KEY (episode_id)"))
        con.execute(text("""ALTER TABLE episodes ADD CONSTRAINT fk_episode_issue
                            FOREIGN KEY (issue_key)
                            REFERENCES issues(issue_key)"""))

    print("tables: episodes, baseline_scores")
    print("\nFREEZE:  git add episodes/episodes.jsonl && "
          "git commit -m 'M1: frozen episode set (60 held-out + 15 dev)'")


if __name__ == "__main__":
    main()