#!/usr/bin/env python3
"""
M1 - The scorer. Every arm is measured through this module.

Writing it before any agent exists is deliberate: the scikit-learn
baseline, the single agent, the MAS and HOMAS arms all score through the
same code path, so their numbers are directly comparable rather than
computed four slightly different ways.

PREDICTIONS FORMAT
------------------
JSONL, one object per episode per repetition:

    {"episode_id": "held_001",
     "arm": "A0",                       # A0 | B | C | sklearn | trivial
     "rep": 1,
     "predictions": {"severity": "Major", "component": "Git",
                     "domain": "Core VCS", "resolution_class": "Fixed"},
     "confidence": {"severity": 0.8, "component": 0.6},   # optional, 0-1
     "escalated": {"severity": false},                    # optional, arm C
     "meta": {"tokens_in": 4210, "tokens_out": 340, "seconds": 12.4}}

A missing or null prediction counts as wrong, never as skipped. An arm
that fails to answer has failed, and hiding that would flatter it.

USAGE
-----
    # score one predictions file
    python m1_score.py --predictions runs/sklearn_dev.jsonl --split dev

    # compare several arms at once
    python m1_score.py --predictions runs/*.jsonl --split held_out --save
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_DB = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://homas:homas@localhost:5432/homas")
LABELS = ["severity", "component", "domain", "resolution_class"]
CATCH_ALL = {"component": "General", "domain": "General"}
MIN_SUPPORT = 20          # below this, per-class figures are not reported
CALIBRATION_BINS = [0.0, 0.5, 0.7, 0.85, 1.01]


# ---------------------------------------------------------------- loading

def load_episodes(path="episodes/episodes.jsonl", split=None):
    eps = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if split and e["split"] != split:
                continue
            eps[e["episode_id"]] = e
    if not eps:
        raise SystemExit(f"No episodes for split={split!r} in {path}")
    return eps


def load_predictions(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No predictions matched {patterns}")
    return rows


# ---------------------------------------------------------------- scoring

def pairs(episodes, preds, label):
    """(truth, predicted) for episodes where this label was actually asked."""
    out = []
    for p in preds:
        ep = episodes.get(p["episode_id"])
        if ep is None or label not in asked_labels(ep):
            continue
        truth = ep["truth"][label]
        if truth is None:
            continue
        got = (p.get("predictions") or {}).get(label)
        out.append((truth, got if got is not None else "<none>"))
    return out


def asked_labels(ep):
    """domain is never asked directly - the Router predicts it."""
    return set(ep["labels_requested"]) | {"domain"}


def score_label(episodes, preds, label, baseline_class=None):
    pr = pairs(episodes, preds, label)
    if not pr:
        return None
    y_true = [a for a, _ in pr]
    y_pred = [b for _, b in pr]

    res = {
        "label": label,
        "n": len(pr),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "unanswered": sum(1 for b in y_pred if b == "<none>"),
    }
    if baseline_class is not None:
        res["baseline"] = sum(1 for a in y_true if a == baseline_class) / len(y_true)
        res["lift"] = res["accuracy"] - res["baseline"]

    # The catch-all problem: 'General' is what a triager picks when not
    # choosing a specific area, so predicting it is partly predicting an
    # absence of judgement. Report the specific-label subset separately.
    ca = CATCH_ALL.get(label)
    if ca:
        sub = [(a, b) for a, b in pr if a != ca]
        if sub:
            res["accuracy_excl_catchall"] = accuracy_score(
                [a for a, _ in sub], [b for _, b in sub])
            res["n_excl_catchall"] = len(sub)

    res["_y_true"], res["_y_pred"] = y_true, y_pred
    return res


def per_class_table(res):
    """Precision/recall/F1, only for classes with enough support."""
    rep = classification_report(res["_y_true"], res["_y_pred"],
                                output_dict=True, zero_division=0)
    rows = []
    for cls, m in rep.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if m["support"] < MIN_SUPPORT:
            continue
        rows.append({"class": cls, "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1-score"],
                     "support": int(m["support"])})
    return rows


def calibration(episodes, preds, label):
    """Accuracy by confidence band. The question is whether low-confidence
    predictions really are wrong more often - if not, gating on confidence
    is not justified and the escalation mechanism needs rethinking."""
    buckets = defaultdict(lambda: [0, 0])
    for p in preds:
        ep = episodes.get(p["episode_id"])
        if ep is None or label not in asked_labels(ep):
            continue
        conf = (p.get("confidence") or {}).get(label)
        if conf is None:
            continue
        truth = ep["truth"][label]
        if truth is None:
            continue
        got = (p.get("predictions") or {}).get(label)
        for lo, hi in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:]):
            if lo <= conf < hi:
                buckets[(lo, hi)][0] += 1
                buckets[(lo, hi)][1] += int(got == truth)
                break
    return [{"band": f"{lo:.0%}-{hi:.0%}" if hi <= 1 else f"{lo:.0%}-100%",
             "n": n, "accuracy": hits / n if n else None}
            for (lo, hi), (n, hits) in sorted(buckets.items())]


def cost(preds):
    tin = sum((p.get("meta") or {}).get("tokens_in", 0) for p in preds)
    tout = sum((p.get("meta") or {}).get("tokens_out", 0) for p in preds)
    secs = sum((p.get("meta") or {}).get("seconds", 0) for p in preds)
    n = max(len({p["episode_id"] for p in preds}), 1)
    return {"tokens_in": tin, "tokens_out": tout, "seconds": secs,
            "tokens_per_episode": (tin + tout) / n,
            "seconds_per_episode": secs / n}


# ---------------------------------------------------------------- display

def baseline_classes(episodes, db=None):
    """Majority class per label.

    Taken from the CORPUS (label_stats), not from the episodes being
    scored. Using the episode-set majority would mean the baseline had
    seen the answer key, which flatters it against any honest predictor.
    Falls back to the episode majority only if the database is
    unreachable, and says so.
    """
    if db:
        try:
            with create_engine(db).connect() as con:
                rows = con.execute(sa_text(
                    "SELECT label, majority_class FROM label_stats")).all()
            out = {lbl: cls for lbl, cls in rows}
            out["resolution_class"] = out.get("resolution")
            out.setdefault("domain", "General")
            if any(out.get(l) for l in LABELS):
                return out
        except Exception:
            print("  (label_stats unreadable - falling back to episode "
                  "majority, which is optimistic)")

    out = {}
    for label in LABELS:
        vals = [e["truth"][label] for e in episodes.values()
                if e["truth"][label] is not None]
        out[label] = pd.Series(vals).value_counts().idxmax() if vals else None
    return out


def score_arm(episodes, preds, arm, db=None):
    sub = [p for p in preds if p.get("arm") == arm]
    if not sub:
        return None
    bl = baseline_classes(episodes, db)
    results = {}
    for label in LABELS:
        r = score_label(episodes, sub, label, bl.get(label))
        if r:
            results[label] = r
    return {"arm": arm, "labels": results, "cost": cost(sub),
            "n_predictions": len(sub)}


def print_arm(a):
    print(f"\n{'=' * 72}\nARM: {a['arm']}   ({a['n_predictions']} predictions)\n{'=' * 72}")
    print(f"{'label':18s}{'n':>5s}{'acc':>8s}{'base':>8s}{'lift':>8s}"
          f"{'macroF1':>9s}{'kappa':>8s}{'unans':>7s}")
    for label, r in a["labels"].items():
        print(f"{label:18s}{r['n']:>5}{r['accuracy']:>7.1%}"
              f"{r.get('baseline', 0):>8.1%}{r.get('lift', 0):>+8.1%}"
              f"{r['macro_f1']:>9.3f}{r['kappa']:>8.3f}{r['unanswered']:>7}")
        if "accuracy_excl_catchall" in r:
            print(f"{'  (excl General)':18s}{r['n_excl_catchall']:>5}"
                  f"{r['accuracy_excl_catchall']:>7.1%}")

    c = a["cost"]
    if c["tokens_per_episode"]:
        print(f"\n  cost: {c['tokens_per_episode']:.0f} tokens/episode, "
              f"{c['seconds_per_episode']:.1f} s/episode")

    for label, r in a["labels"].items():
        rows = per_class_table(r)
        if rows:
            print(f"\n  per-class, {label} (support >= {MIN_SUPPORT}):")
            print(f"    {'class':24s}{'prec':>7s}{'rec':>7s}{'f1':>7s}{'n':>6s}")
            for x in rows:
                print(f"    {x['class']:24s}{x['precision']:>7.2f}"
                      f"{x['recall']:>7.2f}{x['f1']:>7.2f}{x['support']:>6}")


def print_calibration(episodes, preds, arm):
    sub = [p for p in preds if p.get("arm") == arm]
    any_shown = False
    for label in LABELS:
        bands = calibration(episodes, sub, label)
        if not bands:
            continue
        if not any_shown:
            print(f"\n  calibration (does low confidence mean low accuracy?):")
            any_shown = True
        cells = "  ".join(f"{b['band']}: {b['accuracy']:.0%} (n={b['n']})"
                          for b in bands if b["accuracy"] is not None)
        print(f"    {label:18s}{cells}")


def to_frame(arms):
    rows = []
    for a in arms:
        for label, r in a["labels"].items():
            rows.append({
                "arm": a["arm"], "label": label, "n": r["n"],
                "accuracy": r["accuracy"], "baseline": r.get("baseline"),
                "lift": r.get("lift"), "macro_f1": r["macro_f1"],
                "kappa": r["kappa"], "unanswered": r["unanswered"],
                "accuracy_excl_catchall": r.get("accuracy_excl_catchall"),
                "tokens_per_episode": a["cost"]["tokens_per_episode"],
                "seconds_per_episode": a["cost"]["seconds_per_episode"],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--episodes", default="episodes/episodes.jsonl")
    ap.add_argument("--split", default=None, choices=[None, "dev", "held_out"])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--save", action="store_true",
                    help="write the results table to Postgres")
    args = ap.parse_args()

    episodes = load_episodes(args.episodes, args.split)
    preds = load_predictions(args.predictions)
    print(f"{len(episodes)} episodes (split={args.split or 'all'}), "
          f"{len(preds)} predictions")

    arms = []
    for arm in sorted({p.get("arm", "?") for p in preds}):
        a = score_arm(episodes, preds, arm, args.db)
        if a:
            arms.append(a)
            print_arm(a)
            print_calibration(episodes, preds, arm)

    if len(arms) > 1:
        print(f"\n{'=' * 72}\nSUMMARY: accuracy by arm\n{'=' * 72}")
        print(f"{'label':18s}" + "".join(f"{a['arm']:>12s}" for a in arms))
        for label in LABELS:
            cells = ""
            for a in arms:
                r = a["labels"].get(label)
                cells += f"{r['accuracy']:>11.1%} " if r else f"{'-':>12s}"
            print(f"{label:18s}{cells}")

    if args.save:
        df = to_frame(arms)
        df["split"] = args.split or "all"
        engine = create_engine(args.db)
        df.to_sql("results", engine, if_exists="append", index=False)
        print(f"\nappended {len(df)} rows to results")


if __name__ == "__main__":
    main()