#!/usr/bin/env python3
"""
M1 - Trained classifier baseline.

TF-IDF over summary + description, one logistic regression per label.
This is NOT an agent and is not part of the HOMAS architecture. It is a
reference point: if a 1990s text classifier matches a hierarchy of
orchestrated language models, orchestration has not earned its cost, and
that is a finding worth reporting honestly.

It answers today's open question: do these tasks contain learnable
signal at all? If severity lands near 60% against a 36.7% baseline, the
agents have a real target. If it lands at 40%, the ceiling is low and
that is better known before a week is spent on agents.

LEAKAGE CONTROL
---------------
Training uses only issues that appear in NO episode, dev or held-out.
The episode issues are never seen during fitting.

One asymmetry to declare in the thesis: the classifier trains on a
global snapshot, so for some episodes its training data includes issues
resolved after that episode was created. The agents, by contrast, see
strictly time-truncated reference examples. The classifier therefore
holds a mild temporal advantage and should be read as an optimistic
reference rather than a strictly matched comparison.

USAGE
-----
    python m1_sklearn_baseline.py --db "postgresql+psycopg://you@localhost:5432/homas"
    python m1_score.py --predictions runs/baselines_dev.jsonl --split dev
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

DEFAULT_DB = "postgresql+psycopg://homas:homas@localhost:5432/homas"
SEED = 20261117

RESOLUTION_MAP = {
    "Fixed": "Fixed", "Done": "Fixed",
    "Duplicate": "Duplicate",
    "Cannot Reproduce": "Cannot Reproduce", "Incomplete": "Cannot Reproduce",
    "Spam": "Invalid", "Not a bug": "Invalid", "Invalid": "Invalid",
    "Incorrectly Filed": "Invalid", "Answered": "Invalid",
    "Resolved Locally": "Invalid",
}
DOMAIN_MAP = {
    "Git": "Core VCS", "branch": "Core VCS", "remote": "Core VCS",
    "Stash": "Core VCS", "Mercurial": "Core VCS", "hg": "Core VCS",
    "Bitbucket": "Integrations", "REST": "Integrations", "Web": "Integrations",
    "Askpass": "Integrations", "MercurialCredentialManager": "Integrations",
    "UX": "Client and UX", "ChangeSetViewManager": "Client and UX",
    "Beta": "Client and UX", "mac": "Client and UX",
    "General": "General",
}
TARGETS = ["severity", "component", "resolution_class"]


def text_of(row):
    """Summary plus description. Summary is weighted by repetition - a
    bug title is denser in signal than a long rambling description."""
    s = row.get("summary") or ""
    d = row.get("description") or ""
    return f"{s} {s} {d}".strip()


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=2, max_features=40000,
            stop_words="english")),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=2.0, random_state=SEED)),
    ])


def load(engine, episodes_path):
    with engine.connect() as con:
        corpus = pd.read_sql(text("""
            SELECT issue_key, summary, description, resolution,
                   severity, component
            FROM issues
        """), con)
    corpus["resolution_class"] = corpus["resolution"].map(RESOLUTION_MAP)

    episodes = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            episodes.append(json.loads(line))
    return corpus, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--episodes", default="episodes/episodes.jsonl")
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    engine = create_engine(args.db)
    corpus, episodes = load(engine, args.episodes)
    print(f"corpus {len(corpus):,} issues | episodes {len(episodes)}")

    episode_keys = {e["issue_key"] for e in episodes}
    train = corpus[~corpus["issue_key"].isin(episode_keys)].copy()
    print(f"training pool {len(train):,} "
          f"({len(episode_keys)} episode issues excluded)")

    train["text"] = train.apply(text_of, axis=1)
    ep_text = [text_of(e["input"]) for e in episodes]

    models, trivial = {}, {}
    for target in TARGETS:
        sub = train[train[target].notna() & (train["text"].str.len() > 0)]
        n_classes = sub[target].nunique()
        if len(sub) < 30 or n_classes < 2:
            print(f"  {target}: too little data ({len(sub)} rows) - skipped")
            continue

        pipe = build_pipeline()
        pipe.fit(sub["text"], sub[target])
        models[target] = pipe
        trivial[target] = sub[target].value_counts().idxmax()
        print(f"  {target}: fitted on {len(sub):,} rows, "
              f"{n_classes} classes, majority '{trivial[target]}'")

    # ---- how much signal is actually there? -----------------------------
    # Fifteen development episodes cannot answer this: the confidence
    # interval on 15 samples spans about +/-24 points. Cross-validation on
    # the full training pool can. Kappa matters more than accuracy here,
    # because it corrects for the class distribution - a kappa near zero
    # means the text carries no usable information about the label.
    print("\n=== cross-validated signal (5-fold, training pool) ===\n")
    print(f"{'label':20s}{'n':>7s}{'accuracy':>11s}{'kappa':>9s}   verdict")
    for target, pipe in models.items():
        sub = train[train[target].notna() & (train["text"].str.len() > 0)]
        counts = sub[target].value_counts()
        keep = counts[counts >= 5].index           # folds need each class
        sub = sub[sub[target].isin(keep)]
        if sub[target].nunique() < 2:
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        acc, kap = [], []
        for tr_i, te_i in cv.split(sub["text"], sub[target]):
            p = build_pipeline()
            p.fit(sub["text"].iloc[tr_i], sub[target].iloc[tr_i])
            pred = p.predict(sub["text"].iloc[te_i])
            acc.append(accuracy_score(sub[target].iloc[te_i], pred))
            kap.append(cohen_kappa_score(sub[target].iloc[te_i], pred))
        k = float(np.mean(kap))
        verdict = ("no usable signal" if k < 0.05 else
                   "weak" if k < 0.20 else
                   "moderate" if k < 0.40 else "strong")
        print(f"{target:20s}{len(sub):>7,}{np.mean(acc):>10.1%}"
              f"{k:>9.3f}   {verdict}")
    print("\n> Kappa near zero means the text does not predict the label, "
          "whatever\n  the accuracy figure looks like.")

    # ---- predict --------------------------------------------------------
    preds_sklearn, preds_trivial = [], []
    for i, e in enumerate(episodes):
        asked = set(e["labels_requested"])

        p, conf = {}, {}
        for target, pipe in models.items():
            if target not in asked and target != "component":
                continue
            probs = pipe.predict_proba([ep_text[i]])[0]
            j = int(np.argmax(probs))
            p[target] = pipe.classes_[j]
            conf[target] = float(probs[j])

        # The Router's job, done by the classifier: domain follows from
        # the PREDICTED component, never from the recorded one.
        if "component" in p:
            p["domain"] = DOMAIN_MAP.get(p["component"])
            conf["domain"] = conf["component"]

        preds_sklearn.append({
            "episode_id": e["episode_id"], "arm": "sklearn", "rep": 1,
            "predictions": p, "confidence": conf,
            "meta": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0},
        })

        tp = {t: v for t, v in trivial.items() if t in asked or t == "component"}
        if "component" in tp:
            tp["domain"] = DOMAIN_MAP.get(tp["component"])
        preds_trivial.append({
            "episode_id": e["episode_id"], "arm": "trivial", "rep": 1,
            "predictions": tp, "confidence": {},
            "meta": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0},
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "held_out"):
        ids = {e["episode_id"] for e in episodes if e["split"] == split}
        path = outdir / f"baselines_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in preds_sklearn + preds_trivial:
                if row["episode_id"] in ids:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"written: {path} ({len(ids)} episodes x 2 arms)")

    print("\nScore the development split:\n"
          "  python m1_score.py --predictions runs/baselines_dev.jsonl "
          "--split dev")
    print("\nLeave runs/baselines_held_out.jsonl alone until the final run.")


if __name__ == "__main__":
    main()