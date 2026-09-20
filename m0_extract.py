#!/usr/bin/env python3
"""
M0 - Extract the Sourcetree Jira CSV into three Postgres tables.

  issues       one row per issue, cleaned, with derived labels
  issue_links  the 57 repeated link slots melted into long form
  label_stats  the trivial (majority-class) baseline for every label

Why it reads by COLUMN INDEX rather than name: the export repeats column
names ("Comment", "Comment", ... / "Outward issue link (Duplicate)" x N).
pandas renames these to Comment.1, Comment.2 on read, so selecting by name
is fragile. Indices are unambiguous.

Usage
-----
    pip install pandas sqlalchemy "psycopg[binary]" tqdm
    python m0_extract.py --csv data/GFG_FINAL.csv

    # preview without writing to the database:
    python m0_extract.py --csv data/GFG_FINAL.csv --dry-run
"""

import argparse
import re
import sys

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB = "postgresql+psycopg://homas:homas@localhost:5432/homas"

# Single-slot fields we want, by exact header name.
WANTED = {
    "Issue key":        "issue_key",
    "Issue id":         "issue_id",
    "Summary":          "summary",
    "Description":      "description",
    "Issue Type":       "issue_type",
    "Status":           "status",
    "Priority":         "priority",
    "Resolution":       "resolution",
    "Created":          "created",
    "Resolved":         "resolved",
    "Updated":          "updated",
    "Votes":            "votes",
    "Reporter":         "reporter",
    "Component/s":      "component",
    "Labels":           "labels",
    "Environment":      "environment",
    "Attachment":       "attachment",
    "Affects Version/s": "affects_version",
    "Fix Version/s":    "fix_version",
    "Custom field (Symptom Severity)": "severity_raw",
}

ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")

# Severity arrives under two schemes (an older one and a newer one).
# Collapse to three classes; record the decision in the thesis.
SEVERITY_MAP = {
    "severity 1 - critical": "Critical",
    "critical":              "Critical",
    "severity 2 - major":    "Major",
    "major":                 "Major",
    "severity 3 - minor":    "Minor",
    "minor":                 "Minor",
}


def build_column_plan(csv_path):
    """Map wanted headers -> column indices, and find every link slot."""
    header = pd.read_csv(csv_path, nrows=0)
    cols = header.columns.tolist()

    # pandas has already de-duplicated names here (Comment.1 etc).
    # Strip the suffix to recover the original header for matching.
    raw = [re.sub(r"\.\d+$", "", c) for c in cols]

    # This export has a stray character prepended to the first header
    # ("qSummary" instead of "Summary"). Repair any header that ends with a
    # wanted name but carries leading junk, and say so out loud.
    for i, name in enumerate(raw):
        if name in WANTED:
            continue
        for want in WANTED:
            if name.endswith(want) and len(name) - len(want) <= 3:
                print(f"  repaired header: {name!r} -> {want!r}")
                raw[i] = want
                break

    single = {}
    for i, name in enumerate(raw):
        if name in WANTED and WANTED[name] not in single:
            single[WANTED[name]] = i          # keep the FIRST slot only

    links = []
    for i, name in enumerate(raw):
        m = re.match(r"(Inward|Outward) issue link \((.+?)\)$", name)
        if m:
            links.append((i, m.group(1).lower(), m.group(2)))

    missing = set(WANTED.values()) - set(single)
    if missing:
        print(f"  note: not present in this export -> {sorted(missing)}")

    return single, links, cols


def normalise_severity(val):
    if not isinstance(val, str):
        return None
    return SEVERITY_MAP.get(val.strip().lower())


def clean_issues(df):
    df = df.copy()

    df["severity"] = df["severity_raw"].map(normalise_severity) \
        if "severity_raw" in df else None

    for c in ("created", "resolved", "updated"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed",
                                   utc=True).dt.tz_localize(None)

    if "votes" in df:
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

    if "created" in df and "resolved" in df:
        df["days_to_resolve"] = (
            (df["resolved"] - df["created"]).dt.total_seconds() / 86400
        )

    # A stray version string ("2.6.10") appears in the component field.
    if "component" in df:
        df.loc[df["component"].astype(str).str.match(r"^[\d.]+$", na=False),
               "component"] = None

    df["has_attachment"] = df["attachment"].notna() if "attachment" in df else False
    df["desc_len"] = df["description"].astype(str).str.len().where(
        df["description"].notna(), 0) if "description" in df else 0

    return df.drop(columns=[c for c in ("severity_raw", "attachment")
                            if c in df])


def melt_links(chunk_raw, keys, link_plan):
    """chunk_raw: positional DataFrame. Returns long-form link rows."""
    out = []
    for idx, direction, ltype in link_plan:
        if idx >= chunk_raw.shape[1]:
            continue
        col = chunk_raw.iloc[:, idx]
        mask = col.notna()
        if not mask.any():
            continue
        out.append(pd.DataFrame({
            "source_key": keys[mask].values,
            "direction": direction,
            "link_type": ltype,
            "target_key": col[mask].astype(str).str.strip().values,
        }))
    if not out:
        return pd.DataFrame(columns=["source_key", "direction",
                                     "link_type", "target_key"])
    links = pd.concat(out, ignore_index=True)
    return links[links["target_key"].str.match(ISSUE_KEY_RE, na=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows (for a quick trial)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("planning columns...")
    single, link_plan, all_cols = build_column_plan(args.csv)
    print(f"  {len(single)} single fields, {len(link_plan)} link slots, "
          f"{len(all_cols)} columns total")

    key_idx = single.get("issue_key")
    if key_idx is None:
        sys.exit("No 'Issue key' column found - wrong file?")

    read_idx = sorted(set(single.values()) | {i for i, _, _ in link_plan})
    pos_of = {orig: n for n, orig in enumerate(read_idx)}

    issues_all, links_all = [], []
    seen = 0

    reader = pd.read_csv(args.csv, usecols=read_idx, dtype=str,
                         chunksize=args.chunksize, header=0)

    for n, chunk in enumerate(reader):
        keys = chunk.iloc[:, pos_of[key_idx]]

        cols = {name: chunk.iloc[:, pos_of[i]] for name, i in single.items()}
        issues_all.append(clean_issues(pd.DataFrame(cols)))

        remapped = [(pos_of[i], d, t) for i, d, t in link_plan if i in pos_of]
        links_all.append(melt_links(chunk, keys, remapped))

        seen += len(chunk)
        print(f"  chunk {n + 1}: {seen:,} rows", end="\r")
        if args.limit and seen >= args.limit:
            break

    issues_raw = pd.concat(issues_all, ignore_index=True)
    issues = issues_raw.drop_duplicates("issue_key")

    # This export repeats each issue once per view event, so links repeat too.
    links = pd.concat(links_all, ignore_index=True).drop_duplicates(
        ["source_key", "direction", "link_type", "target_key"])

    factor = len(issues_raw) / max(len(issues), 1)
    if factor > 1.5:
        print(f"\n\n  !! CORPUS IS REPLICATED {factor:.0f}x "
              f"({len(issues_raw):,} rows -> {len(issues):,} unique issues)")
        print("     Row counts in the source file are NOT issue counts.")

    print(f"\nissues: {len(issues):,}   links: {len(links):,}")
    for c in ("severity", "component", "resolution"):
        if c in issues:
            v = issues[c].dropna()
            if len(v):
                top = v.value_counts(normalize=True)
                print(f"  {c:12s} {len(v):>7,} labelled  "
                      f"baseline {top.iloc[0]:.1%} ({top.index[0]})")

    if args.dry_run:
        print("\ndry run - nothing written")
        print(issues.head(3).T)
        return

    print(f"\nwriting to {args.db.split('@')[-1]} ...")
    engine = create_engine(args.db)

    issues.to_sql("issues", engine, if_exists="replace",
                  index=False, chunksize=5000)
    links.to_sql("issue_links_raw", engine, if_exists="replace",
                 index=False, chunksize=5000)

    with engine.begin() as con:
        # Keep only links whose target exists in the corpus - an unresolvable
        # target cannot be scored, so those episodes would be invalid.
        con.execute(text("DROP TABLE IF EXISTS issue_links"))
        con.execute(text("""
            CREATE TABLE issue_links AS
            SELECT l.* FROM issue_links_raw l
            JOIN issues i ON i.issue_key = l.target_key
        """))
        con.execute(text("DROP TABLE issue_links_raw"))

        con.execute(text("CREATE INDEX ON issues (issue_key)"))
        con.execute(text("CREATE INDEX ON issues (created)"))
        con.execute(text("CREATE INDEX ON issue_links (source_key)"))
        con.execute(text("CREATE INDEX ON issue_links (target_key)"))

        # Time band: terciles, so the baseline is 33.3% by construction.
        con.execute(text("ALTER TABLE issues ADD COLUMN time_band text"))
        con.execute(text("""
            WITH q AS (
              SELECT percentile_cont(0.333) WITHIN GROUP (ORDER BY days_to_resolve) t1,
                     percentile_cont(0.667) WITHIN GROUP (ORDER BY days_to_resolve) t2
              FROM issues WHERE days_to_resolve IS NOT NULL
            )
            UPDATE issues SET time_band =
              CASE WHEN days_to_resolve IS NULL THEN NULL
                   WHEN days_to_resolve <= (SELECT t1 FROM q) THEN 'fast'
                   WHEN days_to_resolve <= (SELECT t2 FROM q) THEN 'medium'
                   ELSE 'slow' END
        """))

        # Baselines, stored once so every reported result carries its floor.
        con.execute(text("DROP TABLE IF EXISTS label_stats"))
        con.execute(text("""
            CREATE TABLE label_stats (
              label text, n_labelled bigint, n_classes bigint,
              majority_class text, majority_share double precision
            )
        """))
        for col in ("severity", "component", "resolution", "time_band",
                    "priority", "issue_type"):
            con.execute(text(f"""
                INSERT INTO label_stats
                SELECT '{col}', COUNT(*), (SELECT COUNT(DISTINCT {col})
                                           FROM issues WHERE {col} IS NOT NULL),
                       MODE() WITHIN GROUP (ORDER BY {col}),
                       COUNT(*) FILTER (WHERE {col} = (
                         SELECT MODE() WITHIN GROUP (ORDER BY {col}) FROM issues
                       ))::float / NULLIF(COUNT(*), 0)
                FROM issues WHERE {col} IS NOT NULL
            """))

        rows = con.execute(text("""
            SELECT label, n_labelled, n_classes, majority_class, majority_share
            FROM label_stats ORDER BY n_labelled DESC
        """)).fetchall()

    print("\n=== BASELINES (every result you report sits against these) ===\n")
    print(f"{'label':14s}{'n':>9s}{'classes':>9s}  {'majority':22s}{'baseline':>9s}")
    for label, n, k, cls, share in rows:
        print(f"{label:14s}{n:>9,}{k:>9}  {str(cls)[:20]:22s}{share:>8.1%}")

    print("\nDone. Tables: issues, issue_links, label_stats")


if __name__ == "__main__":
    main()