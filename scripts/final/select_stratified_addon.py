"""
Select an add-on batch of suspicious documents (outside the existing 308-doc
part1-only subset) that, when combined with the 308, pulls the combined
obfuscation-category case distribution closer to the full PAN 2011 corpus's
distribution — without letting any single high-case-count document dominate
one category (which is what made the naive quota-fill pick only ~6 docs,
each one artificially "fixing" a category by itself).

Method:
    1. Classify every ground-truth case (span) into one of 5 categories:
       none / low / high / manual / translation (see classify_obfuscation).
    2. Compute the corpus-wide target % per category.
    3. Compute how many more cases of each category the current 308-doc
       subset needs to reach that target %, holding the total case count of
       the 308 subset fixed as a floor (we only ever add, never remove).
    4. Greedily select candidate documents (not already in the 308 subset)
       that are richest in the still-needed categories, but STOP counting a
       document's contribution to any one category at MAX_CASES_PER_DOC —
       so one 60-case document can't silently supply an entire category's
       quota by itself. This caps apply to what actually gets counted
       towards the running total, not just to the selection score.
    5. Stop once every category's need is met or the candidate pool is
       exhausted, and report the resulting combined distribution.

This is a targeted (non-random) selection, not a random sample — that must
be disclosed as such in the thesis. It answers a different question than
random sampling: "what is the smallest number of additional documents that
brings the case-type mix in line with the corpus, without over-relying on
any single document."

Run:
    python scripts/final/select_stratified_addon.py [--max-cases-per-doc N] [--parts-cap N]

Output:
    Prints the selected doc_id list and the resulting combined distribution.
    Does NOT run the pipeline — feed the printed doc_ids to
    run_pipeline.py --doc-id <id> (one at a time) or extend run_pipeline.py
    to accept an explicit doc_id list.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_SPANS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_ground_truth" / "pan2011_plagiarism_spans.parquet"
SUSPICIOUS_DOCS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300" / "suspicious_documents.parquet"

EXISTING_SUBSET_SIZE = 308
CATEGORIES = ["low", "high", "translation", "manual", "none"]


def classify_obfuscation(row) -> str:
    ptype = row["plagiarism_type"]
    obf = row["obfuscation"]
    if ptype == "artificial" and obf == "none":
        return "none"
    if ptype == "artificial" and obf == "low":
        return "low"
    if ptype == "artificial" and obf == "high":
        return "high"
    if ptype == "simulated":
        return "manual"
    if ptype == "translation":
        return "translation"
    return "other"


def select_addon(gt: pd.DataFrame, existing_doc_ids: set[str], max_cases_per_doc: int) -> tuple[list[str], pd.Series]:
    cur_counts = gt[gt["suspicious_doc_id"].isin(existing_doc_ids)]["cat"].value_counts()
    cur_counts = cur_counts.reindex(CATEGORIES, fill_value=0)

    target_pct = gt["cat"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0.0)

    # Minimal final total N such that every category's current count already
    # fits under its target share (can only add, never remove).
    n_candidates = cur_counts / target_pct.replace(0, pd.NA)
    n_final = n_candidates.max()
    needed = (target_pct * n_final - cur_counts).clip(lower=0)

    pool = gt[~gt["suspicious_doc_id"].isin(existing_doc_ids)]
    doc_mix = pool.groupby(["suspicious_doc_id", "cat"]).size().unstack(fill_value=0)
    for c in CATEGORIES:
        if c not in doc_mix.columns:
            doc_mix[c] = 0
    doc_mix = doc_mix[CATEGORIES]

    # Drop any candidate whose REAL per-category count exceeds the cap for
    # a category that's still needed — a 60-manual-case doc is excluded
    # outright rather than truncated, so it can never single-handedly
    # satisfy (and overshoot) a category's quota.
    over_cap = (doc_mix > max_cases_per_doc)
    eligible_mask = ~(over_cap & (doc_mix > 0)).any(axis=1)
    eligible = doc_mix[eligible_mask]

    score = eligible[["manual", "none", "translation", "low"]].sum(axis=1) - eligible["high"] * 1.5
    ordered_doc_ids = score.sort_values(ascending=False).index

    remaining = needed.copy()
    selected: list[str] = []

    for doc_id in ordered_doc_ids:
        if (remaining <= 0).all():
            break
        row = eligible.loc[doc_id]
        # Only take a doc if it contributes to a category that STILL has
        # remaining need, and only while every one of its contributions
        # stays within what's still needed (no overshoot allowed).
        contributes = (row > 0) & (remaining > 0)
        if not contributes.any():
            continue
        if (row[contributes] > remaining[contributes]).any():
            continue
        selected.append(doc_id)
        remaining = (remaining - row).clip(lower=0)

    actual_added = gt[gt["suspicious_doc_id"].isin(selected)]["cat"].value_counts().reindex(CATEGORIES, fill_value=0)

    return selected, actual_added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases-per-doc", type=int, default=8,
                         help="Cap on how many cases of one category a single document may contribute toward the quota during selection (default: 8)")
    args = parser.parse_args()

    gt = pd.read_parquet(GT_SPANS_PATH)
    gt["cat"] = gt.apply(classify_obfuscation, axis=1)
    gt = gt[gt["cat"] != "other"]

    susp_docs = pd.read_parquet(SUSPICIOUS_DOCS_PATH)
    existing_doc_ids = set(susp_docs["doc_id"].tolist()[:EXISTING_SUBSET_SIZE])

    selected, actual_added = select_addon(gt, existing_doc_ids, args.max_cases_per_doc)

    cur_counts = gt[gt["suspicious_doc_id"].isin(existing_doc_ids)]["cat"].value_counts().reindex(CATEGORIES, fill_value=0)
    combined = cur_counts.add(actual_added, fill_value=0)
    combined_pct = (combined / combined.sum() * 100).round(1)
    target_pct = (gt["cat"].value_counts(normalize=True) * 100).round(1).reindex(CATEGORIES)

    print(f"max_cases_per_doc = {args.max_cases_per_doc}")
    print(f"Existing subset   : {EXISTING_SUBSET_SIZE} docs, {int(cur_counts.sum())} cases")
    print(f"Add-on selected   : {len(selected)} docs, {int(actual_added.sum())} cases")
    print(f"Combined total    : {EXISTING_SUBSET_SIZE + len(selected)} docs, {int(combined.sum())} cases")
    print()
    print(pd.DataFrame({"combined_%": combined_pct, "corpus_target_%": target_pct}))
    print()

    parts = [d.split("__")[0] for d in selected]
    print("Parts represented in add-on:", pd.Series(parts).value_counts().to_dict())
    print()
    print("Selected doc_ids:")
    for doc_id in selected:
        print(f"  {doc_id}")


if __name__ == "__main__":
    main()
