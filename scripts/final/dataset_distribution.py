"""
Reproduce the PAN 2011 "Plagiarism Case Statistics" table (obfuscation type
breakdown + case-length breakdown, cf. Potthast et al.) from the ground-truth
Parquet, for both the full corpus and the 308-document evaluation subset used
in run_pipeline.py (the first 308 doc_ids in datasets/processed/PAN2011_300).

A "case" here = one <feature name="plagiarism"> element in a suspicious-document
XML file, i.e. one row of pan2011_plagiarism_spans.parquet (one plagiarised
passage pair). Case length = suspicious_length in words (word count of the
plagiarised passage in the suspicious document), matching the PAN definition
of short/medium/long.

Note: the PAN paper additionally splits "translation" into automatic vs.
automatic+manual-correction. That distinction is NOT encoded anywhere in this
project's parsed XML attributes (type/obfuscation/language only), so it cannot
be reconstructed from pan2011_plagiarism_spans.parquet. This script reports
"translation" as a single combined category and flags this limitation.

Run:
    python scripts/final/dataset_distribution.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_SPANS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_ground_truth" / "pan2011_plagiarism_spans.parquet"
SUSPICIOUS_DOCS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300" / "suspicious_documents.parquet"

SUBSET_SIZE = 308

# PAN case-length bands (word count of the plagiarised passage)
SHORT_MAX = 150
MEDIUM_MAX = 1150


def classify_obfuscation(row) -> str:
    ptype = row["plagiarism_type"]
    obf = row["obfuscation"]
    if ptype == "artificial" and obf == "none":
        return "none"
    if ptype == "artificial" and obf == "low":
        return "paraphrasing - automatic (low)"
    if ptype == "artificial" and obf == "high":
        return "paraphrasing - automatic (high)"
    if ptype == "simulated":
        return "paraphrasing - manual"
    if ptype == "translation":
        return "translation (automatic + manual correction not distinguishable)"
    return f"unknown ({ptype}/{obf})"


def classify_length(word_count: float) -> str:
    if pd.isna(word_count):
        return "unknown"
    if word_count < SHORT_MAX:
        return "short (<150 words)"
    if word_count <= MEDIUM_MAX:
        return "medium (150-1150 words)"
    return "long (>1150 words)"


def summarize(spans: pd.DataFrame, label: str) -> None:
    n = len(spans)
    print(f"\n=== {label}: {n:,} plagiarism cases ===")

    print("\n-- Obfuscation --")
    obf_counts = spans["obfuscation_category"].value_counts()
    for cat, count in obf_counts.items():
        print(f"  {cat:<60} {count:>7,}  {count / n * 100:5.1f}%")

    print("\n-- Case length --")
    len_counts = spans["length_category"].value_counts()
    order = ["short (<150 words)", "medium (150-1150 words)", "long (>1150 words)", "unknown"]
    for cat in order:
        if cat in len_counts.index:
            count = len_counts[cat]
            print(f"  {cat:<30} {count:>7,}  {count / n * 100:5.1f}%")


def main() -> None:
    spans = pd.read_parquet(GT_SPANS_PATH)
    spans["obfuscation_category"] = spans.apply(classify_obfuscation, axis=1)
    # PAN case length is defined on word count; approximate suspicious-side
    # word count from suspicious_length (chars) using ~5.5 chars/word (mean
    # English word length + space) since the ground-truth table only stores
    # char offsets. This is an approximation, not the exact PAN word count.
    spans["suspicious_word_count_approx"] = spans["suspicious_length"] / 5.5
    spans["length_category"] = spans["suspicious_word_count_approx"].apply(classify_length)

    summarize(spans, "Full PAN 2011 corpus (this project's parsed ground truth)")

    susp_docs = pd.read_parquet(SUSPICIOUS_DOCS_PATH)
    subset_doc_ids = set(susp_docs["doc_id"].tolist()[:SUBSET_SIZE])
    subset_spans = spans[spans["suspicious_doc_id"].isin(subset_doc_ids)]

    summarize(subset_spans, f"First {SUBSET_SIZE}-document subset (run_pipeline.py --docs {SUBSET_SIZE})")

    print(
        "\nNOTE: PAN's published table splits 'translation' into automatic (10%) "
        "and automatic+manual-correction (1%). The parsed XML in this project "
        "(plagiarism_type/obfuscation/this_language/source_language attributes) "
        "does not carry a manual-correction flag, so that split cannot be "
        "reproduced from pan2011_plagiarism_spans.parquet — 'translation' is "
        "reported here as one combined category."
    )
    print(
        "NOTE: case length is approximated from suspicious_length (chars) via "
        "chars/5.5 since only character offsets are stored; it will not exactly "
        "match PAN's own word-count-based bands."
    )


if __name__ == "__main__":
    main()
