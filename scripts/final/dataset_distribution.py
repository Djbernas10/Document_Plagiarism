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
automatic+manual-correction. pan2011_plagiarism_spans.parquet does not carry a
manual_obfuscation column itself, but this script re-derives it directly from
the raw PAN 2011 XML (keyed by relative_xml_path + plagiarism_index_in_xml,
both already columns in the ground-truth Parquet) so its "translation" row is
split the same way compute_plagdet.py / compute_plagdet_official.py have split
it since compute_plagdet.py commit 53bbb75 (2026-07-06) -- see load_gt_spans()
and classify_case_category() there.
(An earlier version of this script incorrectly claimed the manual_obfuscation
flag was not available anywhere in the project and reported "translation" as
one combined, unsplit category as a result; that claim was wrong from the
moment it was written -- compute_plagdet.py had already parsed and used the
flag for over two months by the time this script was written. Both the claim
and the unsplit category have since been corrected.)

Run:
    python scripts/final/dataset_distribution.py
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_SPANS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_ground_truth" / "pan2011_plagiarism_spans.parquet"
SUSPICIOUS_DOCS_PATH = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300" / "suspicious_documents.parquet"
# Base that relative_xml_path (e.g. "suspicious-document/part1/suspicious-document00047.xml")
# is relative to -- same PAN2011 XML root compute_plagdet.py's GT_XML_DIR points into.
PAN2011_XML_ROOT = PROJECT_ROOT / "datasets" / "PAN2011" / "usable"

SUBSET_SIZE = 308

# PAN case-length bands (word count of the plagiarised passage)
SHORT_MAX = 150
MEDIUM_MAX = 1150


@lru_cache(maxsize=None)
def _load_plagiarism_features(relative_xml_path: str) -> list[dict]:
    """Parse one suspicious-document XML once and cache its <feature
    name="plagiarism"> elements in document order, so manual_obfuscation can be
    looked up by plagiarism_index_in_xml without re-parsing per row."""
    tree = ET.parse(PAN2011_XML_ROOT / relative_xml_path)
    return [f.attrib for f in tree.findall('.//feature[@name="plagiarism"]')]


def _lookup_manual_obfuscation(row) -> str:
    """Re-derive the manual_obfuscation flag from the raw XML for translation
    cases, keyed by (relative_xml_path, plagiarism_index_in_xml) -- both already
    columns in pan2011_plagiarism_spans.parquet -- since the Parquet itself
    doesn't carry this attribute as a column."""
    features = _load_plagiarism_features(row["relative_xml_path"])
    idx = int(row["plagiarism_index_in_xml"]) - 1
    if 0 <= idx < len(features):
        return features[idx].get("manual_obfuscation", "false")
    return "false"


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
        manual = _lookup_manual_obfuscation(row)
        return "translation - manual correction" if manual == "true" else "translation - automatic"
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
        "\nNOTE: 'translation - automatic' vs 'translation - manual correction' "
        "above is re-derived from the manual_obfuscation attribute in the raw "
        "PAN 2011 XML (pan2011_plagiarism_spans.parquet itself doesn't carry that "
        "column), looked up per row via relative_xml_path + "
        "plagiarism_index_in_xml. This is the same flag compute_plagdet.py / "
        "compute_plagdet_official.py have used since compute_plagdet.py commit "
        "53bbb75 (2026-07-06)."
    )
    print(
        "NOTE: case length is approximated from suspicious_length (chars) via "
        "chars/5.5 since only character offsets are stored; it will not exactly "
        "match PAN's own word-count-based bands."
    )


if __name__ == "__main__":
    main()
