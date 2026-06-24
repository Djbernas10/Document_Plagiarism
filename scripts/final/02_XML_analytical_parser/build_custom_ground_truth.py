"""Build the ground-truth span Parquet for the custom_dataset corpus, mirroring
pan2011_plagiarism_spans.parquet's schema so run_pipeline.py can evaluate against
it unchanged via --dataset custom.

Unlike the PAN XML parser, this dataset has no part*/ subfolders (flat
source_documents/ and suspicious_documents/ directories), and critically the
GT XML offsets were computed against the *raw* .txt files, which differ from the
*clean_text* offsets used by chunk start_char/end_char (clean_pan_source_text
collapses whitespace and normalizes quotes/dashes, shifting positions). This
script re-locates each ground-truth passage by exact text search inside the
cleaned text, so the resulting offsets line up with the chunk Parquet produced
by 01_preprocessing.

Run from project root:
    python scripts/final/02_XML_analytical_parser/build_custom_ground_truth.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "final" / "01_preprocessing"))
from preprocessing import clean_pan_source_text  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "datasets" / "custom_dataset"
GT_XML_DIR = DATASET_ROOT / "ground_truth"
SUSPICIOUS_DIR = DATASET_ROOT / "suspicious_documents"
SOURCE_DIR = DATASET_ROOT / "source_documents"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_ground_truth"
OUTPUT_SPANS_PATH = OUTPUT_DIR / "custom_plagiarism_spans.parquet"

_clean_text_cache: dict[str, str] = {}


def get_clean_text(doc_dir: Path, doc_filename: str) -> str:
    if doc_filename not in _clean_text_cache:
        raw_text = (doc_dir / doc_filename).read_text(encoding="utf-8")
        _clean_text_cache[doc_filename] = clean_pan_source_text(raw_text)
    return _clean_text_cache[doc_filename]


def locate_in_clean_text(clean_text: str, raw_text: str, raw_offset: int, raw_length: int, label: str) -> tuple[int, int]:
    """
    Re-locate a passage (known from the raw text at [raw_offset:raw_offset+raw_length])
    inside clean_text by exact substring search. This is robust to clean_pan_source_text's
    whitespace collapsing because the search string itself is taken verbatim from this
    document's raw text, then we just find where that same passage's *start* landed.
    """
    raw_passage = raw_text[raw_offset:raw_offset + raw_length]
    # Use a long-enough unique anchor (first ~80 chars) to find the start position
    # robustly even if internal whitespace shifted within the passage.
    anchor = raw_passage[:80].strip()
    if not anchor:
        raise ValueError(f"Empty anchor for {label} at offset {raw_offset}")

    clean_anchor = clean_pan_source_text(anchor)
    start = clean_text.find(clean_anchor)
    if start == -1:
        # Fall back to a shorter anchor in case cleaning altered the head of the passage.
        clean_anchor = clean_pan_source_text(anchor[:40])
        start = clean_text.find(clean_anchor)
    if start == -1:
        raise ValueError(f"Could not locate {label} anchor in clean text: {clean_anchor[:60]!r}")

    clean_length = len(clean_pan_source_text(raw_passage))
    end = start + clean_length
    return start, end


def parse_one_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    suspicious_reference = root.attrib.get("reference")
    if suspicious_reference is None:
        raise ValueError(f"Missing document reference in XML root: {xml_path}")

    susp_raw_text = (SUSPICIOUS_DIR / suspicious_reference).read_text(encoding="utf-8")
    susp_clean_text = get_clean_text(SUSPICIOUS_DIR, suspicious_reference)

    rows = []
    for feature in root.findall("feature"):
        if feature.attrib.get("name") != "plagiarism":
            continue

        source_reference = feature.attrib.get("source_reference")
        if source_reference is None:
            raise ValueError(f"Missing source_reference in plagiarism feature: {xml_path}")

        this_offset = int(feature.attrib["this_offset"])
        this_length = int(feature.attrib["this_length"])
        source_offset = int(feature.attrib["source_offset"])
        source_length = int(feature.attrib["source_length"])

        src_raw_text = (SOURCE_DIR / source_reference).read_text(encoding="utf-8")
        src_clean_text = get_clean_text(SOURCE_DIR, source_reference)

        susp_start, susp_end = locate_in_clean_text(
            susp_clean_text, susp_raw_text, this_offset, this_length,
            label=f"{suspicious_reference} this_offset={this_offset}",
        )
        src_start, src_end = locate_in_clean_text(
            src_clean_text, src_raw_text, source_offset, source_length,
            label=f"{source_reference} source_offset={source_offset}",
        )

        rows.append({
            "suspicious_doc_id": suspicious_reference,
            "source_doc_id": source_reference,
            "suspicious_offset": susp_start,
            "suspicious_length": susp_end - susp_start,
            "suspicious_end": susp_end,
            "source_offset": src_start,
            "source_length": src_end - src_start,
            "source_end": src_end,
            "plagiarism_type": feature.attrib.get("type"),
            "obfuscation": feature.attrib.get("obfuscation"),
        })

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(GT_XML_DIR.glob("suspicious-document*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No ground-truth XML files found under: {GT_XML_DIR}")

    all_rows = []
    for xml_path in xml_files:
        all_rows.extend(parse_one_xml(xml_path))

    spans_df = pd.DataFrame(all_rows)
    if not spans_df.empty:
        spans_df = spans_df.sort_values(
            ["suspicious_doc_id", "suspicious_offset", "source_doc_id", "source_offset"]
        ).reset_index(drop=True)

    spans_df.to_parquet(OUTPUT_SPANS_PATH, index=False)
    print(f"Parsed XML files: {len(xml_files)}")
    print(f"Plagiarism spans found: {len(spans_df)}")
    print(f"Saved span-level ground truth to: {OUTPUT_SPANS_PATH}")


if __name__ == "__main__":
    main()
