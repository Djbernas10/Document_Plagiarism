from pathlib import Path
from typing import Optional
import json
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# XML PARSING HELPERS
# ============================================================

def safe_int(value: Optional[str], field_name: str, xml_path: Path) -> int:
    """
    Convert XML attribute to int with a useful error message.
    """

    if value is None:
        raise ValueError(f"Missing required field '{field_name}' in XML: {xml_path}")

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for field '{field_name}' in XML {xml_path}: {value}"
        ) from exc


def build_doc_id(part: str, reference: str) -> str:
    """
    Keep doc_id format aligned with your chunk metadata.

    Example:
    part1__suspicious-document00007.txt
    part1__source-document06022.txt
    """

    return f"{part}__{reference}"


def parse_pan_xml_file(xml_path: Path, dataset_root: Path) -> list[dict]:
    """
    Parse one PAN suspicious-document XML file.

    Expected structure:
        dataset_root/
            suspicious-document/partX/suspicious-documentXXXXX.xml
            suspicious-document/partX/suspicious-documentXXXXX.txt
            source-document/partX/source-documentXXXXX.txt

    Returns one row per plagiarism feature/span.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    suspicious_reference = root.attrib.get("reference")

    if suspicious_reference is None:
        raise ValueError(f"Missing document reference in XML root: {xml_path}")

    relative_xml_path = xml_path.relative_to(dataset_root).as_posix()

    # Example:
    # suspicious-document/part1/suspicious-document00007.xml
    part = xml_path.parent.name

    suspicious_doc_id = build_doc_id(
        part=part,
        reference=suspicious_reference,
    )

    suspicious_relative_path = (
        Path("suspicious-document") / part / suspicious_reference
    ).as_posix()

    rows = []

    plagiarism_index = 0

    for feature in root.findall("feature"):
        if feature.attrib.get("name") != "plagiarism":
            continue

        plagiarism_index += 1

        source_reference = feature.attrib.get("source_reference")

        if source_reference is None:
            raise ValueError(
                f"Missing source_reference in plagiarism feature: {xml_path}"
            )

        suspicious_offset = safe_int(
            feature.attrib.get("this_offset"),
            "this_offset",
            xml_path,
        )

        suspicious_length = safe_int(
            feature.attrib.get("this_length"),
            "this_length",
            xml_path,
        )

        source_offset = safe_int(
            feature.attrib.get("source_offset"),
            "source_offset",
            xml_path,
        )

        source_length = safe_int(
            feature.attrib.get("source_length"),
            "source_length",
            xml_path,
        )

        suspicious_end = suspicious_offset + suspicious_length
        source_end = source_offset + source_length

        source_doc_id = build_doc_id(
            part=part,
            reference=source_reference,
        )

        source_relative_path = (
            Path("source-document") / part / source_reference
        ).as_posix()

        rows.append(
            {
                # XML metadata
                "xml_path": str(xml_path),
                "relative_xml_path": relative_xml_path,
                "part": part,
                "plagiarism_index_in_xml": plagiarism_index,

                # suspicious document
                "suspicious_reference": suspicious_reference,
                "suspicious_doc_id": suspicious_doc_id,
                "suspicious_relative_path": suspicious_relative_path,

                # suspicious passage/span
                "suspicious_offset": suspicious_offset,
                "suspicious_length": suspicious_length,
                "suspicious_end": suspicious_end,
                "suspicious_range": f"{suspicious_offset}-{suspicious_end}",

                # source document
                "source_reference": source_reference,
                "source_doc_id": source_doc_id,
                "source_relative_path": source_relative_path,

                # source passage/span
                "source_offset": source_offset,
                "source_length": source_length,
                "source_end": source_end,
                "source_range": f"{source_offset}-{source_end}",

                # PAN metadata
                "plagiarism_type": feature.attrib.get("type"),
                "obfuscation": feature.attrib.get("obfuscation"),
                "this_language": feature.attrib.get("this_language"),
                "source_language": feature.attrib.get("source_language"),
            }
        )

    return rows


# ============================================================
# COLLECT SPAN-LEVEL GROUND TRUTH
# ============================================================

def collect_pan_plagiarism_spans(
    dataset_root: Path,
    output_path: Path,
    save_csv: bool = False,
) -> pd.DataFrame:
    """
    Collect all PAN XML plagiarism annotations.

    Output is span-level:
    one row = one plagiarized passage pair.
    """

    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suspicious_root = dataset_root / "suspicious-document"

    xml_files = sorted(
        suspicious_root.glob("part*/suspicious-document*.xml")
    )

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found under: {suspicious_root}/part*/"
        )

    all_rows = []

    for xml_path in tqdm(xml_files, desc="Parsing PAN XML files"):
        rows = parse_pan_xml_file(
            xml_path=xml_path,
            dataset_root=dataset_root,
        )
        all_rows.extend(rows)

    spans_df = pd.DataFrame(all_rows)

    if not spans_df.empty:
        spans_df = spans_df.sort_values(
            [
                "part",
                "suspicious_reference",
                "suspicious_offset",
                "source_reference",
                "source_offset",
            ]
        ).reset_index(drop=True)

    spans_df.to_parquet(output_path, index=False)

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        spans_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved CSV copy to: {csv_path}")

    print(f"Parsed XML files: {len(xml_files):,}")
    print(f"Plagiarism spans found: {len(spans_df):,}")
    print(f"Saved span-level ground truth to: {output_path}")

    return spans_df


# ============================================================
# DOCUMENT-LEVEL SOURCE VALIDATION TABLE
# ============================================================

def list_to_json(values: list) -> str:
    """
    Store list values as JSON strings.

    Parquet can store lists, but JSON strings are easier to inspect,
    merge, and debug across tools.
    """

    return json.dumps(values, ensure_ascii=False)


def build_source_doc_validation_table(
    spans_df: pd.DataFrame,
    output_path: Path,
    save_csv: bool = False,
) -> pd.DataFrame:
    """
    Build document-level validation table.

    One row = one suspicious document + one true source document.

    This is the table you use to validate retrieval.

    Example output:
        suspicious_doc_id
        source_doc_id
        true_plagiarism_passage_count
        suspicious_ranges_json
        source_ranges_json
        suspicious_total_plagiarized_chars
        source_total_plagiarized_chars
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if spans_df.empty:
        validation_df = pd.DataFrame()
        validation_df.to_parquet(output_path, index=False)
        return validation_df

    required_columns = {
        "part",
        "suspicious_reference",
        "suspicious_doc_id",
        "suspicious_relative_path",
        "source_reference",
        "source_doc_id",
        "source_relative_path",
        "suspicious_offset",
        "suspicious_length",
        "suspicious_end",
        "suspicious_range",
        "source_offset",
        "source_length",
        "source_end",
        "source_range",
        "plagiarism_type",
        "obfuscation",
    }

    missing = required_columns - set(spans_df.columns)
    if missing:
        raise ValueError(f"Missing columns in spans_df: {missing}")

    grouped_rows = []

    group_columns = [
        "part",
        "suspicious_reference",
        "suspicious_doc_id",
        "suspicious_relative_path",
        "source_reference",
        "source_doc_id",
        "source_relative_path",
    ]

    grouped = spans_df.groupby(group_columns, dropna=False)

    for group_key, group_df in grouped:
        (
            part,
            suspicious_reference,
            suspicious_doc_id,
            suspicious_relative_path,
            source_reference,
            source_doc_id,
            source_relative_path,
        ) = group_key

        group_df = group_df.sort_values(
            ["suspicious_offset", "source_offset"]
        ).reset_index(drop=True)

        suspicious_ranges = [
            {
                "offset": int(row.suspicious_offset),
                "length": int(row.suspicious_length),
                "end": int(row.suspicious_end),
                "range": row.suspicious_range,
            }
            for row in group_df.itertuples(index=False)
        ]

        source_ranges = [
            {
                "offset": int(row.source_offset),
                "length": int(row.source_length),
                "end": int(row.source_end),
                "range": row.source_range,
            }
            for row in group_df.itertuples(index=False)
        ]

        aligned_passages = [
            {
                "suspicious_offset": int(row.suspicious_offset),
                "suspicious_length": int(row.suspicious_length),
                "suspicious_end": int(row.suspicious_end),
                "suspicious_range": row.suspicious_range,

                "source_offset": int(row.source_offset),
                "source_length": int(row.source_length),
                "source_end": int(row.source_end),
                "source_range": row.source_range,

                "plagiarism_type": row.plagiarism_type,
                "obfuscation": row.obfuscation,
            }
            for row in group_df.itertuples(index=False)
        ]

        plagiarism_types = sorted(
            value for value in group_df["plagiarism_type"].dropna().unique().tolist()
        )

        obfuscations = sorted(
            value for value in group_df["obfuscation"].dropna().unique().tolist()
        )

        grouped_rows.append(
            {
                "part": part,

                # suspicious document
                "suspicious_reference": suspicious_reference,
                "suspicious_doc_id": suspicious_doc_id,
                "suspicious_relative_path": suspicious_relative_path,

                # true source document
                "source_reference": source_reference,
                "source_doc_id": source_doc_id,
                "source_relative_path": source_relative_path,

                # validation target
                "is_true_source_document": True,
                "true_plagiarism_passage_count": int(len(group_df)),

                # suspicious-side aggregate spans
                "suspicious_min_offset": int(group_df["suspicious_offset"].min()),
                "suspicious_max_end": int(group_df["suspicious_end"].max()),
                "suspicious_total_plagiarized_chars": int(
                    group_df["suspicious_length"].sum()
                ),
                "suspicious_ranges_json": list_to_json(suspicious_ranges),

                # source-side aggregate spans
                "source_min_offset": int(group_df["source_offset"].min()),
                "source_max_end": int(group_df["source_end"].max()),
                "source_total_plagiarized_chars": int(
                    group_df["source_length"].sum()
                ),
                "source_ranges_json": list_to_json(source_ranges),

                # aligned suspicious/source passage pairs
                "aligned_passages_json": list_to_json(aligned_passages),

                # PAN metadata summary
                "plagiarism_types_json": list_to_json(plagiarism_types),
                "obfuscations_json": list_to_json(obfuscations),
            }
        )

    validation_df = pd.DataFrame(grouped_rows)

    validation_df = validation_df.sort_values(
        [
            "part",
            "suspicious_reference",
            "source_reference",
        ]
    ).reset_index(drop=True)

    validation_df.to_parquet(output_path, index=False)

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        validation_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved CSV copy to: {csv_path}")

    print(f"Validation rows: {len(validation_df):,}")
    print(f"Saved document-level validation table to: {output_path}")

    return validation_df


# ============================================================
# OPTIONAL: FILTER ONE SUSPICIOUS DOCUMENT
# ============================================================

def get_ground_truth_for_suspicious_doc(
    validation_df: pd.DataFrame,
    suspicious_doc_id: str,
) -> pd.DataFrame:
    """
    Return true source documents for one suspicious document.

    Useful for checking whether your retrieval top-N contains the correct source.
    """

    filtered_df = validation_df[
        validation_df["suspicious_doc_id"] == suspicious_doc_id
    ].copy()

    filtered_df = filtered_df.sort_values(
        [
            "true_plagiarism_passage_count",
            "suspicious_total_plagiarized_chars",
            "source_reference",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return filtered_df


# ============================================================
# OPTIONAL: VALIDATE RETRIEVED TOP DOCUMENTS
# ============================================================

def validate_retrieved_source_documents(
    retrieved_docs_df: pd.DataFrame,
    ground_truth_validation_df: pd.DataFrame,
    suspicious_doc_id: str,
    retrieved_source_doc_column: str = "source_doc_id",
    retrieved_rank_column: str = "source_doc_rank",
) -> pd.DataFrame:
    """
    Compare retrieved top source documents against PAN ground truth.

    Works with your TF-IDF / LSA / ESA / embedding document-level outputs,
    as long as they have:
        - source_doc_id
        - source_doc_rank

    Returns retrieved rows with:
        - is_true_source_document
        - true_plagiarism_passage_count
        - suspicious/source range JSON columns
    """

    if retrieved_source_doc_column not in retrieved_docs_df.columns:
        raise ValueError(
            f"Missing retrieved source doc column: {retrieved_source_doc_column}"
        )

    if retrieved_rank_column not in retrieved_docs_df.columns:
        raise ValueError(
            f"Missing retrieved rank column: {retrieved_rank_column}"
        )

    true_sources_df = ground_truth_validation_df[
        ground_truth_validation_df["suspicious_doc_id"] == suspicious_doc_id
    ].copy()

    if true_sources_df.empty:
        print(f"No ground-truth plagiarism sources for: {suspicious_doc_id}")

    merge_columns = [
        "suspicious_doc_id",
        "source_doc_id",
        "is_true_source_document",
        "true_plagiarism_passage_count",
        "suspicious_ranges_json",
        "source_ranges_json",
        "aligned_passages_json",
        "suspicious_total_plagiarized_chars",
        "source_total_plagiarized_chars",
        "plagiarism_types_json",
        "obfuscations_json",
    ]

    true_sources_df = true_sources_df[merge_columns].copy()

    retrieved_df = retrieved_docs_df.copy()

    if "suspicious_doc_id" not in retrieved_df.columns:
        retrieved_df["suspicious_doc_id"] = suspicious_doc_id

    validated_df = retrieved_df.merge(
        true_sources_df,
        left_on=["suspicious_doc_id", retrieved_source_doc_column],
        right_on=["suspicious_doc_id", "source_doc_id"],
        how="left",
        suffixes=("", "_ground_truth"),
    )

    validated_df["is_true_source_document"] = (
        validated_df["is_true_source_document"]
        .fillna(False)
        .astype(bool)
    )

    validated_df["true_plagiarism_passage_count"] = (
        validated_df["true_plagiarism_passage_count"]
        .fillna(0)
        .astype(int)
    )

    validated_df = validated_df.sort_values(
        retrieved_rank_column
    ).reset_index(drop=True)

    return validated_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # CONFIG
    # ============================================================

    DATASET_ROOT = Path("../datasets/PAN2011/usable")

    OUTPUT_DIR = Path("../datasets/processed/PAN2011_ground_truth")

    OUTPUT_SPANS_PATH = OUTPUT_DIR / "pan2011_plagiarism_spans.parquet"
    OUTPUT_SOURCE_DOC_VALIDATION_PATH = OUTPUT_DIR / "pan2011_source_doc_validation.parquet"

    # Optional CSV outputs for quick inspection/debugging.
    SAVE_CSV = True


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spans_df = collect_pan_plagiarism_spans(
        dataset_root=DATASET_ROOT,
        output_path=OUTPUT_SPANS_PATH,
        save_csv=SAVE_CSV,
    )

    validation_df = build_source_doc_validation_table(
        spans_df=spans_df,
        output_path=OUTPUT_SOURCE_DOC_VALIDATION_PATH,
        save_csv=SAVE_CSV,
    )
