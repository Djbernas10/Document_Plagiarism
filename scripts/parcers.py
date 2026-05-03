from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm.auto import tqdm


def parse_pan_xml_file(xml_path: Path, dataset_root: Path) -> list[dict]:
    """
    Parse one PAN suspicious-document XML file.

    Expected structure:
    dataset_root/
        suspicious-document/partX/*.xml
        source-document/partX/*.txt
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    suspicious_reference = root.attrib.get("reference")

    relative_xml_path = xml_path.relative_to(dataset_root).as_posix()

    # Example:
    # datasets/PAN2011/suspicious-document/part1/suspicious-document00005.xml
    part = xml_path.parent.name

    suspicious_relative_path = (
        Path("suspicious-document") / part / suspicious_reference
    ).as_posix()

    suspicious_doc_id = f"{part}__{suspicious_reference}"

    rows = []

    for feature in root.findall("feature"):
        if feature.attrib.get("name") != "plagiarism":
            continue

        source_reference = feature.attrib.get("source_reference")

        this_offset = int(feature.attrib["this_offset"])
        this_length = int(feature.attrib["this_length"])

        source_offset = int(feature.attrib["source_offset"])
        source_length = int(feature.attrib["source_length"])

        source_relative_path = (
            Path("source-document") / part / source_reference
        ).as_posix()

        source_doc_id = f"{part}__{source_reference}"

        rows.append(
            {
                "xml_path": str(xml_path),
                "relative_xml_path": relative_xml_path,
                "part": part,

                # suspicious document
                "suspicious_reference": suspicious_reference,
                "suspicious_doc_id": suspicious_doc_id,
                "suspicious_relative_path": suspicious_relative_path,

                # suspicious span
                "suspicious_offset": this_offset,
                "suspicious_length": this_length,
                "suspicious_end": this_offset + this_length,

                # source document
                "source_reference": source_reference,
                "source_doc_id": source_doc_id,
                "source_relative_path": source_relative_path,

                # source span
                "source_offset": source_offset,
                "source_length": source_length,
                "source_end": source_offset + source_length,

                # metadata
                "plagiarism_type": feature.attrib.get("type"),
                "obfuscation": feature.attrib.get("obfuscation"),
                "this_language": feature.attrib.get("this_language"),
                "source_language": feature.attrib.get("source_language"),
            }
        )

    return rows


def collect_pan_plagiarism_annotations(
    dataset_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Collect all PAN XML plagiarism annotations.

    Reads from:
    dataset_root/suspicious-document/part*/suspicious-document*.xml
    """

    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suspicious_root = dataset_root / "suspicious-document"

    xml_files = sorted(suspicious_root.glob("part*/suspicious-document*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found under: {suspicious_root}/part*/"
        )

    all_rows = []

    for xml_path in tqdm(xml_files, desc="Parsing PAN XML files"):
        rows = parse_pan_xml_file(xml_path, dataset_root=dataset_root)
        all_rows.extend(rows)

    annotations_df = pd.DataFrame(all_rows)

    if not annotations_df.empty:
        annotations_df = annotations_df.sort_values(
            [
                "part",
                "suspicious_reference",
                "suspicious_offset",
                "source_reference",
            ]
        ).reset_index(drop=True)

    if output_path.suffix.lower() == ".parquet":
        annotations_df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        annotations_df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        raise ValueError("output_path must end with .parquet or .csv")

    print(f"Parsed XML files: {len(xml_files):,}")
    print(f"Plagiarism annotations found: {len(annotations_df):,}")
    print(f"Saved annotations to: {output_path}")

    return annotations_df
