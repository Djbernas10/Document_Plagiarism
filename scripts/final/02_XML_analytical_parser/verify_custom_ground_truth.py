"""Verify custom_plagiarism_spans.parquet offsets against the actual clean_text
stored in datasets/processed/custom_300/source_documents.parquet and
suspicious_documents.parquet (the same clean_text chunk offsets reference).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GT_PATH = PROJECT_ROOT / "datasets" / "processed" / "custom_ground_truth" / "custom_plagiarism_spans.parquet"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_300"

gt_df = pd.read_parquet(GT_PATH)
susp_docs = pd.read_parquet(PROCESSED_DIR / "suspicious_documents.parquet").set_index("doc_id")
src_docs = pd.read_parquet(PROCESSED_DIR / "source_documents.parquet").set_index("doc_id")

for _, row in gt_df.iterrows():
    susp_clean = susp_docs.loc[row["suspicious_doc_id"], "clean_text"]
    src_clean = src_docs.loc[row["source_doc_id"], "clean_text"]

    susp_slice = susp_clean[row["suspicious_offset"]:row["suspicious_end"]]
    src_slice = src_clean[row["source_offset"]:row["source_end"]]

    print(f"{row['suspicious_doc_id']} -> {row['source_doc_id']}  [{row['plagiarism_type']}/{row['obfuscation']}]")
    print(f"  SUSP: {susp_slice[:90]!r}")
    print(f"  SRC : {src_slice[:90]!r}")
    print()
