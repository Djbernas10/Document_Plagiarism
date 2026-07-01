"""Sanity-check the custom_300 LSA index by querying a known plagiarised doc
(suspicious-document00006, whose ground-truth source is source-document00019)
and confirming it surfaces as a top candidate.

Run from project root:
    python scripts/final/03_index_creation/verify_lsa_custom.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = PROJECT_ROOT / "scripts" / "final" / "04_source_retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

import source_retrieval_branches as srb  # noqa: E402

srb.PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_300"
srb.SUSPICIOUS_CHUNKS_PATH = srb.PROCESSED_DIR / "suspicious_chunks_lsa_esa.parquet"
srb.SOURCE_CANONICAL_CHUNKS_PATH = srb.PROCESSED_DIR / "source_chunks.parquet"
srb.ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "lsa_custom"
srb.OUTPUT_CANDIDATES_PATH = srb.PROCESSED_DIR / "lsa_candidates_suspicious_doc_custom_test.parquet"

srb.SUSPICIOUS_DOC_ID = "suspicious-document00006.txt"

result = srb.lsa_lookup()
print("\n=== LSA top candidates for suspicious-document00006.txt (expected: source-document00019.txt) ===")
print(result[["source_doc_rank", "source_doc_id", "mean_LSA_score", "max_LSA_score", "match_count"]].head(10).to_string())
