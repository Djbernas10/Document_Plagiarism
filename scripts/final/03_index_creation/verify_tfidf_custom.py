"""Sanity-check the custom_300 TF-IDF index by querying a known plagiarised doc
(suspicious-document00006, whose ground-truth source is source-document00019)
and confirming it surfaces as a top candidate.

Run from project root:
    python scripts/final/03_index_creation/verify_tfidf_custom.py
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
srb.ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_hashing_custom"
srb.OUTPUT_TOP_DOCS_MAX_PATH = srb.PROCESSED_DIR / "tfidf_top_source_documents_by_max_score_custom_test.parquet"
srb.OUTPUT_CANDIDATES_PATH = srb.PROCESSED_DIR / "tfidf_candidates_suspicious_doc_custom_test.parquet"

srb.SUSPICIOUS_DOC_ID = "suspicious-document00006.txt"

result = srb.tf_idf_lookup()
print("\n=== TF-IDF top candidates for suspicious-document00006.txt (expected: source-document00019.txt) ===")
print(result.head(10).to_string())
