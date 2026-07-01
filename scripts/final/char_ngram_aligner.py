"""
Character N-gram Aligner — post-LLM span extension.

PURPOSE
-------
After the LLM confirms a source document, the pipeline only detects spans
where embedding chunk pairs scored highly. This misses plagiarised passages
where the embedding similarity dropped (e.g. heavily obfuscated passages,
or passages whose chunks didn't align well with the source chunks).

This aligner runs AFTER LLM confirmation. For each confirmed (suspicious_doc,
source_doc) pair, it slides a window over both documents and computes Jaccard
character trigram similarity. Windows above a threshold are added as additional
detected spans — extending coverage without re-running the LLM.

WHY CHAR TRIGRAMS WORK ON OBFUSCATED TEXT
------------------------------------------
Word-salad obfuscation (obfuscation=high) destroys word order but preserves
individual characters. Character trigrams capture sub-word patterns that
survive scrambling better than word-level or embedding similarity.

Synonym-swap obfuscation (obfuscation=low) changes content words but preserves
sentence structure. Char trigrams of common function words and punctuation
patterns still match across synonym-swapped text.

ALGORITHM
---------
1. Load confirmed spans from per_doc/<suspicious_doc_id>.parquet
2. For each unique (suspicious_doc_id, source_doc_id) pair in confirmed spans:
   a. Load raw text of both documents
   b. Slide a window of W characters over the suspicious doc
   c. For each suspicious window, find the best-matching source window
      using Jaccard char trigram similarity
   d. If Jaccard >= threshold AND the window doesn't already overlap an
      existing confirmed span → add as a new span
3. Merge new spans with existing confirmed spans
4. Save extended spans to per_doc/<suspicious_doc_id>_extended.parquet

PARAMETERS
----------
--window     : Window size in characters (default: 800)
--stride     : Stride between windows in characters (default: 400, = 50% overlap)
--threshold  : Jaccard threshold for new span (default: 0.12)
--merge-gap  : Max gap in chars to merge adjacent new spans (default: 1800)
--doc-id     : Process single doc (for testing)
--all        : Process all docs that have confirmed spans

USAGE
-----
# Test on single doc
uv run python scripts/final/char_ngram_aligner.py --doc-id part1__suspicious-document00012.txt

# Run on all docs with confirmed spans
uv run python scripts/final/char_ngram_aligner.py --all

# Custom threshold
uv run python scripts/final/char_ngram_aligner.py --all --threshold 0.15 --window 1000
"""

import argparse
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "pipeline_results"
PER_DOC_DIR = RESULTS_DIR / "per_doc"
PROCESSED_DIR = SCRIPT_DIR.parents[1] / "datasets" / "processed" / "PAN2011_300"

# ---------------------------------------------------------------------------
# Char trigram Jaccard
# ---------------------------------------------------------------------------

def char_trigrams(text: str) -> set:
    """Generate character trigram set from text."""
    text = text.lower()
    return {text[i:i+3] for i in range(len(text) - 2)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Core aligner
# ---------------------------------------------------------------------------

def align_doc_pair(
    susp_text: str,
    src_text: str,
    existing_spans: list[tuple[int, int]],  # (start, end) on suspicious side
    window: int = 800,
    stride: int = 400,
    threshold: float = 0.12,
) -> list[dict]:
    """
    Slide a window over susp_text, find best-matching source window,
    return new spans not already covered by existing_spans.
    """
    new_spans = []

    # Pre-compute source windows
    src_windows = []
    for src_start in range(0, max(1, len(src_text) - window + 1), stride):
        src_end = min(src_start + window, len(src_text))
        src_windows.append((src_start, src_end, char_trigrams(src_text[src_start:src_end])))

    if not src_windows:
        return []

    for susp_start in range(0, max(1, len(susp_text) - window + 1), stride):
        susp_end = min(susp_start + window, len(susp_text))
        susp_chunk = susp_text[susp_start:susp_end]
        susp_tris = char_trigrams(susp_chunk)

        # Check if already covered by an existing confirmed span
        already_covered = any(
            s <= susp_start and susp_end <= e
            for s, e in existing_spans
        )
        if already_covered:
            continue

        # Find best matching source window
        best_j = 0.0
        best_src_start = 0
        best_src_end = 0
        for src_start, src_end, src_tris in src_windows:
            j = jaccard(susp_tris, src_tris)
            if j > best_j:
                best_j = j
                best_src_start = src_start
                best_src_end = src_end

        if best_j >= threshold:
            new_spans.append({
                "suspicious_start_char": susp_start,
                "suspicious_end_char":   susp_end,
                "source_start_char":     best_src_start,
                "source_end_char":       best_src_end,
                "jaccard_score":         round(best_j, 4),
            })

    return new_spans


def merge_spans(spans: list[dict], gap: int = 1800) -> list[dict]:
    """Merge adjacent spans on the suspicious side."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x["suspicious_start_char"])
    merged = [spans[0].copy()]
    for s in spans[1:]:
        if s["suspicious_start_char"] - merged[-1]["suspicious_end_char"] <= gap:
            merged[-1]["suspicious_end_char"] = max(merged[-1]["suspicious_end_char"],
                                                     s["suspicious_end_char"])
            merged[-1]["source_start_char"]   = min(merged[-1]["source_start_char"],
                                                     s["source_start_char"])
            merged[-1]["source_end_char"]     = max(merged[-1]["source_end_char"],
                                                     s["source_end_char"])
            if s.get("jaccard_score", 0) > merged[-1].get("jaccard_score", 0):
                merged[-1]["jaccard_score"] = s["jaccard_score"]
        else:
            merged.append(s.copy())
    return merged


# ---------------------------------------------------------------------------
# Per-doc processing
# ---------------------------------------------------------------------------

def process_doc(
    doc_id: str,
    susp_docs: pd.DataFrame,
    src_docs: pd.DataFrame,
    window: int,
    stride: int,
    threshold: float,
    merge_gap: int,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Extend confirmed spans for one suspicious doc using char n-gram aligner.
    Returns extended DataFrame or None if no confirmed spans exist.
    """
    per_doc_path = PER_DOC_DIR / f"{doc_id}.parquet"
    if not per_doc_path.exists():
        return None

    confirmed = pd.read_parquet(per_doc_path)
    if confirmed.empty:
        return confirmed  # clean doc, nothing to extend

    # Load suspicious doc raw text
    susp_row = susp_docs[susp_docs["doc_id"] == doc_id]
    if susp_row.empty:
        if verbose:
            print(f"  [WARN] Suspicious doc text not found: {doc_id}")
        return None
    susp_text = susp_row.iloc[0]["raw_text"]

    t0 = time.time()
    all_new_spans = []

    # Only extend the dominant source doc (most total confirmed chars) to avoid
    # amplifying LLM false positives that confirmed the wrong source doc.
    coverage_per_src = (
        confirmed.groupby("source_doc_id")
        .apply(lambda g: (g["suspicious_end_char"] - g["suspicious_start_char"]).sum(),
               include_groups=False)
        .sort_values(ascending=False)
    )
    dominant_source = coverage_per_src.index[0]

    for source_doc_id, grp in confirmed.groupby("source_doc_id"):
        if source_doc_id != dominant_source:
            continue
        # Load source doc raw text
        src_row = src_docs[src_docs["doc_id"] == source_doc_id]
        if src_row.empty:
            continue
        src_text = src_row.iloc[0]["raw_text"]

        # Existing spans for this source doc
        existing = list(zip(
            grp["suspicious_start_char"].tolist(),
            grp["suspicious_end_char"].tolist()
        ))

        new_spans = align_doc_pair(
            susp_text, src_text, existing,
            window=window, stride=stride, threshold=threshold
        )

        for s in new_spans:
            s["suspicious_doc_id"] = doc_id
            s["source_doc_id"]     = source_doc_id
            s["suspicious_chunk_id"] = f"{doc_id}_ngram"
            s["source_chunk_id"]     = f"{source_doc_id}_ngram"
            s["embedding_score"]     = s["jaccard_score"]

        all_new_spans.extend(new_spans)

    elapsed = round(time.time() - t0, 1)

    if not all_new_spans:
        if verbose:
            print(f"  {doc_id}: no new spans found  ({elapsed}s)")
        return confirmed

    new_df = pd.DataFrame(all_new_spans)

    # Merge new spans per source doc
    merged_new = []
    for source_doc_id, grp in new_df.groupby("source_doc_id"):
        spans = grp.to_dict("records")
        merged = merge_spans(spans, gap=merge_gap)
        merged_new.extend(merged)

    new_merged_df = pd.DataFrame(merged_new)

    # Fill missing columns to match confirmed schema
    for col in confirmed.columns:
        if col not in new_merged_df.columns:
            new_merged_df[col] = None

    extended = pd.concat(
        [confirmed, new_merged_df[confirmed.columns]],
        ignore_index=True
    )

    if verbose:
        print(f"  {doc_id}: +{len(merged_new)} new span(s) from char n-gram  ({elapsed}s)")

    return extended


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Char n-gram post-LLM span aligner")
    parser.add_argument("--doc-id",    type=str,   default=None,  help="Process single doc ID")
    parser.add_argument("--all",       action="store_true",        help="Process all docs with confirmed spans")
    parser.add_argument("--window",    type=int,   default=800,   help="Window size in chars (default: 800)")
    parser.add_argument("--stride",    type=int,   default=400,   help="Stride in chars (default: 400)")
    parser.add_argument("--threshold", type=float, default=0.12,  help="Jaccard threshold (default: 0.12)")
    parser.add_argument("--merge-gap", type=int,   default=1800,  help="Merge gap in chars (default: 1800)")
    parser.add_argument("--save",      action="store_true",        help="Save extended spans to per_doc/<doc>_extended.parquet")
    args = parser.parse_args()

    if not args.doc_id and not args.all:
        parser.error("Specify --doc-id <id> or --all")

    print("Loading document texts...")
    susp_docs = pd.read_parquet(PROCESSED_DIR / "suspicious_documents.parquet",
                                columns=["doc_id", "raw_text"])
    src_docs  = pd.read_parquet(PROCESSED_DIR / "source_documents.parquet",
                                columns=["doc_id", "raw_text"])
    print(f"  {len(susp_docs)} suspicious docs, {len(src_docs)} source docs loaded")

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        # All docs that have non-empty confirmed spans
        doc_ids = []
        for f in sorted(PER_DOC_DIR.glob("*.parquet")):
            if "_extended" in f.name:
                continue
            df = pd.read_parquet(f)
            if not df.empty:
                doc_ids.append(f.stem)
        print(f"  {len(doc_ids)} docs with confirmed spans")

    print(f"\nParams: window={args.window}  stride={args.stride}  "
          f"threshold={args.threshold}  merge_gap={args.merge_gap}")
    print()

    total_new = 0
    for doc_id in doc_ids:
        extended = process_doc(
            doc_id, susp_docs, src_docs,
            window=args.window,
            stride=args.stride,
            threshold=args.threshold,
            merge_gap=args.merge_gap,
        )
        if extended is None:
            continue

        original = pd.read_parquet(PER_DOC_DIR / f"{doc_id}.parquet")
        n_new = len(extended) - len(original)
        total_new += max(0, n_new)

        if args.save and n_new > 0:
            out_path = PER_DOC_DIR / f"{doc_id}_extended.parquet"
            extended.to_parquet(out_path, index=False)

    print(f"\nTotal new spans added across all docs: {total_new}")
    if not args.save:
        print("(dry run — use --save to write extended parquets)")


if __name__ == "__main__":
    main()
