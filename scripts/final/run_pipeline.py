"""
Full pipeline runner for PAN 2011 plagiarism detection.

Runs source retrieval + LLM text alignment for every suspicious document in
the dataset, stores per-doc results, then computes aggregate analytics.

Usage (run from scripts/final/):
    python run_pipeline.py [--docs N] [--skip-tfidf] [--skip-llm]

    --docs N        Process only the first N documents (default: all)
    --skip-tfidf    Skip TF-IDF branch (faster; redistributes its weight)
    --skip-llm      Skip LLM scoring/classification; evaluate retrieval only

Output layout:
    scripts/final/pipeline_results/
        per_doc/<suspicious_doc_id>.parquet   -- detected spans per doc
        analytics_summary.parquet             -- aggregate metrics table
        retrieval_recall.parquet              -- Recall@K per doc
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (all relative paths in source_retrieval_branches.py are anchored to
# scripts/final/04_source_retrieval/, so we chdir there before importing it)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # scripts/final/
RETRIEVAL_DIR = SCRIPT_DIR / "04_source_retrieval"
RESULTS_DIR  = SCRIPT_DIR / "pipeline_results"
PER_DOC_DIR  = RESULTS_DIR / "per_doc"

PROCESSED_DIR = SCRIPT_DIR.parents[1] / "datasets" / "processed" / "PAN2011_300"
GT_PATH       = SCRIPT_DIR.parents[1] / "datasets" / "processed" / "PAN2011_ground_truth" / "pan2011_plagiarism_spans.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_SCORE_THRESHOLD = 0.95
TOP_PAIRS_PER_DOC   = 15
MAX_GAP             = 1800   # chars — merging adjacent detected chunks
OLLAMA_MODEL        = "gemma4:e4b"
RETRIEVAL_TOP_N     = 20
TYPE_RANK           = {"copy_paste": 3, "shake": 2, "paraphrase": 1, "none": 0}
PLAGIARISM_TYPES    = {"copy_paste", "paraphrase", "shake", "none"}


# ---------------------------------------------------------------------------
# LLM helpers (extracted from 05_text_alignment/process_doc.ipynb)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM response: {raw[:300]}")
    return json.loads(match.group())


def score_source_doc(source_doc_id: str, pairs: list[dict]) -> dict:
    from ollama import chat

    pairs_text = "\n\n".join([
        f"[Pair {i+1}]\n"
        f"SUSPICIOUS: {p['suspicious_text'][:600]}\n"
        f"SOURCE CANDIDATE: {p['source_text'][:600]}"
        for i, p in enumerate(pairs)
    ])

    prompt = (
        f"You are a plagiarism detection expert.\n"
        f"Below are {len(pairs)} text pair(s). Each pair shows a chunk from a SUSPICIOUS document "
        f"alongside a chunk from a CANDIDATE SOURCE document.\n\n"
        f"{pairs_text}\n\n"
        f"Analyze whether the suspicious chunks appear to be copied, paraphrased, or otherwise "
        f"derived from the source document. "
        f"Score the overall likelihood that this source document is the true origin of the "
        f"suspicious text (0.0 = definitely not, 1.0 = definitely yes). "
        f"Respond with ONLY a JSON object — no markdown, no explanation — with keys: "
        f"score (float 0-1), is_likely_source (bool), reasoning (string)."
    )

    t0 = time.time()
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        think=False,
    )
    elapsed = time.time() - t0

    data = _parse_json(response.message.content)
    return {
        "source_doc_id":        source_doc_id,
        "llm_score":            float(data.get("score", 0.0)),
        "llm_is_likely_source": bool(data.get("is_likely_source", False)),
        "llm_reasoning":        data.get("reasoning", ""),
        "elapsed_s":            round(elapsed, 1),
    }


def classify_chunk_pair(suspicious_text: str, source_text: str) -> dict:
    from ollama import chat

    prompt = (
        "You are a plagiarism detection expert.\n\n"
        "Compare the SUSPICIOUS chunk and the SOURCE chunk below.\n\n"
        f"SUSPICIOUS:\n{suspicious_text}\n\n"
        f"SOURCE:\n{source_text}\n\n"
        "Classify the relationship into exactly one of these types:\n"
        "  - copy_paste : text is copied verbatim or near-verbatim (< 5% change)\n"
        "  - paraphrase : meaning preserved but sentences restructured or rewritten\n"
        "  - shake      : words replaced by synonyms / light edits, same structure\n"
        "  - none       : no meaningful plagiarism detected\n\n"
        "Also rate your confidence (0.0-1.0).\n"
        "Respond with ONLY a JSON object — no markdown — with keys: "
        "plagiarism_type (string), confidence (float 0-1), reasoning (string)."
    )

    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        think=False,
    )

    data = _parse_json(response.message.content)
    ptype = data.get("plagiarism_type", "none")
    if ptype not in PLAGIARISM_TYPES:
        ptype = "none"

    return {
        "plagiarism_type": ptype,
        "type_confidence": float(data.get("confidence", 0.0)),
        "type_reasoning":  data.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Text alignment stage (replaces process_doc.ipynb)
# ---------------------------------------------------------------------------

def run_text_alignment(
    suspicious_doc_id: str,
    top20_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    source_chunks: pd.DataFrame,
    suspicious_chunks: pd.DataFrame,
    skip_llm: bool = False,
) -> pd.DataFrame:
    """
    Runs LLM source confirmation + chunk-pair classification for one doc.
    Returns a DataFrame of detected spans (plagiarism_type != 'none' only,
    after merging adjacent chunks), ready for GT evaluation.
    """
    top_source_ids = set(top20_df["source_doc_id"].tolist())

    candidates_filtered = candidates_df[
        (candidates_df["suspicious_doc_id"] == suspicious_doc_id) &
        (candidates_df["source_doc_id"].isin(top_source_ids))
    ].copy()

    if candidates_filtered.empty:
        return pd.DataFrame()

    susp_text = (
        suspicious_chunks[suspicious_chunks["doc_id"] == suspicious_doc_id]
        [["chunk_id", "embedding_text"]]
        .rename(columns={"chunk_id": "suspicious_chunk_id", "embedding_text": "suspicious_text"})
    )

    src_text = (
        source_chunks[source_chunks["doc_id"].isin(top_source_ids)]
        [["chunk_id", "chunk_text"]]
        .rename(columns={"chunk_id": "source_chunk_id", "chunk_text": "source_text"})
    )

    pairs_df = (
        candidates_filtered
        .merge(susp_text, on="suspicious_chunk_id", how="inner")
        .merge(src_text,  on="source_chunk_id",     how="inner")
    )

    top_pairs = (
        pairs_df
        .sort_values("embedding_score", ascending=False)
        .groupby("source_doc_id")
        .head(TOP_PAIRS_PER_DOC)
        .reset_index(drop=True)
    )

    if skip_llm:
        # Treat all candidate pairs as confirmed, skip classification
        confirmed_pairs = top_pairs.copy()
        confirmed_pairs["plagiarism_type"]  = "unknown"
        confirmed_pairs["type_confidence"]  = 0.0
        confirmed_pairs["type_reasoning"]   = ""
        return _merge_spans(confirmed_pairs)

    # ── LLM stage 1: source document scoring ────────────────────────────────
    llm_rows = []
    for source_doc_id, group in top_pairs.groupby("source_doc_id"):
        pairs = group[["suspicious_text", "source_text", "embedding_score"]].to_dict("records")
        try:
            llm_rows.append(score_source_doc(source_doc_id, pairs))
        except Exception as e:
            llm_rows.append({
                "source_doc_id": source_doc_id,
                "llm_score": 0.0,
                "llm_is_likely_source": False,
                "llm_reasoning": f"Error: {e}",
                "elapsed_s": 0.0,
            })

    llm_scores_df = pd.DataFrame(llm_rows)
    confirmed_ids = set(
        llm_scores_df.loc[llm_scores_df["llm_score"] >= LLM_SCORE_THRESHOLD, "source_doc_id"]
    )

    if not confirmed_ids:
        return pd.DataFrame()

    confirmed_pairs = pairs_df[pairs_df["source_doc_id"].isin(confirmed_ids)].copy()

    # ── LLM stage 2: chunk-pair type classification ──────────────────────────
    classification_rows = []
    for row in confirmed_pairs.itertuples(index=False):
        try:
            result = classify_chunk_pair(row.suspicious_text, row.source_text)
        except Exception as e:
            result = {"plagiarism_type": "none", "type_confidence": 0.0, "type_reasoning": f"Error: {e}"}

        classification_rows.append({
            "suspicious_chunk_id":   row.suspicious_chunk_id,
            "suspicious_doc_id":     row.suspicious_doc_id,
            "suspicious_start_char": row.suspicious_start_char,
            "suspicious_end_char":   row.suspicious_end_char,
            "source_chunk_id":       row.source_chunk_id,
            "source_doc_id":         row.source_doc_id,
            "source_start_char":     row.source_start_char,
            "source_end_char":       row.source_end_char,
            "embedding_score":       row.embedding_score,
            **result,
        })

    alignment_df = pd.DataFrame(classification_rows)
    return _merge_spans(alignment_df)


def _merge_spans(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one span per suspicious chunk, then merge adjacent spans."""
    if alignment_df.empty:
        return pd.DataFrame()

    best_spans = (
        alignment_df
        .assign(is_plag=(alignment_df["plagiarism_type"] != "none").astype(int))
        .sort_values(["is_plag", "type_confidence"], ascending=[False, False])
        .drop_duplicates(subset=["suspicious_chunk_id"], keep="first")
        .drop(columns=["is_plag"])
        .reset_index(drop=True)
    )

    detected_chunks = (
        best_spans[best_spans["plagiarism_type"] != "none"]
        .sort_values(["source_doc_id", "suspicious_start_char"])
        .reset_index(drop=True)
    )

    if detected_chunks.empty:
        return pd.DataFrame()

    merged_rows = []
    for _, grp in detected_chunks.groupby("source_doc_id"):
        grp = grp.sort_values("suspicious_start_char").reset_index(drop=True)
        current = grp.iloc[0].to_dict()

        for _, row in grp.iloc[1:].iterrows():
            gap = row["suspicious_start_char"] - current["suspicious_end_char"]
            if gap <= MAX_GAP:
                current["suspicious_end_char"] = max(current["suspicious_end_char"], row["suspicious_end_char"])
                current["source_start_char"]   = min(current["source_start_char"],   row["source_start_char"])
                current["source_end_char"]     = max(current["source_end_char"],     row["source_end_char"])
                if TYPE_RANK.get(row["plagiarism_type"], 0) > TYPE_RANK.get(current["plagiarism_type"], 0):
                    current["plagiarism_type"] = row["plagiarism_type"]
                    current["type_confidence"] = row["type_confidence"]
                    current["type_reasoning"]  = row["type_reasoning"]
            else:
                merged_rows.append(current)
                current = row.to_dict()

        merged_rows.append(current)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ground-truth evaluation (replaces analyze_results.ipynb step 6)
# ---------------------------------------------------------------------------

def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def evaluate_doc(
    suspicious_doc_id: str,
    detected: pd.DataFrame,
    gt_doc: pd.DataFrame,
) -> dict:
    """
    Span-level P/R/F1 for one suspicious document.
    A TP requires: same source_doc_id AND suspicious-side char overlap.

    Clean documents (gt_spans == 0):
      - No detections → precision=1, recall=1, f1=1  (correct silence)
      - Any detections → precision=0, recall=1, f1=0  (false alarms)
    """
    is_clean = gt_doc.empty

    if detected.empty and is_clean:
        # Correctly identified as clean
        return _metrics_row(suspicious_doc_id, tp=0, fp=0, fn=0, gt_spans=0, det_spans=0, is_clean=True)

    if detected.empty:
        # Plagiarised doc, missed entirely
        return _metrics_row(suspicious_doc_id, tp=0, fp=0, fn=len(gt_doc), gt_spans=len(gt_doc), det_spans=0, is_clean=False)

    if is_clean:
        # Clean doc but we raised false alarms
        return _metrics_row(suspicious_doc_id, tp=0, fp=len(detected), fn=0, gt_spans=0, det_spans=len(detected), is_clean=True)

    hit_flags          = []
    matched_gt_indices = set()

    for _, det in detected.iterrows():
        hit = False
        for gt_idx, gt in gt_doc.iterrows():
            if det["source_doc_id"] != gt["source_doc_id"]:
                continue
            if _overlaps(
                det["suspicious_start_char"], det["suspicious_end_char"],
                gt["suspicious_offset"],      gt["suspicious_end"],
            ):
                hit = True
                matched_gt_indices.add(gt_idx)
        hit_flags.append(hit)

    tp = sum(hit_flags)
    fp = len(detected) - tp
    fn = len(gt_doc) - len(matched_gt_indices)

    return _metrics_row(
        suspicious_doc_id,
        tp=tp, fp=fp, fn=fn,
        gt_spans=len(gt_doc),
        det_spans=len(detected),
        is_clean=False,
    )


def _metrics_row(doc_id, tp, fp, fn, gt_spans, det_spans, is_clean: bool) -> dict:
    # Precision: undefined (1.0) when nothing was detected and doc is clean
    if tp + fp == 0:
        precision = 1.0 if is_clean else 0.0
    else:
        precision = tp / (tp + fp)

    # Recall: undefined (1.0) when there is nothing to find (clean doc)
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "suspicious_doc_id": doc_id,
        "is_clean":    is_clean,
        "gt_spans":    gt_spans,
        "det_spans":   det_spans,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def retrieval_recall_at_k(
    suspicious_doc_id: str,
    top_k_df: pd.DataFrame,
    gt_doc: pd.DataFrame,
    k: int = 20,
) -> dict:
    """Did the true source doc(s) appear in the top-K retrieval results?"""
    true_sources = set(gt_doc["source_doc_id"].unique())
    retrieved    = set(top_k_df["source_doc_id"].head(k).tolist())
    hits         = len(true_sources & retrieved)
    # Clean doc: nothing to retrieve → recall is 1.0 by convention
    recall       = hits / len(true_sources) if true_sources else 1.0
    return {
        "suspicious_doc_id": suspicious_doc_id,
        "true_sources":      len(true_sources),
        "retrieved_hits":    hits,
        f"recall_at_{k}":    round(recall, 4),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    global LLM_SCORE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="PAN 2011 end-to-end plagiarism pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Quick smoke test — 7 docs, no TF-IDF, no LLM
    python run_pipeline.py --docs 7 --skip-tfidf --skip-llm

    # 7 docs, no TF-IDF, run embeddings via Docker, include LLM
    python run_pipeline.py --docs 7 --skip-tfidf --run-embeddings

    # Full dataset, all branches
    python run_pipeline.py

    # Re-run a single specific document (ignores resume cache)
    python run_pipeline.py --doc-id part1__suspicious-document00007.txt

    # Wipe resume cache and start fresh
    python run_pipeline.py --fresh
            """,
    )
    parser.add_argument("--docs",           type=int,   default=None,  help="Process only first N docs")
    parser.add_argument("--doc-id",         type=str,   default=None,  help="Process a single specific doc ID")
    parser.add_argument("--skip-tfidf",     action="store_true",        help="Skip TF-IDF branch (faster)")
    parser.add_argument("--skip-esa",       action="store_true",        help="Skip ESA branch (use if RAM is insufficient)")
    parser.add_argument("--skip-llm",       action="store_true",        help="Skip LLM stages (retrieval eval only)")
    parser.add_argument("--run-embeddings", action="store_true",        help="Trigger GPU embeddings via Docker (default: load from parquet)")
    parser.add_argument("--top-n",          type=int,   default=RETRIEVAL_TOP_N, help=f"Top-N candidates from retrieval (default: {RETRIEVAL_TOP_N})")
    parser.add_argument("--llm-threshold",  type=float, default=LLM_SCORE_THRESHOLD, help=f"LLM source confirmation threshold (default: {LLM_SCORE_THRESHOLD})")
    parser.add_argument("--fresh",          action="store_true",        help="Ignore resume cache — reprocess all docs")
    args = parser.parse_args()

    LLM_SCORE_THRESHOLD = args.llm_threshold

    PER_DOC_DIR.mkdir(parents=True, exist_ok=True)

    # ── Change CWD so all relative paths in source_retrieval_branches work ──
    # The module uses ../../artifacts/ and ../../datasets/ relative to 04_source_retrieval/
    os.chdir(RETRIEVAL_DIR)
    sys.path.insert(0, str(RETRIEVAL_DIR))
    from source_retrieval_branches import lookup_pipeline  # noqa: E402

    # ── Load shared data once ────────────────────────────────────────────────
    print("Loading shared data...")
    susp_docs       = pd.read_parquet(PROCESSED_DIR / "suspicious_documents.parquet")
    gt_df           = pd.read_parquet(GT_PATH)
    source_chunks   = pd.read_parquet(PROCESSED_DIR / "source_chunks.parquet")
    susp_chunks_emb = pd.read_parquet(PROCESSED_DIR / "suspicious_chunks_embeddings.parquet")
    # NOTE: embedding_candidates_suspicious.parquet is a per-doc scratch file written
    # by Docker after each embedding run — loaded fresh inside the loop, not here.

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = susp_docs["doc_id"].tolist()
        if args.docs:
            doc_ids = doc_ids[: args.docs]

    print(f"Documents to process : {len(doc_ids)}")
    print(f"TF-IDF               : {'off' if args.skip_tfidf else 'on'}")
    print(f"Embeddings           : {'Docker (live)' if args.run_embeddings else 'load from parquet'}")
    print(f"LLM alignment        : {'off' if args.skip_llm else 'on'}")
    print(f"Top-N retrieval      : {args.top_n}")
    print(f"LLM threshold        : {LLM_SCORE_THRESHOLD}")
    print(f"Resume cache         : {'ignored (--fresh)' if args.fresh else 'active'}")

    metrics_rows   = []
    retrieval_rows = []

    total = len(doc_ids)
    for i, doc_id in enumerate(tqdm(doc_ids, desc="Pipeline"), start=1):
        out_path = PER_DOC_DIR / f"{doc_id}.parquet"
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {doc_id}")
        print(f"{'='*60}")

        # ── Resume: skip if already done ────────────────────────────────────
        if out_path.exists() and not args.fresh:
            print(f"  [SKIP] Already processed — loading cached result")
            detected = pd.read_parquet(out_path)
        else:
            # ── Stage 1: source retrieval ────────────────────────────────────
            branches = " + ".join(filter(None, [
                None if args.skip_tfidf else "TF-IDF",
                None if args.skip_esa   else "ESA",
                "LSA",
                "Docker embeddings" if args.run_embeddings else "cached embeddings",
            ]))
            print(f"  [1/2] Source retrieval ({branches})...")
            try:
                top20_df = lookup_pipeline(
                    doc_id,
                    run_embeddings=args.run_embeddings,
                    run_tfidf=not args.skip_tfidf,
                    top_n=args.top_n,
                )
                print(f"  [1/2] Done — top {len(top20_df)} candidates retrieved")
                print(f"        Top-1 candidate: {top20_df['source_doc_id'].iloc[0]}")
            except Exception as e:
                print(f"  [WARN] Retrieval failed: {e}")
                continue

            # ── Stage 2: text alignment (LLM) ────────────────────────────────
            if not args.skip_llm:
                print(f"  [2/2] LLM text alignment (threshold={LLM_SCORE_THRESHOLD})...")
                # Reload per-doc — Docker overwrites this file after each embedding run
                candidates_path = PROCESSED_DIR / "embedding_candidates_suspicious.parquet"
                candidates_df = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()
                try:
                    detected = run_text_alignment(
                        doc_id,
                        top20_df,
                        candidates_df,
                        source_chunks,
                        susp_chunks_emb,
                        skip_llm=False,
                    )
                    print(f"  [2/2] Done — {len(detected)} detected spans")
                except Exception as e:
                    print(f"  [WARN] Alignment failed: {e}")
                    detected = pd.DataFrame()
            else:
                print(f"  [2/2] Skipped (--skip-llm)")
                detected = pd.DataFrame()

            detected.to_parquet(out_path, index=False)

        # ── GT evaluation ────────────────────────────────────────────────────
        gt_doc = gt_df[gt_df["suspicious_doc_id"] == doc_id].copy()
        metrics = evaluate_doc(doc_id, detected, gt_doc)
        metrics_rows.append(metrics)
        print(f"  [GT]  gt_spans={metrics['gt_spans']}  detected={metrics['det_spans']}  "
              f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
              f"P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

        # ── Retrieval Recall@K ────────────────────────────────────────────────
        emb_top_path = PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"
        if emb_top_path.exists():
            try:
                ret_top_df = pd.read_parquet(emb_top_path)
                rec = retrieval_recall_at_k(doc_id, ret_top_df, gt_doc)
                retrieval_rows.append(rec)
                k_col = [c for c in rec if c.startswith("recall_at_")][0]
                print(f"  [RET] {k_col}={rec[k_col]:.2f}  true_sources={rec['true_sources']}  hits={rec['retrieved_hits']}")
            except Exception:
                pass

    # ── Aggregate analytics ──────────────────────────────────────────────────
    if not metrics_rows:
        print("No results to aggregate.")
        return

    metrics_df = pd.DataFrame(metrics_rows)

    plagiarised_df = metrics_df[~metrics_df["is_clean"]]
    clean_df       = metrics_df[metrics_df["is_clean"]]

    macro_precision = metrics_df["precision"].mean()
    macro_recall    = metrics_df["recall"].mean()
    macro_f1        = metrics_df["f1"].mean()

    print("\n" + "=" * 55)
    print("AGGREGATE RESULTS (macro-averaged over all docs)")
    print("=" * 55)
    print(f"Documents evaluated          : {len(metrics_df)}")
    print(f"  — with GT plagiarism spans : {len(plagiarised_df)}")
    print(f"  — clean (no GT spans)      : {len(clean_df)}")
    print()
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print()
    print(f"Total GT spans  : {metrics_df['gt_spans'].sum()}")
    print(f"Total detected  : {metrics_df['det_spans'].sum()}")
    print(f"Total TP        : {metrics_df['tp'].sum()}")
    print(f"Total FP        : {metrics_df['fp'].sum()}  ← false alarms (incl. on clean docs)")
    print(f"Total FN        : {metrics_df['fn'].sum()}  ← missed plagiarism spans")
    if len(clean_df):
        false_alarm_docs = (clean_df["fp"] > 0).sum()
        print(f"Clean docs with false alarms: {false_alarm_docs} / {len(clean_df)}")

    metrics_df.to_parquet(RESULTS_DIR / "analytics_summary.parquet", index=False)
    print(f"\nSaved per-doc metrics to: {RESULTS_DIR / 'analytics_summary.parquet'}")

    if retrieval_rows:
        ret_df = pd.DataFrame(retrieval_rows)
        col = [c for c in ret_df.columns if c.startswith("recall_at_")][0]
        print(f"\nRetrieval {col} (macro): {ret_df[col].mean():.4f}")
        ret_df.to_parquet(RESULTS_DIR / "retrieval_recall.parquet", index=False)
        print(f"Saved retrieval recall to: {RESULTS_DIR / 'retrieval_recall.parquet'}")


if __name__ == "__main__":
    main()
