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
TOP_PAIRS_PER_DOC   = 25
MAX_GAP             = 1800   # chars — merging adjacent detected chunks
OLLAMA_MODEL        = "gemma4:e4b"
RETRIEVAL_TOP_N     = 20


# ---------------------------------------------------------------------------
# LLM helpers (extracted from 05_text_alignment/process_doc.ipynb)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM response: {raw[:300]}")
    clean = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match.group())
    return json.loads(clean)



def score_source_doc(source_doc_id: str, pairs: list[dict]) -> dict:
    from ollama import chat

    pairs_text = "\n\n".join([
        f"[Pair {i+1}]\n"
        f"SUSPICIOUS: {p['suspicious_text'][:600]}\n"
        f"SOURCE CANDIDATE: {p['source_text'][:600]}"
        for i, p in enumerate(pairs)
    ])

    prompt = (
        f"You are a strict plagiarism detection expert.\n"
        f"Below are {len(pairs)} text pair(s). Each pair shows a chunk from a SUSPICIOUS document "
        f"alongside a chunk from a CANDIDATE SOURCE document.\n\n"
        f"{pairs_text}\n\n"
        f"Your task: determine whether the suspicious text was directly copied or closely paraphrased "
        f"from this specific source document.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Score HIGH (>= 0.95) ONLY if multiple pairs show verbatim copying, near-verbatim text, "
        f"or sentence-level paraphrase where unique phrases, names, or sequences are shared.\n"
        f"- Score LOW (< 0.50) if the texts merely discuss the same topic, share common knowledge, "
        f"or use similar vocabulary without specific shared content.\n"
        f"- Topical similarity alone is NOT plagiarism. The suspicious text must reuse specific "
        f"sentences, phrases, or structure from THIS source.\n"
        f"- If fewer than 3 pairs show strong textual overlap, score below 0.50.\n\n"
        f"Respond with ONLY a JSON object — no markdown, no explanation — with keys: "
        f"score (float 0-1), is_likely_source (bool), reasoning (string)."
    )

    t0 = time.time()
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "seed": 42},
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
    LLM source confirmation for one suspicious doc, then merge confirmed spans.
    Classification (copy_paste/paraphrase/shake) removed — adds no measurable
    quality to P/R/F1 and has no GT labels to validate against.
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
        return _merge_spans(pairs_df), pd.DataFrame()

    # ── LLM: source document confirmation ───────────────────────────────────
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

    llm_scores_df = (
        pd.DataFrame(llm_rows)
        .sort_values("llm_score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"  LLM scores (top 3):")
    for _, row in llm_scores_df.head(3).iterrows():
        verdict = "CONFIRMED" if row["llm_score"] >= LLM_SCORE_THRESHOLD else "rejected"
        likely  = "yes" if row["llm_is_likely_source"] else "no"
        reasoning_preview = row["llm_reasoning"][:120].replace("\n", " ")
        print(f"    [{verdict}] {row['source_doc_id']}  score={row['llm_score']:.3f}  likely={likely}  t={row['elapsed_s']}s")
        print(f"             reasoning: {reasoning_preview}...")

    confirmed_ids = set(
        llm_scores_df.loc[llm_scores_df["llm_score"] >= LLM_SCORE_THRESHOLD, "source_doc_id"]
    )
    print(f"  Confirmed sources  : {len(confirmed_ids)} / {len(llm_scores_df)}")

    if not confirmed_ids:
        return pd.DataFrame(), llm_scores_df

    confirmed_pairs = pairs_df[pairs_df["source_doc_id"].isin(confirmed_ids)].copy()
    return _merge_spans(confirmed_pairs), llm_scores_df


def _merge_spans(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to best embedding-score span per suspicious chunk, then merge adjacent spans."""
    if pairs_df.empty:
        return pd.DataFrame()

    # Keep highest-scoring source match per suspicious chunk
    best = (
        pairs_df
        .sort_values("embedding_score", ascending=False)
        .drop_duplicates(subset=["suspicious_chunk_id"], keep="first")
        .sort_values(["source_doc_id", "suspicious_start_char"])
        .reset_index(drop=True)
    )

    merged_rows = []
    for _, grp in best.groupby("source_doc_id"):
        grp = grp.reset_index(drop=True)
        current = grp.iloc[0].to_dict()

        for _, row in grp.iloc[1:].iterrows():
            gap = row["suspicious_start_char"] - current["suspicious_end_char"]
            if gap <= MAX_GAP:
                current["suspicious_end_char"] = max(current["suspicious_end_char"], row["suspicious_end_char"])
                current["source_start_char"]   = min(current["source_start_char"],   row["source_start_char"])
                current["source_end_char"]     = max(current["source_end_char"],     row["source_end_char"])
                if row["embedding_score"] > current["embedding_score"]:
                    current["embedding_score"] = row["embedding_score"]
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


def _char_overlap(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


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

    gt_chars = int((gt_doc["suspicious_end"] - gt_doc["suspicious_offset"]).sum()) if not gt_doc.empty else 0

    if detected.empty and is_clean:
        return _metrics_row(suspicious_doc_id, tp=0, fp=0, fn=0, gt_spans=0, det_spans=0, is_clean=True,
                            det_chars=0, gt_chars=0, overlap_chars=0)

    if detected.empty:
        return _metrics_row(suspicious_doc_id, tp=0, fp=0, fn=len(gt_doc), gt_spans=len(gt_doc), det_spans=0, is_clean=False,
                            det_chars=0, gt_chars=gt_chars, overlap_chars=0)

    det_chars = int((detected["suspicious_end_char"] - detected["suspicious_start_char"]).sum())

    if is_clean:
        return _metrics_row(suspicious_doc_id, tp=0, fp=len(detected), fn=0, gt_spans=0, det_spans=len(detected), is_clean=True,
                            det_chars=det_chars, gt_chars=0, overlap_chars=0)

    hit_flags          = []
    matched_gt_indices = set()
    overlap_chars      = 0

    for _, det in detected.iterrows():
        hit = False
        for gt_idx, gt in gt_doc.iterrows():
            if det["source_doc_id"] != gt["source_doc_id"]:
                continue
            ov = _char_overlap(
                det["suspicious_start_char"], det["suspicious_end_char"],
                gt["suspicious_offset"],      gt["suspicious_end"],
            )
            if ov > 0:
                hit = True
                matched_gt_indices.add(gt_idx)
                overlap_chars += ov
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
        det_chars=det_chars,
        gt_chars=gt_chars,
        overlap_chars=overlap_chars,
    )


def _metrics_row(doc_id, tp, fp, fn, gt_spans, det_spans, is_clean: bool,
                 det_chars: int = 0, gt_chars: int = 0, overlap_chars: int = 0) -> dict:
    # Binary span metrics
    if tp + fp == 0:
        precision = 1.0 if is_clean else 0.0
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # PAN character-level metrics
    char_precision = overlap_chars / det_chars if det_chars else (1.0 if is_clean else 0.0)
    char_recall    = overlap_chars / gt_chars  if gt_chars  else 1.0
    char_f1        = 2 * char_precision * char_recall / (char_precision + char_recall) if (char_precision + char_recall) else 0.0

    return {
        "suspicious_doc_id": doc_id,
        "is_clean":    is_clean,
        "gt_spans":    gt_spans,
        "det_spans":   det_spans,
        "tp": tp, "fp": fp, "fn": fn,
        "precision":      round(precision,      4),
        "recall":         round(recall,         4),
        "f1":             round(f1,             4),
        "det_chars":      det_chars,
        "gt_chars":       gt_chars,
        "overlap_chars":  overlap_chars,
        "char_precision": round(char_precision, 4),
        "char_recall":    round(char_recall,    4),
        "char_f1":        round(char_f1,        4),
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
    parser.add_argument("--llm-threshold",      type=float, default=LLM_SCORE_THRESHOLD, help=f"LLM source confirmation threshold (default: {LLM_SCORE_THRESHOLD})")
    parser.add_argument("--min-top1-score",       type=float, default=0.60,             help="Gate 1: abort if top-1 retrieval score < this value — treat as clean (default: 0.60)")
    parser.add_argument("--relative-gap",         type=float, default=0.70,             help="Gate 2: keep candidates scoring >= top1_score * this factor (default: 0.70)")
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
    print(f"Retrieval min top-1  : {args.min_top1_score}")
    print(f"Retrieval rel. gap   : {args.relative_gap}")
    print(f"Resume cache         : {'ignored (--fresh)' if args.fresh else 'active'}")

    metrics_rows   = []
    retrieval_rows = []

    total = len(doc_ids)
    for i, doc_id in enumerate(tqdm(doc_ids, desc="Pipeline"), start=1):
        out_path = PER_DOC_DIR / f"{doc_id}.parquet"
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {doc_id}")
        print(f"{'='*60}")

        retrieval_stats = {"retrieval_candidates": 0, "retrieval_top1_score": 0.0}
        doc_start_time = time.time()

        # ── Resume: skip if already done ────────────────────────────────────
        meta_path = PER_DOC_DIR / f"{doc_id}.meta.json"
        if out_path.exists() and not args.fresh:
            print(f"  [SKIP] Already processed — loading cached result")
            detected = pd.read_parquet(out_path)
            # Reload saved recall + retrieval_stats if available
            if meta_path.exists():
                saved_rec = json.load(open(meta_path))
                retrieval_rows.append(saved_rec)
                for k in ["retrieval_candidates", "retrieval_top1_score", "gate1_passed",
                          "branch_lsa_top1", "branch_lsa_score", "branch_esa_top1",
                          "branch_esa_score", "branch_emb_top1", "branch_emb_score",
                          "branch_tfidf_top1", "branch_tfidf_score"]:
                    if k in saved_rec:
                        retrieval_stats[k] = saved_rec[k]
                k_col = [c for c in saved_rec if c.startswith("recall_at_")][0]
                print(f"  [RET] {k_col}={saved_rec[k_col]:.2f}  true_sources={saved_rec['true_sources']}  hits={saved_rec['retrieved_hits']}  (cached)")
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
                    min_top1_score=args.min_top1_score,
                    relative_gap=args.relative_gap,
                )
                gate1_failed = "_top1_score" in top20_df.columns
                if gate1_failed:
                    raw_top1 = float(top20_df["_top1_score"].iloc[0])
                else:
                    raw_top1 = float(top20_df["final_score"].iloc[0]) if not top20_df.empty else 0.0

                def _branch_val(col, default):
                    return top20_df[col].iloc[0] if col in top20_df.columns else default

                retrieval_stats = {
                    "retrieval_candidates":  0 if gate1_failed else len(top20_df),
                    "retrieval_top1_score":  raw_top1,
                    "gate1_passed":          not gate1_failed,
                    "branch_lsa_top1":       _branch_val("_branch_lsa_top1",   ""),
                    "branch_lsa_score":      _branch_val("_branch_lsa_score",  0.0),
                    "branch_esa_top1":       _branch_val("_branch_esa_top1",   ""),
                    "branch_esa_score":      _branch_val("_branch_esa_score",  0.0),
                    "branch_emb_top1":       _branch_val("_branch_emb_top1",   ""),
                    "branch_emb_score":      _branch_val("_branch_emb_score",  0.0),
                    "branch_tfidf_top1":     _branch_val("_branch_tfidf_top1", ""),
                    "branch_tfidf_score":    _branch_val("_branch_tfidf_score",0.0),
                }
                print(f"  [1/2] Done — top {len(top20_df)} candidates retrieved")
                if gate1_failed or top20_df.empty:
                    print(f"  [WARN] Retrieval returned no candidates — skipping LLM stage")
                    detected = pd.DataFrame()
                    detected.to_parquet(out_path, index=False)
                    gt_doc = gt_df[gt_df["suspicious_doc_id"] == doc_id].copy()
                    metrics_rows.append({**evaluate_doc(doc_id, detected, gt_doc), **retrieval_stats})
                    continue
                print(f"        Top-1 candidate: {top20_df['source_doc_id'].iloc[0]}")
            except Exception as e:
                print(f"  [WARN] Retrieval failed: {e}")
                continue

            # ── Stage 2: text alignment (LLM) ────────────────────────────────
            llm_scores_df = pd.DataFrame()
            if not args.skip_llm:
                print(f"  [2/2] LLM text alignment (threshold={LLM_SCORE_THRESHOLD})...")
                candidates_path = PROCESSED_DIR / "embedding_candidates_suspicious.parquet"
                candidates_df = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()
                try:
                    detected, llm_scores_df = run_text_alignment(
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
        elapsed = round(time.time() - doc_start_time, 1)
        metrics = {**evaluate_doc(doc_id, detected, gt_doc), **retrieval_stats, "elapsed_s": elapsed}
        metrics_rows.append(metrics)
        print(f"  [GT]  gt_spans={metrics['gt_spans']}  detected={metrics['det_spans']}  "
              f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
              f"P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  "
              f"charP={metrics['char_precision']:.2f}  charR={metrics['char_recall']:.2f}  charF1={metrics['char_f1']:.2f}")

        # ── Retrieval Recall@K ────────────────────────────────────────────────
        # Only compute from live data when we actually ran retrieval this session
        if not (out_path.exists() and not args.fresh) or not meta_path.exists():
            emb_top_path = PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"
            if emb_top_path.exists():
                try:
                    ret_top_df = pd.read_parquet(emb_top_path)
                    rec = {**retrieval_recall_at_k(doc_id, ret_top_df, gt_doc), **retrieval_stats}
                    retrieval_rows.append(rec)

                    # Build enriched diagnostic JSON
                    def _top5(df, score_col):
                        if df is None or df.empty:
                            return []
                        return [{"doc": r["source_doc_id"], "score": round(float(r[score_col]), 4)}
                                for _, r in df.head(5).iterrows()]

                    gate2_candidates = [
                        {"doc": r["source_doc_id"], "score": round(float(r["final_score"]), 4)}
                        for _, r in top20_df.iterrows()
                        if "final_score" in top20_df.columns
                    ] if not top20_df.empty else []

                    llm_all = []
                    if not llm_scores_df.empty:
                        for _, r in llm_scores_df.iterrows():
                            llm_all.append({
                                "doc": r["source_doc_id"],
                                "score": round(float(r["llm_score"]), 3),
                                "confirmed": bool(r["llm_score"] >= LLM_SCORE_THRESHOLD),
                                "reasoning": r["llm_reasoning"][:150],
                            })

                    confirmed_sources = [
                        r["source_doc_id"] for _, r in detected.drop_duplicates("source_doc_id").iterrows()
                    ] if not detected.empty and "source_doc_id" in detected.columns else []

                    gt_sources = gt_doc["source_doc_id"].unique().tolist() if not gt_doc.empty else []

                    meta = {
                        **rec,
                        "gt_sources": gt_sources,
                        "gate2_candidates": gate2_candidates,
                        "llm_scores": llm_all,
                        "confirmed_sources": confirmed_sources,
                    }
                    json.dump(meta, open(meta_path, "w"), indent=2)
                    k_col = [c for c in rec if c.startswith("recall_at_")][0]
                    print(f"  [RET] {k_col}={rec[k_col]:.2f}  true_sources={rec['true_sources']}  hits={rec['retrieved_hits']}")
                except Exception:
                    pass

        # ── Incremental save after each doc ──────────────────────────────────
        pd.DataFrame(metrics_rows).to_parquet(RESULTS_DIR / "analytics_summary.parquet", index=False)
        if retrieval_rows:
            pd.DataFrame(retrieval_rows).to_parquet(RESULTS_DIR / "retrieval_recall.parquet", index=False)

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

    total_overlap = metrics_df["overlap_chars"].sum()
    total_det_ch  = metrics_df["det_chars"].sum()
    total_gt_ch   = metrics_df["gt_chars"].sum()
    micro_char_p  = total_overlap / total_det_ch if total_det_ch else 0.0
    micro_char_r  = total_overlap / total_gt_ch  if total_gt_ch  else 0.0
    micro_char_f1 = 2 * micro_char_p * micro_char_r / (micro_char_p + micro_char_r) if (micro_char_p + micro_char_r) else 0.0
    macro_char_p  = metrics_df["char_precision"].mean()
    macro_char_r  = metrics_df["char_recall"].mean()
    macro_char_f1 = metrics_df["char_f1"].mean()

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Documents evaluated          : {len(metrics_df)}")
    print(f"  — with GT plagiarism spans : {len(plagiarised_df)}")
    print(f"  — clean (no GT spans)      : {len(clean_df)}")
    print()
    print(f"{'Metric':<22} {'Binary (macro)':>16} {'Char micro':>12} {'Char macro':>12}")
    print(f"{'Precision':<22} {macro_precision:>16.4f} {micro_char_p:>12.4f} {macro_char_p:>12.4f}")
    print(f"{'Recall':<22} {macro_recall:>16.4f} {micro_char_r:>12.4f} {macro_char_r:>12.4f}")
    print(f"{'F1':<22} {macro_f1:>16.4f} {micro_char_f1:>12.4f} {macro_char_f1:>12.4f}")
    print()
    print(f"Total GT spans     : {metrics_df['gt_spans'].sum()}")
    print(f"Total detected     : {metrics_df['det_spans'].sum()}")
    print(f"Total TP / FP / FN : {metrics_df['tp'].sum()} / {metrics_df['fp'].sum()} / {metrics_df['fn'].sum()}")
    print(f"Total GT chars     : {total_gt_ch:,}")
    print(f"Total det chars    : {total_det_ch:,}")
    print(f"Overlap chars      : {total_overlap:,}  ({100*micro_char_r:.1f}% of GT covered)")
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
