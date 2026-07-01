# Pipeline Runner — `run_pipeline.py`

End-to-end plagiarism detection pipeline for PAN 2011. Runs source retrieval,
LLM text alignment, and span-level evaluation for every suspicious document,
storing per-doc results so you can pause and resume at any time.

---

## Quick start

Run from `scripts/final/`:

```bash
# First time on a doc — needs Docker running for embeddings
python run_pipeline.py --run-embeddings --skip-tfidf

# Resume after a pause (already-processed docs are skipped automatically)
python run_pipeline.py --run-embeddings --skip-tfidf

# Smoke test — 7 docs, no TF-IDF, no LLM, no Docker
python run_pipeline.py --docs 7 --skip-tfidf --skip-llm
```

---

## All flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--docs N` | int | all | Process only the first N documents |
| `--doc-id <id>` | str | — | Process a single specific document by ID |
| `--skip-tfidf` | flag | off | Skip TF-IDF retrieval branch (recommended — it's slow) |
| `--skip-llm` | flag | off | Skip LLM stages entirely; evaluates retrieval quality only |
| `--run-embeddings` | flag | off | Trigger GPU embeddings live via Docker (default: load existing parquet) |
| `--top-n N` | int | 20 | Number of top candidate source docs passed from retrieval to alignment |
| `--llm-threshold F` | float | 0.95 | Minimum LLM score for a source doc to be confirmed as a plagiarism origin |
| `--fresh` | flag | off | Ignore resume cache — reprocess all docs even if results already exist |

---

## Resume behaviour

Results are saved **immediately** after each document is processed:

```
scripts/final/pipeline_results/per_doc/<doc_id>.parquet
```

If you kill the process and restart with the same flags, already-saved docs are
skipped. Use `--fresh` to reprocess everything from scratch.

---

## Output files

| File | Description |
|------|-------------|
| `pipeline_results/per_doc/<doc_id>.parquet` | Detected plagiarism spans for one document |
| `pipeline_results/analytics_summary.parquet` | Per-doc precision, recall, F1, TP/FP/FN |
| `pipeline_results/retrieval_recall.parquet` | Retrieval Recall@K per document |

---

## Pipeline stages

```
1. Source Retrieval     ESA + LSA (+ optional TF-IDF) + Embeddings → top-N ranked source docs
2. LLM Confirmation     LLM scores each candidate source doc (0–1); drops below threshold
3. Chunk Classification LLM labels each chunk pair: copy_paste / paraphrase / shake / none
4. Span Merging         Adjacent detected chunks merged into contiguous suspicious spans
5. GT Evaluation        Merged spans compared against PAN 2011 XML ground truth
```

---

## Typical full-dataset run

```bash
# Run everything, one doc at a time, resumable
python run_pipeline.py --run-embeddings --skip-tfidf

# When done, results are in pipeline_results/analytics_summary.parquet
```

Docker container `docplag-rocm` must be running when `--run-embeddings` is used.

---

## Evaluation metrics

Detection is evaluated at **span level** — a detected span is a True Positive
if it overlaps a ground-truth span on the suspicious side and matches the source
document ID.

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of P and R
- Reported as **macro-average** across all documents (standard PAN convention)

Clean documents (no GT spans) contribute `precision=1, recall=1, f1=1` when
correctly left undetected, or `precision=0, f1=0` if false alarms are raised.
