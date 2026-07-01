# 06 — Analytics

Evaluates the full pipeline against the PAN 2011 XML ground truth. Measures
span-level detection quality (Precision, Recall, F1) and source retrieval
quality (Recall@K), macro-averaged across all suspicious documents.

## What it does

```
per_doc detected spans  (from run_pipeline.py)
        +
pan2011_plagiarism_spans.parquet  (ground truth from 02)
        │
        ▼
Span-level TP / FP / FN per document
        │
        ▼
Macro-averaged Precision, Recall, F1
        +
Retrieval Recall@K
        +
Per-type detection breakdown (copy_paste / paraphrase / shake)
        +
Clean-document false-alarm rate
```

## Files

| File | Purpose |
|---|---|
| `analyze_results.ipynb` | Single-document deep-dive: LLM classification, span merging, GT comparison |
| `kb_analysis.py` | Utility scratch file (currently empty) |

## Metrics

| Metric | Description |
|---|---|
| **Precision** | TP / (TP + FP) — of all detected spans, how many overlap a GT span |
| **Recall** | TP / (TP + FN) — of all GT spans, how many were detected |
| **F1** | Harmonic mean of Precision and Recall |
| **Retrieval Recall@K** | Did the true source doc appear in the top-K retrieval results |

All span metrics are **macro-averaged** (one value per document, then averaged)
following the standard PAN 2011 convention.

## Clean document handling

Documents with no GT spans (not plagiarised) are handled correctly:
- No detections → Precision = 1, Recall = 1, F1 = 1 (correct silence)
- Any detections → Precision = 0, F1 = 0 (false alarm penalised)

## For full-dataset evaluation

Use `run_pipeline.py` from `scripts/final/` — it runs the entire pipeline over
all documents and writes aggregate results to `pipeline_results/`:

```bash
python run_pipeline.py --skip-tfidf --run-embeddings
```

See the [pipeline runner README](../README.md) for all flags.
