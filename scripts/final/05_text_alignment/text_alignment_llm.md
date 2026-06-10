# 05 — Text Alignment (LLM Scoring + Classification)

Takes the top-N candidate source documents from `04_source_retrieval` and runs
two LLM passes to confirm true sources and classify the type of plagiarism at
chunk level, then merges adjacent chunks into contiguous detected spans.

## What it does

```
top-N candidate source documents (from fusion)
        │
        ▼
Select top-25 highest-similarity chunk pairs per source doc
(suspicious chunk ↔ source chunk, ranked by embedding score)
        │
        ▼
Pass 1 — LLM source confirmation
  Score each source doc 0–1; drop docs below threshold (default 0.95)
        │
        ▼
Pass 2 — LLM chunk-pair classification
  Label each chunk pair: copy_paste / paraphrase / shake / none
        │
        ▼
Span merging
  Deduplicate to one label per suspicious chunk → merge adjacent
  chunks (gap ≤ 1800 chars) into contiguous plagiarism spans
        │
        ▼
alignment_classified_spans.parquet
```

## Files

| File | Purpose |
|---|---|
| `process_doc.ipynb` | Notebook: builds text pairs, runs both LLM passes, produces classified spans |
| `llm_scores_df.parquet` | Cached LLM source-confirmation scores for the demo document |

## LLM setup

Both passes use **Ollama** running locally via the `ollama` Python package.
Default model: `gemma4:e4b`. Temperature is set to 0 for deterministic output.
Responses are plain JSON parsed with a regex fallback.

## Plagiarism type taxonomy

| Type | Meaning |
|------|---------|
| `copy_paste` | Verbatim or near-verbatim copy (< 5 % change) |
| `paraphrase` | Meaning preserved, sentences restructured |
| `shake` | Synonym substitution / light edits, same structure |
| `none` | No meaningful plagiarism detected |

## Text pair selection

Top-25 chunk pairs per source document by embedding cosine similarity, drawn
from `embedding_candidates_suspicious.parquet`. Each chunk is truncated to
600 characters in the prompt.

## Output

### `alignment_classified_spans.parquet`

| Column | Description |
|---|---|
| `suspicious_doc_id` | Source suspicious document |
| `source_doc_id` | Confirmed source document |
| `suspicious_start_char` / `suspicious_end_char` | Merged span offsets in suspicious doc |
| `source_start_char` / `source_end_char` | Corresponding offsets in source doc |
| `plagiarism_type` | Dominant type in the merged span |
| `type_confidence` | LLM confidence (0–1) |
| `embedding_score` | Max embedding similarity in the span |

## Notes

- LLM re-ranking is a **second-pass filter** — if the true source is not in the
  retrieval top-N it cannot be recovered here.
- The span merge gap (1800 chars ≈ 1 chunk width) absorbs sliding-window overlaps
  without incorrectly joining distant detections.
- This stage is fully automated in `run_pipeline.py` for batch evaluation.
