# 05 — Text Alignment (LLM Re-ranking)

Takes the top-N candidate source documents produced by `04_source_retrieval` and re-ranks them using an LLM that reads the actual text passage pairs. The LLM assigns a plagiarism likelihood score (0–1) to each candidate, producing a final short-list of the most probable source documents.

## What it does

```
top-N candidate source documents (from fusion)
        │
        ▼
For each source doc: select top-K highest-similarity chunk pairs
(suspicious chunk ↔ source chunk, ranked by embedding score)
        │
        ▼
LLM prompt: "Does suspicious text appear copied/paraphrased from source text?"
        │
        ▼
llm_scores_df  — source docs ranked by LLM score
```

## Files

| File | Purpose |
|---|---|
| `process_doc.ipynb` | Main notebook: builds text pairs, calls the LLM, produces the final scored DataFrame |
| `llm_scores_df.parquet` | Cached output from a previous run (for the demo suspicious document) |

## LLM setup

The notebook uses **Ollama** running locally, accessed via the `instructor` library for structured JSON output. The default model is `qwen2.5:9b`.

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
```

Each source document gets a structured response:

```python
class SourceDocScore(BaseModel):
    score: float          # 0.0 (not plagiarised) to 1.0 (definitely plagiarised)
    is_likely_source: bool
    reasoning: str        # free-text explanation
```

## Text pair selection

For each candidate source document, the top-K chunk pairs by embedding cosine similarity are selected from `embedding_candidates_suspicious.parquet`. This ensures the LLM sees the most semantically similar passages — the ones most likely to be plagiarised.

Default: **3 pairs per source document**.

## Output

### `llm_scores_df.parquet`

| Column | Description |
|---|---|
| `source_doc_id` | Candidate source document |
| `llm_score` | Plagiarism likelihood score 0–1 |
| `llm_is_likely_source` | Boolean verdict |
| `llm_reasoning` | LLM explanation text |

## Notes

- The LLM re-ranking is a **second-pass filter**, not a primary retrieval step. Recall at the retrieval stage (stage 04) is critical — if the true source is not in the top-N, the LLM cannot recover it.
- Chunk pairs are truncated to 600 characters each to fit within a reasonable prompt length.
- Temperature is set to 0.1 to keep responses deterministic and consistent.
- The cached `llm_scores_df.parquet` is used by the Streamlit app when the pipeline has already been run for a document.
