# 04 — Source Retrieval

Queries the pre-built indexes (from `03_index_creation`) to find candidate source documents for a given suspicious document, then fuses the four branch scores into a single ranked list.

## What it does

```
suspicious document
        │
        ├── TF-IDF lookup  ──► top-50 source docs by max char n-gram similarity
        ├── ESA lookup     ──► top-50 source docs by max concept-space similarity
        ├── LSA lookup     ──► top-50 source docs by max latent semantic similarity
        └── Embeddings     ──► top-50 source docs by max dense vector similarity
                │
                ▼
        Score fusion (weighted mean + max)
                │
                ▼
        top-N fused candidates (default 20)
```

## Files

| File | Purpose |
|---|---|
| `source_retrieval_branches.py` | All retrieval logic + score fusion |

## Key functions

| Function | Description |
|---|---|
| `search_artifact(artifact_type)` | Sets global path variables (`ARTIFACT_DIR`, output paths) for the chosen branch. Call before each `*_lookup()`. |
| `tf_idf_lookup()` | Loads the sharded hashed TF-IDF index, transforms suspicious chunks, scores all source shards, and returns the top-50 source documents by max chunk score. |
| `lsa_lookup()` | Loads TF-IDF vectorizer + TruncatedSVD, projects suspicious chunks into LSA space, and returns top-50 by max cosine similarity. |
| `esa_lookup()` | Loads the sparse ESA concept-vector index, computes sparse cosine similarity, returns top-50 by max score. |
| `embeddings_lookup(doc_id)` | (GPU path) Loads FAISS index + Qwen3-Embedding-0.6B, encodes suspicious chunks, and returns top-50 by max inner-product score. |
| `embedding_run(doc_id)` | Launches the ROCm Docker container to run `scripts/embeddings.py` for the GPU embedding lookup. |
| `mean_doc_score_aggreg(...)` | Fuses the four branch top-50 DataFrames via weighted score combination. Final score = 30% weighted mean + 70% weighted max (max weighted higher because one strong local match reliably signals plagiarism). Default weights: TF-IDF 10%, ESA 15%, LSA 25%, Embeddings 50%. |
| `lookup_pipeline(doc_id, ...)` | Orchestrates all branches in sequence and returns the fused top-N DataFrame. |

## Score fusion design

Each branch produces two scores per source document:
- **max score** — score of the single best-matching chunk pair
- **mean score** — average score across all matched chunk pairs

The final fusion is:

```
weighted_mean_score = Σ w_i × mean_score_i
weighted_max_score  = Σ w_i × max_score_i
final_score         = 0.30 × weighted_mean_score + 0.70 × weighted_max_score
```

The max component is weighted 70% because plagiarism is often local — one strongly matching passage is a better indicator than a globally high average.

## Output columns (fused DataFrame)

| Column | Description |
|---|---|
| `final_rank` | Rank by `final_score` |
| `source_doc_id` | `partX__source-documentNNNNN.txt` |
| `final_score` | Fused score |
| `methods_found_count` | Number of branches that returned this source doc (1–4) |
| `found_by_tfidf/esa/lsa/emb` | Boolean presence per branch |
| `tfidf/esa/lsa/emb_mean_score` | Per-branch mean chunk score (0 if absent) |
| `tfidf/esa/lsa/emb_max_score` | Per-branch max chunk score (0 if absent) |

## Running

```python
from source_retrieval_branches import lookup_pipeline, search_artifact

# CPU-only (TF-IDF + ESA + LSA + cached embedding results)
result_df = lookup_pipeline("part1__suspicious-document00007.txt", run_embeddings=False, top_n=20)

# With GPU re-run (requires Docker ROCm container)
result_df = lookup_pipeline("part1__suspicious-document00007.txt", run_embeddings=True, top_n=20)
```

Or run directly:
```bash
python scripts/final/04_source_retrieval/source_retrieval_branches.py
```
