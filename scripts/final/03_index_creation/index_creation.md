# 03 — Index Creation

Builds offline retrieval indexes over the source-document chunk Parquets produced by `01_preprocessing`. Each notebook targets one retrieval branch. The resulting index artifacts are stored under `artifacts/` and loaded at query time by `04_source_retrieval`.

## What it does

```
source_chunks_lsa_esa.parquet  ─────┬──► TF-IDF  sharded sparse matrix  (artifacts/tfidf_hashing/)
                                    ├──► LSA     SVD + dense vectors      (artifacts/lsa/)
                                    └──► ESA     sparse TF-IDF concept vectors (artifacts/esa/)

source_chunks_embeddings.parquet ───────► FAISS  dense vector index      (artifacts/embeddings/)
```

## Files

| File | Purpose |
|---|---|
| `index_creation_TF_IDF.ipynb` | Builds a sharded hashed TF-IDF index over the source corpus |
| `index_creation_LSA.ipynb` | Trains a TF-IDF vectorizer + TruncatedSVD and saves dense LSA vectors |
| `index_creation_ESA.ipynb` | Builds an Explicit Semantic Analysis index using Wikipedia concept vectors |
| `index_creation_embeddings.ipynb` | Encodes source chunks with Qwen3-Embedding-0.6B and stores them in a FAISS index |

## Artifact layout

### TF-IDF (`artifacts/tfidf_hashing/{char|word}/`)

| File | Description |
|---|---|
| `tfidf_vectorizer.joblib` | Fitted `HashingVectorizer` |
| `tfidf_idf.npy` | IDF weights array |
| `source_tfidf_metadata.parquet` | Row-index → chunk_id / doc_id mapping |
| `tfidf_shards.parquet` | Shard manifest (path, start_row, num_rows) |
| `tfidf_shard_XXXX.npz` | Sparse TF-IDF matrix shards |
| `tfidf_config.joblib` | Hyperparameter snapshot |

### LSA (`artifacts/lsa/`)

| File | Description |
|---|---|
| `tfidf_vectorizer.joblib` | Fitted `TfidfVectorizer` used as input to SVD |
| `svd_model.joblib` | Fitted `TruncatedSVD` model |
| `source_lsa_vectors.npy` | Dense L2-normalised LSA vectors, shape `(N_chunks, n_components)` |
| `source_lsa_metadata.parquet` | Row-index → chunk_id / doc_id mapping |

### ESA (`artifacts/esa/`)

| File | Description |
|---|---|
| `esa_tfidf_vectorizer.joblib` | TF-IDF vectorizer whose vocabulary maps terms to Wikipedia concept axes |
| `source_esa_vectors.npz` | Sparse L2-normalised ESA concept vectors |
| `source_esa_metadata.parquet` | Row-index → chunk_id / doc_id mapping |
| `esa_config.joblib` | Hyperparameter snapshot |

### Embeddings (`artifacts/embeddings/embeddings_qwen06b/`)

| File | Description |
|---|---|
| `faiss.index` | FAISS `IndexFlatIP` (inner product = cosine on L2-normalised vectors) |
| `source_embedding_metadata.parquet` | Row-index → chunk_id / doc_id mapping |
| `embedding_config.joblib` | Model path, dimension, normalisation flag |

## Notes

- Index creation is a **one-time offline step**. Re-run only when the source chunk Parquet changes (e.g. after re-preprocessing with different chunking parameters).
- TF-IDF sharding is necessary because the full source sparse matrix is too large for RAM. Each shard is searched sequentially at query time.
- The embedding index requires a GPU (ROCm Docker container). See the notebook for the Docker setup.
- All indexes store a metadata Parquet so that a result row index can be mapped back to `chunk_id` and `doc_id` without loading the full chunk Parquet.
