# 01 — Preprocessing

Converts raw PAN corpus `.txt` files into structured Parquet tables ready for indexing by every downstream retrieval branch (TF-IDF, LSA, ESA, Embeddings).

## What it does

```
raw .txt files
      │
      ▼
document-level Parquet   (raw_text + clean_text per file)
      │
      ▼
canonical chunks Parquet  (overlapping word-windows with char offsets)
      │
      ├─── LSA/ESA chunks  (lowercased, stripped punctuation, stopwords removed)
      └─── Embedding chunks (light normalisation, natural sentence structure kept)
```

## Files

| File | Purpose |
|---|---|
| `preprocessing.py` | All preprocessing logic |
| `preprocess_data.ipynb` | Notebook to run the pipeline interactively |

## Key functions

| Function | Description |
|---|---|
| `clean_pan_source_text` | Unicode normalisation, quote/dash standardisation, whitespace collapse. **All char offsets reference this text.** |
| `chunk_clean_text` | Sliding window chunker. Default: 300 words, 150-word overlap, ≥60 words minimum. |
| `collect_documents_to_parquet` | Streams `.txt` files to Parquet in batches; avoids loading the full corpus into RAM. |
| `build_chunks_from_document_parquet` | Reads document Parquet row-group by row-group and writes a chunk Parquet. |
| `normalize_for_lsa_esa` | Aggressive normalisation for bag-of-words models: lowercase, remove punctuation, replace numbers with `NUM`, remove stopwords. |
| `normalize_for_embeddings` | Minimal normalisation: keeps sentence structure intact so transformer models see natural text. |
| `full_preprocessing_pipeline` | Orchestrates all steps in order, producing the eight Parquet files listed below. |

## Output Parquet files (written to `datasets/processed/PAN2011_300/`)

| File | Rows | Description |
|---|---|---|
| `source_documents.parquet` | 1 per source doc | Raw + clean text with word/char counts |
| `suspicious_documents.parquet` | 1 per suspicious doc | Same schema as source |
| `source_chunks.parquet` | ~N×chunks per doc | Canonical chunks with char offsets |
| `suspicious_chunks.parquet` | ~N×chunks per doc | Same schema as source chunks |
| `source_chunks_lsa_esa.parquet` | Same rows | Adds `lsa_esa_text` column |
| `suspicious_chunks_lsa_esa.parquet` | Same rows | Adds `lsa_esa_text` column |
| `source_chunks_embeddings.parquet` | Same rows | Adds `embedding_text` + `estimated_token_count` |
| `suspicious_chunks_embeddings.parquet` | Same rows | Adds `embedding_text` + `estimated_token_count` |

## Running

```bash
# From the project root
python scripts/final/01_preprocessing/preprocessing.py
```

Or open `preprocess_data.ipynb` for an interactive run.

## Important design notes

- **`doc_id` format**: `partX__<filename>` — includes the part folder to avoid collisions when the same filename appears in multiple parts.
- **`chunk_id` format**: `{doc_stem}_c{index:04d}` — zero-padded index for stable lexicographic ordering.
- **Char offsets** (`start_char`, `end_char`) refer to positions in `clean_text`, not the raw file. The XML ground-truth parser uses raw offsets — keep this distinction in mind when comparing spans.
- The same `chunk_id` / `chunk_text` / `start_char` / `end_char` columns are shared across all three branch-specific Parquet files. Only the text column changes (`chunk_text` → `lsa_esa_text` or `embedding_text`).
