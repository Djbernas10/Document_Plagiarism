# 07 — Streamlit App

Interactive web UI that runs the full plagiarism detection pipeline end-to-end for a selected suspicious document and visualises the results.

## What it does

```
Sidebar: select suspicious document + configure parameters
        │
        ▼
Stage 1 — Source Retrieval
  TF-IDF → ESA → LSA → Embeddings → Score Fusion
        │
        ▼
Stage 2 — LLM Re-ranking
  Build top-K chunk pairs per candidate → Ollama LLM scoring
        │
        ▼
Results: Top-5 metric cards + reasoning expanders + full score table
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Full Streamlit application |

## Running

```bash
# From the project root
streamlit run scripts/final/07_streamlit_app/app.py
```

The app changes CWD to `scripts/final/` on startup so that relative paths in `source_retrieval_branches.py` resolve correctly.

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `ollama` | LLM inference via local Ollama server |
| `pandas` / `pyarrow` | Parquet I/O |

The LLM re-ranking stage requires **Ollama** running locally. Default model: `gemma4:e4b`.

## Sidebar controls

| Control | Description |
|---|---|
| Suspicious document | Drop-down of all doc IDs from `suspicious_chunks.parquet` |
| Fused top-N candidates | How many fused source docs to pass to the LLM (5–50) |
| Re-run GPU embeddings | If checked, triggers the ROCm Docker container to re-compute embeddings; otherwise loads the cached parquet |
| Ollama model | Model name passed to Ollama (any locally pulled model) |
| Chunk pairs per source doc | How many suspicious↔source text pairs the LLM sees per candidate (1–5) |

## Results display

- **Top-5 metric cards**: LLM score (0–1) and likely/unlikely verdict for each top-5 source document.
- **Expandable reasoning cards**: Full LLM explanation for each of the top-5 candidates.
- **All LLM scores table**: Full ranked list with score, verdict, and reasoning.
- **Source Retrieval Fusion table**: The pre-LLM fused top-N with per-branch scores.

## Caching

- `@st.cache_data` is used for the suspicious doc ID list (loaded once per session).
- The fused top-N result is written to `scripts/final/top20_df.parquet` after each run and shown as a cached preview when the app starts without a fresh run.
- Session state (`st.session_state`) persists the pipeline results across widget interactions within the same session.
