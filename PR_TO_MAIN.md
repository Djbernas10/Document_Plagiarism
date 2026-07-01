# PR to main checklist

Use this checklist before opening the final PR.

## Include

Core containerization and runner files:

- `README.md`
- `.gitignore`
- `.dockerignore`
- `docker-compose.yml`
- `docker_files/`
- `requirements.txt`
- `scripts/requirements-rocm.txt`
- `scripts/embedding_service.py`
- `scripts/embeddings.py`
- `scripts/final/run_pipeline.py`
- `scripts/final/04_source_retrieval/source_retrieval_branches.py`
- `scripts/final/07_streamlit_app/app.py`

Keep placeholder files that preserve required directories:

- `datasets/.gitkeep`
- `datasets/processed/.gitkeep`
- `artifacts/.gitkeep`

Include compact documentation and curated summaries when useful:

- `scripts/final/*.md`
- `scripts/final/*/*.md`
- `datasets/custom_dataset/README.md`
- `datasets/custom_dataset/EXPERIMENTS_SUMMARY.md`

## Do not include

Large or regenerated local files:

- `artifacts/`
- `datasets/PAN2011/`
- `datasets/PAN2025/`
- `datasets/processed/`
- `datasets/custom_dataset/source_documents/`
- `datasets/custom_dataset/suspicious_documents_pdf/`
- `scripts/final/pipeline_results*/per_doc/`
- `scripts/final/pipeline_results*/llm_debug/`
- `scripts/final/pipeline_results*/run_logs/`
- `*.parquet`
- `*.docx`
- `*.7z`
- `.venv/`
- `.uv-cache/`

Local agent/editor state:

- `.codex/`
- `.agents/`
- `.claude/`
- `.$*`

## Sanity commands

```bash
git status --short
git diff --stat
```

Optional checks:

```bash
python -m py_compile scripts/final/run_pipeline.py scripts/final/07_streamlit_app/app.py scripts/embedding_service.py scripts/embeddings.py
docker compose config
```

The final PR should contain code, config, and documentation only. It should not
contain model weights, datasets, transient run logs, debug prompts, or generated
thesis exports.
