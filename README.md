# Academic Plagiarism Detection Empowered by LLMs


![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%C2%B7%20gemma4%3A26b-000000?logo=ollama&logoColor=white)
![FAISS](https://img.shields.io/badge/Retrieval-FAISS%20%C2%B7%20Qwen3-005571)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)
![ROCm](https://img.shields.io/badge/GPU-AMD%20ROCm-ED1C24?logo=amd&logoColor=white)
![Status](https://img.shields.io/badge/status-not%20maintained-lightgrey)

This repository contains the implementation developed for a project report on
academic document plagiarism detection. The final system is an extrinsic
plagiarism-detection pipeline: it receives suspicious documents, retrieves likely
source documents, asks a local LLM to confirm source alignment, merges detected
spans, and evaluates the result against ground-truth annotations.

The project is primarily a research artifact, not a production service. It is
designed to make experiments reproducible on a single GPU workstation while
keeping enough instrumentation to explain why a document was detected, missed, or
rejected.

### Use of AI assistance

AI tools (large language models) were used during this work to assist with the implementation and documentation: for example, refactoring,
debugging, and drafting explanatory text. All technical decisions, experimentaldesign, results, and their interpretation are the author's own, and all
AI-assisted output was reviewed and verified against the actual implementation andexperimental data before being incorporated. 
This disclosure is provided in theinterest of transparency.


## Final status

The project report implementation is complete.

- PAN 2011 subset evaluation was completed on 308 suspicious documents.
- A custom curated dataset was added for robustness checks.
- The Streamlit UI can launch single-document or batch runs.
- The retrieval pipeline supports ESA, LSA, optional TF-IDF, and Qwen/FAISS
  embeddings. The project report text calls this branch "Corpus-Term" instead of
  "ESA," because it is a TF-IDF vector space fit over this project's own
  source-document collection, not literal Explicit Semantic Analysis, which
  needs a Wikipedia-derived concept space (see
  `scripts/final/03_index_creation/build_esa_index_custom.py` for the "not
  Wikipedia ESA" note already in that file). Code identifiers, file names,
  CLI flags such as `--skip-esa`, and artifact directories like
  `artifacts/esa/` still use `esa` throughout. Only the name used in the
  project report changed, not the codebase.
- The LLM confirmation stage uses Ollama, with `gemma4:26b` as the final model.
- The system can run locally from the CLI or through Docker Compose.
- Streamlit-triggered runs are saved to local run logs for debugging.

The diagram below maps the forms of plagiarism to the detection techniques used:

![Forms of plagiarism and the suitability of detection methods](images/forms_of_plagiarism.png)

*Forms of plagiarism and the suitability of detection methods. This system
targets the character-, syntax-, and semantics-preserving forms by combining
vector-space models (TF-IDF), LSA/ESA, and embedding-based retrieval, followed
by LLM confirmation.*

## Architecture

![End-to-end architecture of the plagiarism detection pipeline](images/Architecture_poster.png)

*End-to-end architecture: offline preprocessing and indexing, source retrieval
with fusion and gating, LLM candidate verification, and span extraction. Solid
boxes are the default path; dashed boxes are opt-in or experimental stages.*

The system is a two-stage extrinsic plagiarism detection pipeline. A suspicious
document comes in through the Streamlit interface and is compared against a
reference collection of source documents: PAN-PC-11 as the primary benchmark, or
a custom collection of 30 source academic papers (see the note below on a
duplicate in that collection). To avoid comparing every pair of documents, the
pipeline narrows the candidates with a retrieval stage and a two-gate filter
before running any expensive LLM inference.

> **Known issue (2026-09-22): the custom dataset's 30 source files contain 29
> distinct papers.** `source-document00020` and `source-document00027` are two
> PDF exports of the same paper: Potthast, Eiselt, Barrón-Cedeño, Stein &
> Rosso, "Overview of the 3rd International Competition on Plagiarism
> Detection" (CLEF 2011), which is also the paper this project cites for the
> official plagdet formula (`scripts/final/compute_plagdet_official.py`).
> Neither document is a ground-truth source for any suspicious document in
> `datasets/custom_dataset/ground_truth/`, so no reported detection,
> precision/recall, or plagdet figure changes. If the corpus size is cited in
> the project report, use "30 files, 29 distinct papers". Details are in
> [datasets/custom_dataset/README.md](datasets/custom_dataset/README.md).

**Document Preparation and Indexing (offline, one-time).** Source documents are
cleaned, split into overlapping word-window chunks, and enriched with document
identifiers, chunk identifiers, and character offsets, then stored as Parquet.
Indexing builds the branch-specific structures reused across all queries:
compressed sparse NumPy shards (TF-IDF), a fitted vectorizer plus SVD model (LSA),
a sparse TF-IDF matrix (ESA), and a FAISS vector index (embeddings).

**Plagiarism Detection Process (per suspicious document).** The suspicious
document is preprocessed identically, queried against each retrieval branch, and
each branch's chunk-level similarities are aggregated to the document level by
taking the maximum score across chunk pairs. The embedding branch runs on GPU
through a dedicated embedding service; the other branches run on CPU. The four
branch scores are combined with a weighted fusion formula into a single ranked
candidate list. A two-gate filter then decides which candidates reach the LLM:
Gate 1 is an absolute floor on the top-1 fusion score (below it the document is
treated as clean and the LLM stage is skipped), and Gate 2 keeps only candidates
within a relative gap of the top-1 score. In addition, the top candidates of each
individual branch are guaranteed a place in the forwarded set (branch union), so
a source found by only one branch is not lost to fusion averaging.

**Candidate Verification and Span Extraction.** The forwarded candidates are read
by the LLM (`gemma4:26b` via Ollama, using a discrete plagiarism-likelihood
rubric), which produces a structured verdict. Confirmed candidates are mapped back
to character-level spans using the preserved chunk offsets, and adjacent spans are
merged when the gap between them is below a configurable threshold, producing
clean non-fragmented detections. Results are returned to the Streamlit interface
and, for PAN-PC-11, scored against the ground-truth XML annotations with the
official plagdet metric.

```text
Offline: source docs -> clean -> chunk -> Parquet
                              -> build indices (TF-IDF | LSA | ESA | FAISS)  [reused]

Suspicious document
       |
       v
Source retrieval
  - ESA
  - LSA
  - optional TF-IDF
  - Qwen3 embeddings + FAISS (GPU service)
       |
       v
Candidate fusion and gating
  - weighted fusion (0.20 * mean + 0.80 * max)
  - Gate 1 minimum top-1 score
  - Gate 2 relative-gap filtering
  - branch-union recovery
  - optional: soft Gate 1, cross-encoder rerank
       |
       v
LLM source confirmation
  - Ollama
  - gemma4:26b
  - discrete plagiarism rubric
  - optional: GPT-2 perplexity cap (word-salad -> 0.25)
       |
       v
Span merging and evaluation
  - per-document metrics
  - retrieval recall
  - PAN-style plagdet summaries
```

An editable, all-in-one diagram of the full pipeline and the containerized
deployment topology is available in `Architecture_poster.drawio` (open with
[draw.io](https://app.diagrams.net/)). Solid boxes are the default path; dashed
boxes are opt-in / experimental stages.

## Repository layout

| Path | Purpose |
|------|---------|
| `scripts/final/run_pipeline.py` | Main end-to-end pipeline runner |
| `scripts/final/04_source_retrieval/source_retrieval_branches.py` | Retrieval branches, fusion, and embedding backend selection |
| `scripts/embeddings.py` | Qwen embedding lookup and FAISS retrieval |
| `scripts/embedding_service.py` | HTTP wrapper used by the Docker embeddings service |
| `scripts/final/07_streamlit_app/app.py` | Streamlit UI and subprocess runner |
| `docker-compose.yml` | Multi-service deployment for Ollama, embeddings, Streamlit, and optional batch pipeline |
| `docker_files/` | Dockerfiles and ROCm helper scripts |
| `datasets/custom_dataset/` | Custom dataset: suspicious documents, ground-truth XML, and the scripts that build them |
| `datasets/`, `artifacts/` | Placeholders for local datasets, models, and indexes, which are ignored by Git |
| `scripts/final/0*_*/` | Preprocessing, ground-truth parsing, index creation, retrieval, alignment, analytics, and Streamlit stages |
| `scripts/final/pipeline_results*/` | READMEs and markdown summaries of the recorded runs (result files stay local) |
| `human_comparison_examples/` | Confirmed detections laid out for manual review |
| `images/`, `Architecture_poster.drawio` | Architecture and method diagrams |

## Requirements

Core environment:

- Python 3.12
- `uv`
- Docker Desktop or Docker Engine
- Ollama, either native on the host or through Compose

Development hardware used:

- AMD GPU through ROCm on WSL2
- `rocm/pytorch:latest` for the embedding service
- `ollama/ollama:rocm` for the LLM service

The Docker Compose GPU settings are intentionally tuned for AMD ROCm on WSL2:

- `/dev/dxg`
- `/usr/lib/wsl/lib/libdxcore.so`
- `/opt/rocm/lib/librocdxg.so`
- `HSA_ENABLE_DXG_DETECTION=1`

Linux-native ROCm or NVIDIA deployments will need adjusted GPU passthrough.

## Setup

Install Python dependencies:

```bash
uv sync
```

## Local data and artifact layout

Datasets, generated indexes, model weights, and run outputs are not stored in
Git. In a fresh clone, create the local-only folders below before running the
pipeline:

```bash
mkdir -p datasets/PAN2011
mkdir -p datasets/PAN2025
mkdir -p datasets/processed
mkdir -p datasets/custom_dataset
mkdir -p artifacts/models
mkdir -p artifacts/embeddings
mkdir -p artifacts/embeddings_custom
mkdir -p scripts/final/pipeline_results
mkdir -p scripts/final/pipeline_results_custom
```

On Windows PowerShell, the equivalent is:

```powershell
New-Item -ItemType Directory -Force `
  datasets/PAN2011, `
  datasets/PAN2025, `
  datasets/processed, `
  datasets/custom_dataset, `
  artifacts/models, `
  artifacts/embeddings, `
  artifacts/embeddings_custom, `
  scripts/final/pipeline_results, `
  scripts/final/pipeline_results_custom
```

Place or generate the required data as follows:

| Folder | What to put there | Required for |
|--------|-------------------|--------------|
| `datasets/PAN2011/usable/` | Raw PAN-PC-11 external-detection corpus (see [Getting the PAN-PC-11 corpus](#getting-the-pan-pc-11-corpus)) | PAN 2011 preprocessing and ground truth |
| `datasets/processed/PAN2011_300/` | Processed PAN chunks and source/suspicious parquet files | PAN 2011 pipeline runs |
| `datasets/processed/custom_300/` | Processed custom-dataset chunks and retrieval inputs | Custom pipeline runs |
| `datasets/processed/custom_ground_truth/` | Custom ground-truth parquet generated from XML annotations | Custom evaluation |
| `datasets/custom_dataset/source_documents/` | The 30 source papers as PDFs, plus the `.txt` files extracted from them (the suspicious documents and ground truth are already committed) | Custom dataset rebuilds and runs |
| `artifacts/models/Qwen3-Embedding-0.6B/` | Local Hugging Face embedding model directory | Live embedding lookup |
| `artifacts/embeddings/` | PAN 2011 FAISS embedding indexes and metadata | PAN 2011 embedding retrieval |
| `artifacts/embeddings_custom/` | Custom-dataset FAISS embedding indexes and metadata | Custom embedding retrieval |
| `artifacts/esa*`, `artifacts/lsa*`, `artifacts/tfidf*` | ESA, LSA, and TF-IDF indexes built by the index-creation scripts | Retrieval branches |
| `scripts/final/pipeline_results/` | PAN 2011 run outputs | PAN 2011 results |
| `scripts/final/pipeline_results_custom/` | Custom dataset run outputs | Custom results |

### Getting the PAN-PC-11 corpus

The PAN 2011 benchmark is the PAN Plagiarism Corpus 2011 (PAN-PC-11) by
Potthast, Stein, Eiselt, Barrón-Cedeño and Rosso, published on Zenodo under
CC BY 4.0:

- Record: <https://zenodo.org/records/3250095>
- DOI: [10.5281/zenodo.3250095](https://doi.org/10.5281/zenodo.3250095)

Download both parts of the multi-volume RAR archive
(`pan-plagiarism-corpus-2011.part1.rar`, about 1.0 GB, and
`pan-plagiarism-corpus-2011.part2.rar`, about 704 MB) into
`datasets/PAN2011/3250095/` and extract `part1` (7-Zip or `unrar` picks up
`part2` automatically). The archive contains `external-detection-corpus/` and
`intrinsic-detection-corpus/`. This project uses only the external-detection
corpus. Copy its two folders to `datasets/PAN2011/usable/`:

```text
datasets/PAN2011/usable/
├── source-document/       # part1 ... part23, 11,093 .txt + .xml
└── suspicious-document/   # part1 ... part23, 11,093 .txt + .xml (ground-truth annotations)
```

The embedding model can be downloaded with:

```bash
uv run hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir artifacts/models/Qwen3-Embedding-0.6B
```

For the final pipeline to run without rebuilding indexes, the important local
directories are:

```text
datasets/processed/PAN2011_300/
datasets/processed/custom_300/
datasets/processed/custom_ground_truth/
artifacts/models/Qwen3-Embedding-0.6B/
artifacts/embeddings/
artifacts/embeddings_custom/
artifacts/esa*/
artifacts/lsa*/
```

These folders are ignored on purpose because they hold large downloaded,
generated, or machine-specific files. The repository stores the code and
documentation, not the local experiment data.

## Running locally

Use the local CLI when debugging the algorithm or reproducing a known experiment.

Example known-good custom document run:

```bash
uv run python scripts/final/run_pipeline.py \
  --dataset custom \
  --doc-id suspicious-document00007.txt \
  --skip-tfidf \
  --retrieval-recall-k 5 \
  --run-embeddings \
  --relative-gap 0.85 \
  --debug-llm \
  --fresh \
  --ollama-model gemma4:26b
```

Useful local flags:

| Flag | Meaning |
|------|---------|
| `--dataset pan2011` / `--dataset custom` | Select corpus |
| `--doc-id ID` | Process one suspicious document |
| `--docs N` | Process first N documents |
| `--fresh` | Ignore cached per-document results |
| `--skip-tfidf` | Skip the slow TF-IDF branch |
| `--skip-esa` | Skip ESA if memory is tight |
| `--skip-llm` | Retrieval-only run |
| `--run-embeddings` | Force live embedding lookup |
| `--embeddings-backend local` | Run `scripts/embeddings.py` directly |
| `--embeddings-backend http` | Use the embedding HTTP service |
| `--ollama-model gemma4:26b` | Select Ollama model |
| `--debug-llm` | Save LLM prompts/debug JSON |

For local Ollama, start the Ollama desktop/app or run:

```bash
ollama serve
ollama pull gemma4:26b
```

## Running with Docker Compose

The default Compose stack starts:

- `ollama`
- `ollama-models`
- `embeddings`
- `streamlit`

It does not auto-run the batch pipeline. Pipeline execution is triggered from the
Streamlit UI.

From WSL, run:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The `ollama-models` one-shot service checks whether `gemma4:26b` exists in the
Compose Ollama volume and pulls it if missing. Normal `docker compose down` does
not delete this volume. `docker compose down -v` does delete it and will force a
model re-download.

Run only the optional batch pipeline profile:

```bash
docker compose --profile pipeline up pipeline
```

Recreate Streamlit after changing Compose settings:

```bash
docker compose up -d --force-recreate streamlit
```

The Compose file bind-mounts `./scripts` into the containers, so edits to pipeline
code are visible without rebuilding the image.

## Streamlit UI

The Streamlit app is intended for quick single-document experiments and small
batches. It exposes:

- dataset selection
- selected document or batch mode
- retrieval parameters
- LLM model and threshold
- embedding backend
- fresh/cache mode
- LLM debug prompt dumping

The UI launches `scripts/final/run_pipeline.py` as a subprocess with the selected
parameters. It writes a timestamped log for every run.

Run logs are saved under:

```text
scripts/final/pipeline_results/run_logs/
scripts/final/pipeline_results_custom/run_logs/
```

These logs include:

- exact command
- `OLLAMA_HOST`
- `OLLAMA_TIMEOUT_SECONDS`
- `EMBEDDINGS_BACKEND`
- `EMBEDDINGS_URL`
- combined stdout/stderr

## Runtime notes

Inside Docker, `localhost` is not the host or a sibling container. The pipeline
therefore uses:

```text
OLLAMA_HOST=http://ollama:11434
EMBEDDINGS_URL=http://embeddings:8000
```

For local CLI runs, `OLLAMA_HOST` is normally left unset so the Ollama Python
package uses the local default.

LLM confirmation can be slow. `gemma4:26b` may take several minutes per candidate
source document, especially with many chunk pairs. The runner prints live
progress:

```text
[LLM] Scoring 4 candidate source docs with Ollama...
[LLM] (1/4) scoring source-document00016.txt with 25 chunk pair(s)...
[LLM] (1/4) done source-document00016.txt: score=0.950 likely=True t=190.2s
```

The default LLM request timeout is controlled by:

```text
OLLAMA_TIMEOUT_SECONDS=600
```

## Troubleshooting

Port 11434 is already in use:

```bash
# Stop native Ollama, or change the Compose port mapping.
docker compose up -d ollama
```

Ollama model missing inside Compose:

```bash
docker compose exec ollama ollama pull gemma4:26b
docker compose exec ollama ollama list
```

Check whether the LLM is loaded/running:

```bash
docker compose logs -f ollama
docker compose exec ollama ollama ps
```

Embedding service health:

```bash
curl http://localhost:8000/health
```

If embedding ROCm fails inside Docker but works in WSL, run Compose from WSL and
confirm the WSL ROCm library paths exist:

```bash
ls -l /usr/lib/wsl/lib/libdxcore.so
ls -l /opt/rocm/lib/librocdxg.so
```

## Outputs

Pipeline outputs are written to the folders below. They are ignored by Git;
only the run READMEs and markdown summaries in these folders are committed.

```text
scripts/final/pipeline_results/
scripts/final/pipeline_results_custom/
```

Main output files:

| File | Meaning |
|------|---------|
| `analytics_summary.parquet` | Per-document precision/recall/F1 |
| `retrieval_recall.parquet` | Retrieval recall@K |
| `per_doc/<doc_id>.parquet` | Final detected spans |
| `llm_debug/<doc_id>/*.json` | Saved LLM prompts when `--debug-llm` is enabled |
| `run_logs/*.log` | Streamlit-launched run logs |

The recorded results of the final runs are in the committed READMEs:

- [scripts/final/pipeline_results/old_results/308_docs_full_v3/README.md](scripts/final/pipeline_results/old_results/308_docs_full_v3/README.md): 308-document PAN 2011 run
- [scripts/final/pipeline_results/tfidf_ablation/README.md](scripts/final/pipeline_results/tfidf_ablation/README.md): PAN 2011 TF-IDF ablation
- [scripts/final/pipeline_results_custom/tfidf_ablation/README.md](scripts/final/pipeline_results_custom/tfidf_ablation/README.md): custom dataset TF-IDF ablation

## Human comparison examples

The [`human_comparison_examples/`](human_comparison_examples/) folder has
side-by-side samples of suspicious and source passages that the pipeline
confirmed as plagiarism in the final PAN 2011 run. Each example shows the
highest-similarity chunk pair the LLM was given, with the matching phrases in
bold, so a reader can judge whether the detection is correct. The set covers
low-obfuscation (near-verbatim synonym swaps) and high-obfuscation (heavy
paraphrase) cases. The folder's [README](human_comparison_examples/README.md)
has the index and a short reading guide.

---

## Disclaimer

This repository is no longer maintained. It is published as a completed
research artifact and provided as-is, without support, updates, or guarantees.
Issues and pull requests may not be reviewed.
