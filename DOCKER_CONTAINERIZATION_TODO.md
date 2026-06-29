# Containerizing the pipeline — handoff scope

## Goal

Ship the plagiarism-detection pipeline (Streamlit app + retrieval + LLM confirmation +
GPU embeddings) as a multi-container setup that runs with one `docker compose up`,
instead of the current manual, host-OS-dependent setup (Ollama installed natively on
Windows, a hand-started ROCm container, paths hardcoded to `localhost`).

This file documents current state, target architecture, and concrete steps. It does not
implement anything — that's the next task.

---

## Current state (as of this commit)

### What exists
- `docker_files/dockerfile_streamlit` — **empty placeholder file, 0 bytes**.
- `docker_files/dockerfile_preprocessing` — **empty placeholder file, 0 bytes**.
- `docker_files/run_rocm_container.sh` — a manual `docker run` (not a Dockerfile) that
  starts the upstream `rocm/pytorch:latest` image, bind-mounts the repo, and
  `pip install`s `scripts/requirements-rocm.txt` on container start. This is the
  container the pipeline currently calls via `docker exec docplag-rocm python
  scripts/embeddings.py ...` (see below). It is **not idempotent** — every fresh
  container re-runs `pip install` and takes minutes to become ready.
- No `docker-compose.yml` anywhere in the repo.
- Ollama is run **natively on the Windows host** (`ollama serve`, default port 11434),
  not in a container.

### How the pieces currently talk to each other (all on the host, no real containers)
- `scripts/final/run_pipeline.py` (`embedding_run()` in
  `scripts/final/04_source_retrieval/source_retrieval_branches.py:1448-1460`) shells out
  via `subprocess.run(["docker", "exec", "docplag-rocm", "python", "scripts/embeddings.py",
  "--doc_id", ..., "--dataset", ...])`. This assumes a container literally named
  `docplag-rocm` is already running on the same Docker daemon as the host process.
- `run_pipeline.py:137` does `from ollama import chat` (the `ollama` Python package),
  which defaults to `http://localhost:11434`. **Not configurable via env var today** —
  it's whatever the `ollama` package's internal default is.
- `scripts/final/07_streamlit_app/app.py:452` hardcodes
  `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` — same problem, in a
  different client.
- `scripts/embeddings.py` (run *inside* the ROCm container) reads/writes Parquet files
  under `datasets/processed/{PAN2011_300,custom_300}/` and
  `artifacts/{embeddings,embeddings_custom}/embeddings_qwen06b/`, and loads the model
  from `artifacts/models/Qwen3-Embedding-0.6B` — all resolved as **relative paths**,
  which only work because the container bind-mounts the whole repo at
  `/workspace/Document_Plagiarism` and `run_rocm_container.sh` sets `-w` to that path.

### Why this matters
None of `localhost:11434` assumptions survive moving Ollama into its own container —
inside a container, `localhost` means that container, not a sibling container. This is
the single most important thing to fix; everything else is packaging.

---

## Target architecture

Four services, one `docker-compose.yml`, one shared Docker network:

1. **`ollama`** — official `ollama/ollama` image (has ROCm/CUDA variants), exposes 11434
   internally. Needs a named volume for model storage (`gemma4:26b` is ~17GB, don't bake
   it into an image layer) and GPU passthrough.
2. **`embeddings`** (replaces the manual ROCm container) — built from a **real**
   `docker_files/dockerfile_preprocessing` (currently empty) based on `rocm/pytorch` (or
   slim if a CUDA box, see open question below), with `scripts/requirements-rocm.txt`
   baked into the image at build time (not `pip install`ed on every container start).
   Needs GPU passthrough and a volume/bind-mount for `datasets/processed/`,
   `artifacts/embeddings*/`, and `artifacts/models/`.
3. **`pipeline`** — runs `scripts/final/run_pipeline.py`. Talks to `ollama` over the
   Docker network (`http://ollama:11434`) and to `embeddings` (replacing the current
   `docker exec docplag-rocm ...` subprocess call with either an HTTP call or
   `docker exec` against the compose-managed container name — see Step 3 below for the
   recommended approach).
4. **`streamlit`** — built from a **real** `docker_files/dockerfile_streamlit` (currently
   empty), runs `streamlit run scripts/final/07_streamlit_app/app.py`, talks to `ollama`
   over the network, port-mapped to the host for browser access.

GPU sharing: Ollama (LLM inference) and the embeddings service (Qwen3-Embedding-0.6B)
both need the GPU but are invoked at different pipeline stages, never concurrently in
the current design (`run_pipeline.py` already loads/unloads them sequentially per
Section 5.7 of the thesis). Compose can give both containers GPU access; they just need
to not be hammered concurrently, which matches existing behavior.

---

## Concrete steps

### Step 1 — Make the Ollama host configurable (blocking, do first)
- `scripts/final/run_pipeline.py:137,178` (`from ollama import chat`, then the `chat(...)`
  call) — the `ollama` Python package respects the `OLLAMA_HOST` env var; verify this
  version does (`pip show ollama`, check `ollama/_client.py` for how it reads host
  config), and if not, switch to constructing an explicit `ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))` instead of the bare `chat()` helper.
- `scripts/final/07_streamlit_app/app.py:452` — replace the hardcoded
  `"http://localhost:11434/v1"` with `os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")`.
- Add `OLLAMA_HOST=http://ollama:11434` / `OLLAMA_BASE_URL=http://ollama:11434/v1` as
  compose-level env vars for the `pipeline` and `streamlit` services later.

### Step 2 — Write the embeddings Dockerfile (`docker_files/dockerfile_preprocessing`)
- Base: `rocm/pytorch:latest` (matches what's proven working today) or a slimmer ROCm
  base if build time matters — confirm with whoever owns the GPU target hardware.
  Note: existing `run_rocm_container.sh` uses several ROCm/WSL2-specific mounts
  (`/usr/lib/wsl/lib/libdxcore.so`, `/opt/rocm/lib/librocdxg.so`, `--device=/dev/dxg`,
  `HSA_ENABLE_DXG_DETECTION=1`) — these are **WSL2-on-Windows-specific** GPU passthrough
  hacks. If the target deployment is Linux-native or a different GPU vendor, these don't
  apply and need to be re-derived for that environment instead of copy-pasted.
- `COPY scripts/requirements-rocm.txt .` then `RUN pip install --no-deps -r requirements-rocm.txt`
  at build time (not container-start time as today).
- `COPY scripts/embeddings.py .` (or bind-mount it — see open question on bind-mount vs.
  bake-in below).
- Entrypoint: the container needs to stay alive for repeated `docker exec` calls per
  document (current design) OR be refactored to run as a long-lived HTTP service the
  `pipeline` container calls instead (recommended — see Step 3).

### Step 3 — Decide how `pipeline` talks to `embeddings` (design decision, not optional)
Current code (`source_retrieval_branches.py:1448-1460`) calls
`docker exec docplag-rocm python scripts/embeddings.py --doc_id ... --dataset ...` from
the **host**. This pattern does not transfer cleanly into compose, because:
- `docker exec` from inside one container into a sibling container requires mounting the
  host's Docker socket into the `pipeline` container (`/var/run/docker.sock`), which
  works but is a known security/portability wart, and the target container name has to
  match whatever compose generates (or be pinned via `container_name:` in the compose
  file).
- The alternative — and the recommended path — is to turn `scripts/embeddings.py`'s
  per-document logic into a tiny HTTP endpoint (FastAPI/Flask, one `/embed?doc_id=...&dataset=...`
  route) inside the `embeddings` container, and change `embedding_run()` in
  `source_retrieval_branches.py:1448-1460` from a `subprocess.run(["docker", "exec", ...])`
  call to a `requests.post("http://embeddings:8000/embed", json={...})` call. This is a
  real code change to `embedding_run()`, not just a packaging change — flag it as such
  when scoping effort.

Pick one of these two approaches explicitly before writing the compose file; don't leave
it ambiguous, since it changes what the embeddings Dockerfile's `ENTRYPOINT`/`CMD` needs
to be.

### Step 4 — Write the Streamlit Dockerfile (`docker_files/dockerfile_streamlit`)
- Base: `python:3.12-slim` is fine — the Streamlit app does not need GPU access itself
  (it delegates to `ollama` and `embeddings` over the network).
- `COPY requirements.txt .` then `pip install -r requirements.txt`.
- `EXPOSE 8501`, `CMD ["streamlit", "run", "scripts/final/07_streamlit_app/app.py", "--server.address=0.0.0.0"]`.
- App sets its working directory to `scripts/final/` at startup (per
  `generate_thesis_chapter.py` Section 5.1) — confirm this still resolves correctly when
  the container's filesystem layout differs from a bare bind-mount of the whole repo.

### Step 5 — Write `docker-compose.yml`
- Four services as above (`ollama`, `embeddings`, `pipeline`, `streamlit`), one network.
- Named volume for Ollama model storage (`ollama_models:/root/.ollama`), so `gemma4:26b`
  persists across container restarts instead of re-pulling ~17GB every time.
- Bind-mount or volume for `datasets/`, `artifacts/`, and `scripts/final/pipeline_results*/`
  so results survive container teardown and match the host-relative-path assumptions
  baked into `run_pipeline.py`'s `SCRIPT_DIR`-relative path resolution
  (`scripts/final/run_pipeline.py:39-72`).
- GPU passthrough block for `ollama` and `embeddings` services — syntax depends on
  whether the target is NVIDIA (`deploy.resources.reservations.devices` with
  `driver: nvidia`) or AMD ROCm (device cgroup rules, `--device=/dev/kfd`,
  `--device=/dev/dri`, group_add `video`) — **the current dev machine is AMD ROCm on
  WSL2** (per Table 5.6 in the thesis), so default to the ROCm/WSL2 device mounts already
  proven in `run_rocm_container.sh`, and treat NVIDIA support as a documented but
  untested alternative.
- `depends_on` ordering: `pipeline`/`streamlit` should wait for `ollama` to be healthy
  (Ollama takes a few seconds to start and longer to load a model on first request) —
  use a healthcheck against `http://ollama:11434/` rather than just container-start
  ordering.

### Step 6 — Model provisioning
- `gemma4:26b` and the embedding model (`artifacts/models/Qwen3-Embedding-0.6B`) need to
  exist inside their respective containers/volumes before first real use.
- For Ollama: either bake a `RUN ollama pull gemma4:26b` into a custom Ollama image (slow
  build, fast first run) or pull it at container startup via an entrypoint script
  checking `ollama list` first (fast build, slow first run). Decide based on whether this
  is meant to be a "clone and run" deployment artifact or a dev convenience.
- For the embedding model: `artifacts/models/Qwen3-Embedding-0.6B` already exists on the
  host (checked: `artifacts/models/` contains `Qwen3-Embedding-0.6B/` and
  `Qwen3-Embedding-4B/`) — simplest path is bind-mounting `artifacts/models/` read-only
  into the `embeddings` container rather than re-downloading inside the image.

### Step 7 — Test plan (don't skip — this is the part most likely to break)
1. `docker compose up ollama embeddings` alone first; confirm `docker exec`-equivalent
   embedding call (or the new HTTP route from Step 3) returns a real candidate parquet
   for one known document ID, compared byte-for-byte or score-for-score against the
   current host-native run's output for the same doc.
2. Confirm Ollama in a container can actually see the GPU
   (`docker exec <ollama-container> ollama run gemma4:26b "hi"` and check it's not
   falling back to CPU-only inference — compare latency against the native-host timing
   already measured, ~5-10s per confirmation call per Section 5.5).
3. Run `pipeline` against a **single document** (`--doc-id`) end-to-end inside compose
   before attempting a full batch — this mirrors how the host-native version was
   actually debugged (see Section 5.10 of the thesis for the bugs that surfaced doing
   exactly this on the custom dataset).
4. Only after 1-3 pass, run the full custom-dataset batch (10 docs) inside compose and
   diff results against the host-native run already on record (`pipeline_results_custom/`)
   to confirm containerizing didn't change pipeline behavior.

---

## Open questions to resolve before/during implementation

1. **Target deployment platform** — is this shipping for the same AMD ROCm/WSL2 dev box,
   or for a different machine (Linux-native, NVIDIA, cloud GPU)? This changes the GPU
   passthrough section of the compose file substantially (Step 5).
2. **`docker exec` vs. HTTP for the embeddings service** (Step 3) — needs an explicit
   decision, not a default.
3. **Model provisioning strategy** — bake into image vs. pull on first run (Step 6).
4. **Where does Ollama's model storage live in CI/sharing context** — if this is meant to
   be handed to someone else to run, a 17GB model download is part of the "first run"
   experience; document that expectation rather than assuming it away.

---

## Why this belongs in the thesis (Chapter 5/6 framing, for after this is built and tested)

Once this is implemented and verified end-to-end (Step 7 test plan passes), it becomes:
- **Chapter 5**: a new "Containerized Deployment" section describing the four-service
  architecture, replacing the current ad hoc native-Ollama + manual-docker-exec setup
  documented in Sections 5.4/5.7/5.8.5.
- **Chapter 6**: a line in the contributions/future-work framing noting that the pipeline
  moved from a single-machine, manually-orchestrated research prototype to a
  reproducible, `docker compose up`-deployable artifact — relevant to the "production
  deployment" caveat already present in Section 6.7 ("not suitable for production
  deployment, but acceptable for a research prototype").

**Do not write these thesis sections until the containerized setup is actually built and
the Step 7 test plan has passed** — Chapter 5/6 should describe what was built and
verified, not what was planned, consistent with how the rest of these chapters are
written (see the custom-dataset section, which reports real per-document results, not
intentions).
