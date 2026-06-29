"""
Plagiarism Source Detector — Streamlit app
Run from project root:  streamlit run scripts/final/07_streamlit_app/app.py
"""

import sys
import io
import contextlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve to absolute so paths are stable regardless of the process CWD.
APP_DIR      = Path(__file__).resolve().parent
SCRIPTS_FINAL = APP_DIR.parent                              # scripts/final/
PROJECT_ROOT  = SCRIPTS_FINAL.parent.parent                 # Document_Plagiarism/
TOP20_PATH    = SCRIPTS_FINAL / "top20_df.parquet"

DATASET_CONFIGS = {
    "pan2011": {
        "label": "PAN2011",
        "processed_dir": PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300",
        "gt_path": PROJECT_ROOT / "datasets" / "processed" / "PAN2011_ground_truth" / "pan2011_plagiarism_spans.parquet",
        "default_doc": "part1__suspicious-document00007.txt",
        "evaluated_only": True,
    },
    "custom": {
        "label": "Custom dataset",
        "processed_dir": PROJECT_ROOT / "datasets" / "processed" / "custom_300",
        "gt_path": PROJECT_ROOT / "datasets" / "processed" / "custom_ground_truth" / "custom_plagiarism_spans.parquet",
        "default_doc": "suspicious-document00007.txt",
        "evaluated_only": False,
    },
}

PROCESSED_DIR = DATASET_CONFIGS["pan2011"]["processed_dir"]
GT_PATH       = DATASET_CONFIGS["pan2011"]["gt_path"]

# NOTE: do NOT os.chdir() here. source_retrieval_branches.py anchors all of its
# paths to Path(__file__).resolve().parents[3], so it does not depend on the CWD.
# Changing the CWD breaks Streamlit's ability to re-read this script on widget
# reruns (it launches the script by its relative path), causing a doubled-path
# FileNotFoundError. Keep the process CWD as Streamlit set it.
sys.path.insert(0, str(SCRIPTS_FINAL / "04_source_retrieval"))
sys.path.insert(0, str(SCRIPTS_FINAL / "05_text_alignment"))

# ── Helpers ───────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def capture_stdout():
    """Redirect stdout to a StringIO buffer so pipeline logs can be shown in the UI."""
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


class _LiveWriter(io.TextIOBase):
    """
    stdout/stderr proxy that mirrors everything into a Streamlit placeholder
    *as it is written*, so long-running lookups (and their tqdm bars) show
    live progress instead of appearing frozen until the stage finishes.

    tqdm redraws its bar with carriage returns (\\r); we treat \\r like a
    line reset so only the latest bar state is shown rather than hundreds of
    stale frames.
    """
    def __init__(self, placeholder, max_lines: int = 18):
        self._ph = placeholder
        self._lines = [""]          # logical lines; _lines[-1] is "current"
        self._max_lines = max_lines
        self._full = io.StringIO()  # complete transcript for the final expander

    def write(self, s: str) -> int:
        self._full.write(s)
        for ch in s:
            if ch == "\r":
                self._lines[-1] = ""          # carriage return → rewrite current line
            elif ch == "\n":
                self._lines.append("")
            else:
                self._lines[-1] += ch
        # Render only the tail so the UI stays light
        tail = [ln for ln in self._lines if ln != ""][-self._max_lines:]
        self._ph.code("\n".join(tail) or "…")
        return len(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._full.getvalue()


@contextlib.contextmanager
def live_log(placeholder):
    """Stream stdout AND stderr into a Streamlit placeholder in real time."""
    old_out, old_err = sys.stdout, sys.stderr
    writer = _LiveWriter(placeholder)
    sys.stdout = sys.stderr = writer
    try:
        yield writer
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def abort_if_stopped() -> None:
    """If the user pressed Stop, surface a notice and halt this script run.

    Cooperative cancellation: checked at stage boundaries and inside the LLM
    loop. It cannot kill an in-flight native call (FAISS/sklearn lookup), but
    prevents the pipeline from proceeding to the next expensive stage.
    """
    if st.session_state.get("stop_requested"):
        st.session_state.stop_requested = False
        st.warning("⏹ Run stopped by user.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_suspicious_doc_ids(processed_dir: str) -> list[str]:
    """Load all unique suspicious document IDs from the processed chunk Parquet."""
    df = pd.read_parquet(Path(processed_dir) / "suspicious_chunks.parquet", columns=["doc_id"])
    return sorted(df["doc_id"].unique().tolist())


@st.cache_data(show_spinner=False)
def load_doc_texts(processed_dir: str) -> pd.DataFrame:
    """Load suspicious document texts + char/word counts, indexed by doc_id."""
    df = pd.read_parquet(
        Path(processed_dir) / "suspicious_documents.parquet",
        columns=["doc_id", "clean_text", "clean_char_count", "clean_word_count"],
    )
    return df.set_index("doc_id")


@st.cache_data(show_spinner=False)
def load_gt_spans(gt_path: str) -> pd.DataFrame:
    """Load PAN 2011 ground-truth plagiarism spans (one row per plagiarized passage)."""
    wanted_cols = [
        "suspicious_doc_id", "suspicious_offset", "suspicious_length", "suspicious_end",
        "source_doc_id", "source_reference", "source_offset", "source_length",
        "source_end", "plagiarism_type", "obfuscation",
    ]
    available_cols = set(pd.read_parquet(Path(gt_path), engine="pyarrow").columns)
    df = pd.read_parquet(
        Path(gt_path),
        columns=[col for col in wanted_cols if col in available_cols],
    )
    if "source_reference" not in df.columns and "source_doc_id" in df.columns:
        df["source_reference"] = df["source_doc_id"]
    return df


def get_doc_gt(doc_id: str, gt_path: Path) -> pd.DataFrame:
    """Return GT spans for one suspicious doc, sorted by suspicious offset."""
    gt = load_gt_spans(str(gt_path))
    return (gt[gt["suspicious_doc_id"] == doc_id]
            .sort_values("suspicious_offset")
            .reset_index(drop=True))


def highlight_plagiarized_html(text: str, spans: pd.DataFrame, max_chars: int = 8000) -> str:
    """
    Render document text as HTML with GT plagiarized regions highlighted.
    Truncates to max_chars for performance; truncation never splits a highlight.
    """
    import html as _html

    if spans.empty:
        return f"<div style='white-space:pre-wrap;font-family:monospace;font-size:0.8rem'>{_html.escape(text[:max_chars])}</div>"

    # Build a sorted, merged list of (start, end) highlight regions
    regions = sorted(
        (int(r.suspicious_offset), int(r.suspicious_offset) + int(r.suspicious_length))
        for r in spans.itertuples()
    )
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    out, cursor = [], 0
    for s, e in merged:
        if cursor >= max_chars:
            break
        # plain text before the highlight
        out.append(_html.escape(text[cursor:min(s, max_chars)]))
        if s >= max_chars:
            break
        seg = _html.escape(text[s:min(e, max_chars)])
        out.append(f"<mark style='background:#ffd54f'>{seg}</mark>")
        cursor = e
    if cursor < max_chars:
        out.append(_html.escape(text[cursor:max_chars]))

    truncated = "<br><i>… (truncated for preview)</i>" if len(text) > max_chars else ""
    return (f"<div style='white-space:pre-wrap;font-family:monospace;font-size:0.8rem;"
            f"max-height:420px;overflow-y:auto;border:1px solid #444;padding:8px;border-radius:6px'>"
            f"{''.join(out)}{truncated}</div>")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Source Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
# Persist results across Streamlit reruns triggered by widget interactions
_defaults = {
    "pipeline_done":  False,
    "last_doc_id":    None,
    "top20_df":       None,
    "llm_scores_df":  None,
    "top5_df":        None,
    "stop_requested": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Plagiarism Detector")
    st.markdown("---")

    selected_dataset = st.selectbox(
        "Dataset",
        options=list(DATASET_CONFIGS.keys()),
        format_func=lambda k: DATASET_CONFIGS[k]["label"],
        index=0,
    )
    dataset_cfg = DATASET_CONFIGS[selected_dataset]
    PROCESSED_DIR = dataset_cfg["processed_dir"]
    GT_PATH = dataset_cfg["gt_path"]

    all_doc_ids = load_suspicious_doc_ids(str(PROCESSED_DIR))
    gt_doc_ids  = set(load_gt_spans(str(GT_PATH))["suspicious_doc_id"].unique())

    # Evaluated subset = the part1 docs the pipeline was benchmarked on (the ones
    # with PAN ground truth). Default to these; offer the full corpus on request.
    scope_options = (
        ["Evaluated subset (part1 w/ ground truth)", "Full corpus (11,093 docs)"]
        if selected_dataset == "pan2011"
        else ["Custom dataset"]
    )
    scope = st.radio("Document set", scope_options, index=0)
    if selected_dataset == "pan2011" and scope.startswith("Evaluated"):
        doc_ids = [d for d in all_doc_ids if d.startswith("part1__") and d in gt_doc_ids]
    else:
        doc_ids = all_doc_ids

    # Optional text filter to find a doc quickly within the chosen scope
    query = st.text_input("Filter by id (substring)", value="", placeholder="e.g. 00007")
    if query:
        doc_ids = [d for d in doc_ids if query in d]
    if not doc_ids:
        st.warning("No documents match the filter.")
        st.stop()

    default_doc = dataset_cfg["default_doc"]
    default_idx = doc_ids.index(default_doc) if default_doc in doc_ids else 0
    selected_doc = st.selectbox(f"Suspicious document ({len(doc_ids)} available)",
                                options=doc_ids, index=default_idx)

    st.markdown("**Run Control**")
    run_mode = st.radio(
        "Run mode",
        ["Selected document", "Batch"],
        index=0,
        help="Both modes launch run_pipeline.py from this UI with the flags selected below.",
    )
    fresh_run = st.checkbox(
        "Fresh run",
        value=False,
        help="Pass --fresh to ignore cached per-doc parquet results.",
    )
    batch_docs = 1
    if run_mode == "Batch":
        batch_docs = st.number_input(
            "Batch documents",
            min_value=1,
            max_value=len(doc_ids),
            value=min(10, len(doc_ids)),
            step=1,
        )

    st.markdown("**Source Retrieval**")
    top_n = st.number_input("Fused top-N candidates", min_value=5, max_value=50, value=20, step=5)
    run_embeddings = st.checkbox("Re-run GPU embeddings", value=False,
                                 help="Leave off to reuse the existing embedding parquet.")
    embeddings_backend_label = st.selectbox(
        "Embeddings backend",
        ["ROCm service", "Local Python", "Legacy docker exec"],
        index=0,
        disabled=not run_embeddings,
    )
    embeddings_backend = {
        "ROCm service": "http",
        "Local Python": "local",
        "Legacy docker exec": "docker-exec",
    }[embeddings_backend_label]
    skip_tfidf = st.checkbox(
        "Skip TF-IDF",
        value=(run_mode == "Batch"),
        help="Recommended for quick container smoke tests; TF-IDF can be memory-heavy.",
    )
    skip_esa = st.checkbox("Skip ESA", value=False, help="Use if RAM is tight.")
    min_top1_score = st.number_input("Gate 1 min top-1 score", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    relative_gap = st.number_input("Gate 2 relative gap", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
    retrieval_recall_k = st.number_input("Retrieval Recall@K", min_value=1, max_value=50, value=20, step=1)
    cross_encoder = st.checkbox("Cross-encoder rerank", value=False)

    st.markdown("**LLM Re-ranking**")
    skip_llm = st.checkbox(
        "Skip LLM",
        value=(run_mode == "Batch"),
        help="Useful for retrieval-only smoke tests.",
    )
    ollama_model    = st.text_input("Ollama model", value="gemma4:26b")
    llm_threshold = st.number_input("LLM threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    top_pairs_per_doc = st.slider("Chunk pairs per source doc", min_value=1, max_value=25, value=3,
                                   help="How many top embedding-similarity pairs to show the LLM per candidate doc.")
    debug_llm = st.checkbox("Dump LLM debug prompts", value=False)
    perplexity_filter = False
    perplexity_threshold = 250.0
    soft_gate1 = st.checkbox("Soft Gate 1", value=False)
    soft_gate1_min_branch = st.number_input("Soft Gate 1 min branch", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

    st.markdown("---")
    run_btn  = st.button("▶  Run Pipeline", type="primary", width='stretch')
    # Stop sets a flag the run loop checks between stages / LLM candidates.
    # It cannot interrupt a single in-flight lookup, but aborts cleanly at the
    # next stage boundary so you don't have to wait for the whole pipeline.
    stop_btn = st.button("⏹  Stop", width='stretch',
                         help="Abort the current run at the next stage boundary.")
    if stop_btn:
        st.session_state.stop_requested = True

    if st.session_state.pipeline_done:
        st.success(f"Done: {st.session_state.last_doc_id}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📄 Plagiarism Source Detector")


def render_corpus_preview(doc_id: str) -> None:
    """Show document stats, plagiarized/clean status, GT span table, and highlighted text."""
    texts = load_doc_texts(str(PROCESSED_DIR))
    gt    = get_doc_gt(doc_id, GT_PATH)

    if doc_id not in texts.index:
        st.warning(f"No cached text found for `{doc_id}`.")
        return

    row        = texts.loc[doc_id]
    clean_text = row["clean_text"] or ""
    is_plag    = not gt.empty

    st.markdown(f"### 📖 Corpus Preview — `{doc_id}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "🔴 plagiarized" if is_plag else "🟢 Clean")
    c2.metric("GT spans", len(gt))
    c3.metric("Characters", f"{int(row['clean_char_count']):,}")
    c4.metric("Words", f"{int(row['clean_word_count']):,}")

    if is_plag:
        n_sources = gt["source_doc_id"].nunique()
        obf_counts = gt["obfuscation"].value_counts().to_dict()
        obf_str = ", ".join(f"{k}×{v}" for k, v in obf_counts.items())
        st.caption(f"plagiarized from **{n_sources}** source doc(s) · obfuscation: {obf_str}")

        with st.expander("📋 Ground-truth spans", expanded=False):
            display_cols = [
                "suspicious_offset", "suspicious_length", "source_reference",
                "source_offset", "source_length", "obfuscation", "plagiarism_type",
            ]
            disp = gt[[col for col in display_cols if col in gt.columns]].copy()
            st.dataframe(disp, width='stretch', hide_index=True)

    with st.expander("📄 Document text (plagiarized regions highlighted)", expanded=True):
        st.markdown(highlight_plagiarized_html(clean_text, gt), unsafe_allow_html=True)
        if is_plag:
            st.caption("🟡 Highlighted = ground-truth plagiarized passage")


if not run_btn and not st.session_state.pipeline_done:
    render_corpus_preview(selected_doc)
    st.markdown("---")
    st.info(
        "Review the document above, then click **▶ Run Pipeline** in the sidebar to detect its sources.\n\n"
        "The pipeline runs: **TF-IDF → ESA → LSA → Embeddings → Fusion → LLM re-ranking**."
    )
    if TOP20_PATH.exists():
        with st.expander("📂 Cached top20_df.parquet (from previous run)", expanded=False):
            st.dataframe(pd.read_parquet(TOP20_PATH), width='stretch')
    st.stop()

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    # Reset state for a fresh run (including any stale stop request)
    for k in ("pipeline_done", "top20_df", "llm_scores_df", "top5_df"):
        st.session_state[k] = None if k != "pipeline_done" else False
    st.session_state.stop_requested = False

    import subprocess

    cmd = [
        sys.executable,
        str(SCRIPTS_FINAL / "run_pipeline.py"),
        "--dataset", selected_dataset,
        "--top-n", str(int(top_n)),
        "--retrieval-recall-k", str(int(retrieval_recall_k)),
        "--llm-threshold", str(float(llm_threshold)),
        "--min-top1-score", str(float(min_top1_score)),
        "--relative-gap", str(float(relative_gap)),
        "--top-pairs", str(int(top_pairs_per_doc)),
        "--ollama-model", ollama_model,
        "--embeddings-backend", embeddings_backend,
    ]
    if run_mode == "Selected document":
        cmd.extend(["--doc-id", selected_doc])
    else:
        cmd.extend(["--docs", str(int(batch_docs))])
    if fresh_run:
        cmd.append("--fresh")
    if run_embeddings:
        cmd.append("--run-embeddings")
    if skip_tfidf:
        cmd.append("--skip-tfidf")
    if skip_esa:
        cmd.append("--skip-esa")
    if skip_llm:
        cmd.append("--skip-llm")
    if debug_llm:
        cmd.append("--debug-llm")
    if perplexity_filter:
        cmd.append("--perplexity-filter")
        cmd.extend(["--perplexity-threshold", str(float(perplexity_threshold))])
    if soft_gate1:
        cmd.append("--soft-gate1")
        cmd.extend(["--soft-gate1-min-branch", str(float(soft_gate1_min_branch))])
    if cross_encoder:
        cmd.append("--cross-encoder")

    st.subheader("Pipeline Runner")
    st.code(" ".join(cmd))
    log_ph = st.empty()
    lines: list[str] = []
    child_env = os.environ.copy()
    child_env["EMBEDDINGS_BACKEND"] = embeddings_backend
    if child_env.get("RUNNING_IN_DOCKER") != "1":
        child_env.pop("OLLAMA_HOST", None)

    if run_embeddings and embeddings_backend == "http":
        import urllib.request
        embeddings_url = child_env.get("EMBEDDINGS_URL", "http://localhost:8000").rstrip("/")
        try:
            urllib.request.urlopen(f"{embeddings_url}/health", timeout=2).read()
        except Exception as exc:
            st.error(
                f"Embedding service is not reachable at `{embeddings_url}`. "
                "Start `docker_files/run_rocm_embedding_service.sh` first, or turn off live embeddings."
            )
            st.exception(exc)
            st.stop()

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_ph.code("\n".join(lines[-30:]) or "(no output yet)")

    return_code = proc.wait()
    if return_code != 0:
        st.error(f"Pipeline runner exited with code {return_code}.")
        st.stop()

    st.success("Pipeline runner finished.")
    st.stop()

    import source_retrieval_branches as srb

    srb.SUSPICIOUS_DOC_ID = selected_doc

    # Keep the corpus preview visible at the top while the pipeline runs
    with st.expander(f"📖 Running against `{selected_doc}` — show document preview", expanded=False):
        render_corpus_preview(selected_doc)

    # ── Stage 1: Source Retrieval ─────────────────────────────────────────────
    st.subheader("Stage 1 — Source Retrieval")

    stage_error = None
    tf_df = esa_df = lsa_df = emb_df = None

    # TF-IDF branch — char n-gram hashed vectors, best for near-copy plagiarism
    with st.status("TF-IDF lookup …", expanded=True) as s:
        log_ph = st.empty()
        try:
            srb.search_artifact("tf-idf")
            with live_log(log_ph) as buf:
                tf_df = srb.tf_idf_lookup()
            log_ph.code(buf.getvalue() or "(no output)")
            s.update(label=f"✅ TF-IDF — {len(tf_df)} source docs ranked", state="complete", expanded=False)
        except Exception as exc:
            s.update(label=f"❌ TF-IDF failed", state="error")
            st.exception(exc)
            stage_error = exc

    abort_if_stopped()

    # ESA branch — explicit semantic analysis over Wikipedia concept space
    if not stage_error:
        with st.status("ESA lookup …", expanded=True) as s:
            log_ph = st.empty()
            try:
                srb.search_artifact("esa")
                with live_log(log_ph) as buf:
                    esa_df = srb.esa_lookup()
                log_ph.code(buf.getvalue() or "(no output)")
                s.update(label=f"✅ ESA — {len(esa_df)} source docs ranked", state="complete", expanded=False)
            except Exception as exc:
                s.update(label=f"❌ ESA failed", state="error")
                st.exception(exc)
                stage_error = exc

    abort_if_stopped()

    # LSA branch — latent semantic analysis via TruncatedSVD
    if not stage_error:
        with st.status("LSA lookup …", expanded=True) as s:
            log_ph = st.empty()
            try:
                srb.search_artifact("lsa")
                with live_log(log_ph) as buf:
                    lsa_df = srb.lsa_lookup()
                log_ph.code(buf.getvalue() or "(no output)")
                s.update(label=f"✅ LSA — {len(lsa_df)} source docs ranked", state="complete", expanded=False)
            except Exception as exc:
                s.update(label=f"❌ LSA failed", state="error")
                st.exception(exc)
                stage_error = exc

    abort_if_stopped()

    # Embeddings branch — dense neural vectors via Qwen3-Embedding-0.6B + FAISS
    if not stage_error:
        with st.status("Embeddings …", expanded=True) as s:
            try:
                if run_embeddings:
                    # Re-run the GPU embedding lookup through the embedding service.
                    proc = srb.embedding_run(selected_doc)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr)
                # Load results whether freshly computed or cached from a prior run
                emb_path = PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"
                emb_df = pd.read_parquet(emb_path)
                label = f"✅ Embeddings — {len(emb_df)} source docs ranked"
                label += " (re-run)" if run_embeddings else " (cached)"
                s.update(label=label, state="complete", expanded=False)
            except Exception as exc:
                s.update(label="❌ Embeddings failed", state="error")
                st.exception(exc)
                stage_error = exc

    abort_if_stopped()

    # Fusion — weighted combination of all four branch scores
    if not stage_error:
        with st.status("Fusing branch scores …", expanded=True) as s:
            try:
                with capture_stdout() as buf:
                    top20_df = srb.mean_doc_score_aggreg(
                        top_tf_idf=tf_df,
                        top_esa=esa_df,
                        top_lsa=lsa_df,
                        top_emb=emb_df,
                        final_top_n=int(top_n),
                    )
                # Persist the fused result so it survives Streamlit reruns
                top20_df.to_parquet(TOP20_PATH, index=False)
                st.session_state.top20_df = top20_df
                s.update(label=f"✅ Fusion — top {len(top20_df)} candidates", state="complete", expanded=False)
            except Exception as exc:
                s.update(label="❌ Fusion failed", state="error")
                st.exception(exc)
                stage_error = exc

    if stage_error:
        st.stop()

    abort_if_stopped()

    # ── Stage 2: LLM Re-ranking ───────────────────────────────────────────────
    st.subheader("Stage 2 — LLM Re-ranking")

    import instructor
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class SourceDocScore(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        is_likely_source: bool
        reasoning: str

    # Ollama serves a local LLM; instructor forces structured JSON output
    ollama_client = instructor.from_openai(
        OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"),
        mode=instructor.Mode.JSON,
    )

    # Build chunk pairs: for each candidate source doc, select the top-K
    # suspicious ↔ source chunk pairs by embedding similarity score
    with st.status("Building text pairs …", expanded=True) as s:
        try:
            top_source_ids = set(top20_df["source_doc_id"].tolist())

            cand_df   = pd.read_parquet(PROCESSED_DIR / "embedding_candidates_suspicious.parquet")
            src_chunks = pd.read_parquet(PROCESSED_DIR / "source_chunks.parquet")
            susp_emb   = pd.read_parquet(PROCESSED_DIR / "suspicious_chunks_embeddings.parquet")

            # Keep only candidates for the selected suspicious doc and top source docs
            cand_filtered = cand_df[
                (cand_df["suspicious_doc_id"] == selected_doc) &
                (cand_df["source_doc_id"].isin(top_source_ids))
            ].copy()

            # Join the actual text so the LLM can read it
            susp_text = (
                susp_emb[susp_emb["doc_id"] == selected_doc]
                [["chunk_id", "embedding_text"]]
                .rename(columns={"chunk_id": "suspicious_chunk_id", "embedding_text": "suspicious_text"})
            )
            src_text = (
                src_chunks[src_chunks["doc_id"].isin(top_source_ids)]
                [["chunk_id", "chunk_text"]]
                .rename(columns={"chunk_id": "source_chunk_id", "chunk_text": "source_text"})
            )

            pairs_df = (
                cand_filtered
                .merge(susp_text, on="suspicious_chunk_id", how="inner")
                .merge(src_text,  on="source_chunk_id",     how="inner")
            )
            # Take the top-K pairs per source doc by embedding similarity
            top_pairs = (
                pairs_df
                .sort_values("embedding_score", ascending=False)
                .groupby("source_doc_id")
                .head(int(top_pairs_per_doc))
                .reset_index(drop=True)
            )
            n_docs = top_pairs["source_doc_id"].nunique()
            s.update(
                label=f"✅ {len(top_pairs)} pairs across {n_docs} source docs",
                state="complete", expanded=False,
            )
        except Exception as exc:
            s.update(label="❌ Pair building failed", state="error")
            st.exception(exc)
            st.stop()

    # LLM scoring — live progress bar shown while each candidate is scored
    abort_if_stopped()
    st.markdown("**Scoring each candidate with the LLM…**")
    st.caption("ℹ️ Stop aborts at stage boundaries. To interrupt mid-LLM-scoring, "
               "stop the `streamlit run` process (Ctrl+C in the terminal).")
    source_groups   = list(top_pairs.groupby("source_doc_id"))
    progress_bar    = st.progress(0, text="Starting LLM scoring…")
    status_text     = st.empty()
    llm_results     = []

    for i, (source_doc_id, group) in enumerate(source_groups):
        short_id = source_doc_id.split("__")[-1]
        progress_bar.progress((i) / len(source_groups), text=f"Scoring {i+1}/{len(source_groups)}: {short_id}")
        status_text.info(f"🤖 `{source_doc_id}`")

        pairs = group[["suspicious_text", "source_text", "embedding_score"]].to_dict("records")
        # Truncate each passage to 600 chars to keep the prompt within model context limits
        pairs_text = "\n\n".join(
            f"[Pair {j+1}]\nSUSPICIOUS: {p['suspicious_text'][:600]}\nSOURCE: {p['source_text'][:600]}"
            for j, p in enumerate(pairs)
        )
        prompt = (
            f"You are a plagiarism detection expert.\n"
            f"Below are {len(pairs)} text pair(s) — chunks from a SUSPICIOUS document paired with "
            f"chunks from a CANDIDATE SOURCE document.\n\n"
            f"{pairs_text}\n\n"
            f"Decide whether the suspicious text appears copied or paraphrased from this source. "
            f"Score likelihood 0.0 (definitely not) to 1.0 (definitely yes). "
            f"Return JSON with: score (float 0-1), is_likely_source (bool), reasoning (string)."
        )

        try:
            res = ollama_client.chat.completions.create(
                model=ollama_model,
                response_model=SourceDocScore,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # low temperature for reproducible judgements
            )
            llm_results.append({
                "source_doc_id":        source_doc_id,
                "llm_score":            res.score,
                "llm_is_likely_source": res.is_likely_source,
                "llm_reasoning":        res.reasoning,
            })
        except Exception as exc:
            # On error, record a zero score so this doc is ranked last
            llm_results.append({
                "source_doc_id":        source_doc_id,
                "llm_score":            0.0,
                "llm_is_likely_source": False,
                "llm_reasoning":        f"Error: {exc}",
            })

    progress_bar.progress(1.0, text="LLM scoring complete!")
    status_text.empty()

    llm_scores_df = (
        pd.DataFrame(llm_results)
        .sort_values("llm_score", ascending=False)
        .reset_index(drop=True)
    )
    st.session_state.llm_scores_df = llm_scores_df
    st.session_state.top5_df       = llm_scores_df.head(5).copy()
    st.session_state.pipeline_done = True
    st.session_state.last_doc_id   = selected_doc

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.pipeline_done:
    top5       = st.session_state.top5_df
    scores_df  = st.session_state.llm_scores_df
    top20_df   = st.session_state.top20_df or (pd.read_parquet(TOP20_PATH) if TOP20_PATH.exists() else None)

    st.subheader("Results")
    st.markdown(f"#### Top 5 Most Likely Sources for `{st.session_state.last_doc_id}`")

    # Show the top-5 as metric cards with score and likely/unlikely label
    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        score = row["llm_score"]
        delta_color = "normal"
        with cols[i]:
            st.metric(
                label=f"#{i+1}",
                value=f"{score:.2f}",
                delta="likely ✓" if row["llm_is_likely_source"] else "unlikely",
                delta_color="normal" if row["llm_is_likely_source"] else "inverse",
            )
            st.caption(row["source_doc_id"].split("__")[-1])

    st.markdown("---")

    # Expandable cards with full reasoning for each top-5 source
    for i, (_, row) in enumerate(top5.iterrows()):
        with st.expander(
            f"#{i+1}  `{row['source_doc_id']}`  —  score {row['llm_score']:.3f}",
            expanded=(i == 0),
        ):
            col1, col2 = st.columns([1, 4])
            col1.metric("Score", f"{row['llm_score']:.3f}")
            col1.metric("Likely source", "Yes" if row["llm_is_likely_source"] else "No")
            col2.markdown(f"**Reasoning:**\n\n{row['llm_reasoning']}")

    st.markdown("---")
    st.markdown("#### All LLM Scores")
    st.dataframe(scores_df, width='stretch', hide_index=True)

    if top20_df is not None:
        st.markdown("#### Source Retrieval Fusion (top 20 pre-LLM)")
        st.dataframe(top20_df, width='stretch', hide_index=True)
