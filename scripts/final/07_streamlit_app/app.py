"""
Plagiarism Source Detector — Streamlit app
Run from project root:  streamlit run scripts/final/07_streamlit_app/app.py
"""

import os
import sys
import io
import contextlib
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent
SCRIPTS_FINAL = APP_DIR.parent                              # scripts/final/
PROJECT_ROOT  = SCRIPTS_FINAL.parent.parent                 # Document_Plagiarism/
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300"
TOP20_PATH    = SCRIPTS_FINAL / "top20_df.parquet"

# source_retrieval_branches.py uses Path("../datasets/…") relative to CWD,
# which resolves correctly only when CWD == scripts/final/
os.chdir(SCRIPTS_FINAL)
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


@st.cache_data(show_spinner=False)
def load_suspicious_doc_ids() -> list[str]:
    """Load all unique suspicious document IDs from the processed chunk Parquet."""
    df = pd.read_parquet(PROCESSED_DIR / "suspicious_chunks.parquet", columns=["doc_id"])
    return sorted(df["doc_id"].unique().tolist())


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Source Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
# Persist results across Streamlit reruns triggered by widget interactions
_defaults = {
    "pipeline_done": False,
    "last_doc_id":   None,
    "top20_df":      None,
    "llm_scores_df": None,
    "top5_df":       None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Plagiarism Detector")
    st.markdown("---")

    doc_ids = load_suspicious_doc_ids()
    default_idx = doc_ids.index("part1__suspicious-document00007.txt") if "part1__suspicious-document00007.txt" in doc_ids else 0
    selected_doc = st.selectbox("Suspicious document", options=doc_ids, index=default_idx)

    st.markdown("**Source Retrieval**")
    top_n = st.number_input("Fused top-N candidates", min_value=5, max_value=50, value=20, step=5)
    run_embeddings = st.checkbox("Re-run GPU embeddings (Docker)", value=False,
                                 help="Runs the Docker ROCm container for embedding lookup. Leave off to reuse the existing parquet.")

    st.markdown("**LLM Re-ranking**")
    ollama_model    = st.text_input("Ollama model", value="gemma4:26b")
    top_pairs_per_doc = st.slider("Chunk pairs per source doc", min_value=1, max_value=5, value=3,
                                   help="How many top embedding-similarity pairs to show the LLM per candidate doc.")

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

    if st.session_state.pipeline_done:
        st.success(f"Done: {st.session_state.last_doc_id}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📄 Plagiarism Source Detector")

if not run_btn and not st.session_state.pipeline_done:
    st.info(
        "Pick a suspicious document in the sidebar and click **Run Pipeline**.\n\n"
        "The pipeline runs: **TF-IDF → ESA → LSA → Embeddings → Fusion → LLM re-ranking**."
    )
    if TOP20_PATH.exists():
        with st.expander("📂 Cached top20_df.parquet (from previous run)", expanded=False):
            st.dataframe(pd.read_parquet(TOP20_PATH), use_container_width=True)
    st.stop()

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    # Reset state for a fresh run
    for k in ("pipeline_done", "top20_df", "llm_scores_df", "top5_df"):
        st.session_state[k] = None if k != "pipeline_done" else False

    import source_retrieval_branches as srb

    srb.SUSPICIOUS_DOC_ID = selected_doc

    # ── Stage 1: Source Retrieval ─────────────────────────────────────────────
    st.subheader("Stage 1 — Source Retrieval")

    stage_error = None
    tf_df = esa_df = lsa_df = emb_df = None

    # TF-IDF branch — char n-gram hashed vectors, best for near-copy plagiarism
    with st.status("TF-IDF lookup …", expanded=True) as s:
        try:
            srb.search_artifact("tf-idf")
            with capture_stdout() as buf:
                tf_df = srb.tf_idf_lookup()
            s.update(label=f"✅ TF-IDF — {len(tf_df)} source docs ranked", state="complete", expanded=False)
            with st.expander("TF-IDF log"):
                st.code(buf.getvalue() or "(no output)")
        except Exception as exc:
            s.update(label=f"❌ TF-IDF failed", state="error")
            st.exception(exc)
            stage_error = exc

    # ESA branch — explicit semantic analysis over Wikipedia concept space
    if not stage_error:
        with st.status("ESA lookup …", expanded=True) as s:
            try:
                srb.search_artifact("esa")
                with capture_stdout() as buf:
                    esa_df = srb.esa_lookup()
                s.update(label=f"✅ ESA — {len(esa_df)} source docs ranked", state="complete", expanded=False)
                with st.expander("ESA log"):
                    st.code(buf.getvalue() or "(no output)")
            except Exception as exc:
                s.update(label=f"❌ ESA failed", state="error")
                st.exception(exc)
                stage_error = exc

    # LSA branch — latent semantic analysis via TruncatedSVD
    if not stage_error:
        with st.status("LSA lookup …", expanded=True) as s:
            try:
                srb.search_artifact("lsa")
                with capture_stdout() as buf:
                    lsa_df = srb.lsa_lookup()
                s.update(label=f"✅ LSA — {len(lsa_df)} source docs ranked", state="complete", expanded=False)
                with st.expander("LSA log"):
                    st.code(buf.getvalue() or "(no output)")
            except Exception as exc:
                s.update(label=f"❌ LSA failed", state="error")
                st.exception(exc)
                stage_error = exc

    # Embeddings branch — dense neural vectors via Qwen3-Embedding-0.6B + FAISS
    if not stage_error:
        with st.status("Embeddings …", expanded=True) as s:
            try:
                if run_embeddings:
                    # Re-run the GPU embedding lookup inside the ROCm Docker container
                    import subprocess
                    proc = subprocess.run(
                        ["docker", "exec", "docplag-rocm", "python", "scripts/embeddings.py",
                         "--doc_id", selected_doc],
                        capture_output=True, text=True,
                    )
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
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
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
    st.markdown("**Scoring each candidate with the LLM…**")
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
    st.dataframe(scores_df, use_container_width=True, hide_index=True)

    if top20_df is not None:
        st.markdown("#### Source Retrieval Fusion (top 20 pre-LLM)")
        st.dataframe(top20_df, use_container_width=True, hide_index=True)
