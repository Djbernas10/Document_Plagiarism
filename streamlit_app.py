import streamlit as st
from pathlib import Path
import tempfile
import re
import math
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Basic text utilities
# -----------------------------

def read_uploaded_file(uploaded_file) -> str:
    """Read TXT/MD/CSV/PDF/DOCX where possible."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.warning(f"Could not read PDF {uploaded_file.name}: {e}")
            return ""

    if name.endswith(".docx"):
        try:
            import docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            document = docx.Document(tmp_path)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            st.warning(f"Could not read DOCX {uploaded_file.name}: {e}")
            return ""

    st.warning(f"Unsupported file type: {uploaded_file.name}")
    return ""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?-]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    words = normalize_text(text).split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= 20:
            chunks.append(" ".join(chunk))
    return chunks


def compare_documents(suspicious_text: str, dataset_docs: Dict[str, str], top_k: int = 10) -> pd.DataFrame:
    """Document-level candidate retrieval using TF-IDF cosine similarity."""
    names = ["SUSPICIOUS_DOCUMENT"] + list(dataset_docs.keys())
    texts = [normalize_text(suspicious_text)] + [normalize_text(t) for t in dataset_docs.values()]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    rows = []
    for name, score in zip(names[1:], scores):
        rows.append({"source_document": name, "similarity_score": round(float(score), 4)})

    return pd.DataFrame(rows).sort_values("similarity_score", ascending=False).head(top_k)


def align_passages(
    suspicious_text: str,
    source_text: str,
    chunk_size: int = 120,
    overlap: int = 30,
    threshold: float = 0.20,
    top_k: int = 10,
) -> pd.DataFrame:
    """Passage-level alignment using chunk similarity."""
    suspicious_chunks = chunk_text(suspicious_text, chunk_size, overlap)
    source_chunks = chunk_text(source_text, chunk_size, overlap)

    if not suspicious_chunks or not source_chunks:
        return pd.DataFrame()

    all_chunks = suspicious_chunks + source_chunks
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(all_chunks)

    suspicious_matrix = matrix[:len(suspicious_chunks)]
    source_matrix = matrix[len(suspicious_chunks):]
    sim_matrix = cosine_similarity(suspicious_matrix, source_matrix)

    rows = []
    for i in range(sim_matrix.shape[0]):
        best_j = int(sim_matrix[i].argmax())
        best_score = float(sim_matrix[i][best_j])
        if best_score >= threshold:
            rows.append({
                "score": round(best_score, 4),
                "suspicious_passage_id": i + 1,
                "source_passage_id": best_j + 1,
                "suspicious_passage": suspicious_chunks[i],
                "matched_source_passage": source_chunks[best_j],
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Plagiarism Detection UI",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Plagiarism Detection Prototype")
st.caption("Upload one suspicious document and compare it against a source dataset.")

with st.sidebar:
    st.header("Settings")
    top_docs = st.slider("Top source documents", 1, 20, 5)
    top_passages = st.slider("Top passage matches", 1, 30, 10)
    similarity_threshold = st.slider("Passage similarity threshold", 0.05, 0.90, 0.20, 0.05)
    chunk_size = st.slider("Chunk size / words", 50, 300, 120, 10)
    overlap = st.slider("Chunk overlap / words", 0, 100, 30, 10)

    st.divider()
    st.info(
        "This is a prototype using TF-IDF cosine similarity. "
        "Later you can replace this comparison layer with ESA, LSA, embeddings, FAISS, or an LLM verifier."
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Suspicious document")
    suspicious_file = st.file_uploader(
        "Upload the document you want to check",
        type=["txt", "md", "csv", "pdf", "docx"],
        accept_multiple_files=False,
    )

with col2:
    st.subheader("2. Source dataset")
    dataset_files = st.file_uploader(
        "Upload source documents from your dataset",
        type=["txt", "md", "csv", "pdf", "docx"],
        accept_multiple_files=True,
    )

run_button = st.button("Run plagiarism comparison", type="primary", use_container_width=True)

if run_button:
    if suspicious_file is None:
        st.error("Upload a suspicious document first.")
        st.stop()

    if not dataset_files:
        st.error("Upload at least one source document from your dataset.")
        st.stop()

    with st.spinner("Reading files..."):
        suspicious_text = read_uploaded_file(suspicious_file)
        dataset_docs = {}
        for file in dataset_files:
            text = read_uploaded_file(file)
            if text.strip():
                dataset_docs[file.name] = text

    if not suspicious_text.strip():
        st.error("The suspicious document could not be read or is empty.")
        st.stop()

    if not dataset_docs:
        st.error("No valid source documents were loaded.")
        st.stop()

    st.success(f"Loaded 1 suspicious document and {len(dataset_docs)} source documents.")

    with st.spinner("Retrieving candidate source documents..."):
        doc_results = compare_documents(suspicious_text, dataset_docs, top_k=top_docs)

    st.subheader("3. Candidate source retrieval")
    st.dataframe(doc_results, use_container_width=True)

    selected_source = st.selectbox(
        "Choose a source document for passage-level alignment",
        doc_results["source_document"].tolist(),
    )

    with st.spinner("Aligning suspicious passages with source passages..."):
        passage_results = align_passages(
            suspicious_text=suspicious_text,
            source_text=dataset_docs[selected_source],
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=similarity_threshold,
            top_k=top_passages,
        )

    st.subheader("4. Passage-level matches")

    if passage_results.empty:
        st.warning("No passage matches found above the selected threshold.")
    else:
        st.dataframe(
            passage_results[["score", "suspicious_passage_id", "source_passage_id"]],
            use_container_width=True,
        )

        for idx, row in passage_results.iterrows():
            with st.expander(f"Match score: {row['score']} | Suspicious #{row['suspicious_passage_id']} → Source #{row['source_passage_id']}"):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Suspicious passage**")
                    st.write(row["suspicious_passage"])
                with right:
                    st.markdown("**Matched source passage**")
                    st.write(row["matched_source_passage"])

        csv = passage_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="plagiarism_passage_matches.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Upload your files, then click **Run plagiarism comparison**.")


# -----------------------------
# Recommended install command
# -----------------------------
# pip install streamlit pandas scikit-learn pypdf python-docx
# streamlit run app.py
