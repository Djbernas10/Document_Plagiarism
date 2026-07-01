from pathlib import Path
import pandas as pd

from branch_faiss import FaissEmbeddingBranch


SOURCE_DOCUMENTS_PATH = Path("processed_files/source_documents.parquet")

FAISS_INDEX_PATH = Path("vector_db/source_faiss.index")
FAISS_METADATA_PATH = Path("vector_db/source_metadata.parquet")


def load_source_docs_from_parquet(parquet_path: Path) -> dict:
    """
    Load source documents from Parquet and convert to:
    {
        "source-document00001.txt": "full text..."
    }
    """

    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    required_columns = {"relative_path", "text"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in Parquet: {missing_columns}")

    source_docs = {}

    for _, row in df.iterrows():
        doc_id = row["relative_path"]
        text = row["text"]

        if isinstance(text, str) and text.strip():
            source_docs[doc_id] = text

    return source_docs


def build_faiss_db():
    print("Loading source documents...")
    source_docs = load_source_docs_from_parquet(SOURCE_DOCUMENTS_PATH)

    print(f"Loaded {len(source_docs)} source documents.")

    print("Building FAISS embedding index...")
    faiss_branch = FaissEmbeddingBranch(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=180,
        overlap=50,
        min_words=40,
        batch_size=32,
    )

    faiss_branch.build_index(source_docs)

    print("Saving FAISS DB...")
    faiss_branch.save(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
    )

    print("Done.")
    print(f"FAISS index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {FAISS_METADATA_PATH}")


if __name__ == "__main__":
    build_faiss_db()