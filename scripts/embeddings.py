from pathlib import Path
from typing import Optional

import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = Path("../datasets/processed/PAN2011_300")
ARTIFACT_DIR = Path("../artifacts/embeddings/embeddings_qwen06b")

SOURCE_CHUNKS_PATH = PROCESSED_DIR / "source_chunks_embeddings.parquet"
SUSPICIOUS_CHUNKS_PATH = PROCESSED_DIR / "suspicious_chunks_embeddings.parquet"
SOURCE_CANONICAL_CHUNKS_PATH = PROCESSED_DIR / "source_chunks.parquet"

MODEL_PATH = Path("../artifacts/models/Qwen3-Embedding-0.6B")

SUSPICIOUS_DOC_ID = "part1__suspicious-document00001.txt"

OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / "embedding_candidates_suspicious_doc_00001.parquet"
OUTPUT_TOP_DOCS_PATH = PROCESSED_DIR / "embedding_top_source_documents_by_mean_score.parquet"

BUILD_INDEX = False


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_embedding_chunks(
    path: Path,
    text_column: str = "embedding_text",
) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required_columns = {
        "chunk_id",
        "doc_id",
        "chunk_index",
        "start_char",
        "end_char",
        text_column,
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""].reset_index(drop=True)

    return df


# ============================================================
# MODEL
# ============================================================

def load_embedding_model(model_path: Path) -> SentenceTransformer:
    import torch

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model folder not found: {model_path.resolve()}")

    print(f"Loading embedding model from: {model_path.resolve()}")
    print("torch version:", torch.__version__)
    print("torch HIP:", torch.version.hip)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch device count:", torch.cuda.device_count())
    print("torch num threads:", torch.get_num_threads())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = SentenceTransformer(str(model_path), device=device)

    # 256 is faster and enough for your 300-word chunk test.
    # Use 512 later only if quality is clearly worse.
    model.max_seq_length = 256

    # FP16 usually improves speed and reduces VRAM usage on GPU.
    if device == "cuda":
        model = model.half()

    print("SentenceTransformer device:", model.device)
    print("Model max_seq_length:", model.max_seq_length)

    return model


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_embedding_index(
    source_chunks_path: Path,
    artifact_dir: Path,
    model_path: Path,
    text_column: str = "embedding_text",
    batch_size: int = 16,
    max_source_chunks: Optional[int] = None,
) -> None:
    """
    Build FAISS index from source chunks.

    Creates:
    - faiss.index
    - source_embedding_metadata.parquet
    - embedding_config.joblib
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_embedding_chunks(
        path=source_chunks_path,
        text_column=text_column,
    )

    if max_source_chunks is not None:
        source_df = source_df.head(max_source_chunks).reset_index(drop=True)

    print(f"Loaded {len(source_df)} source chunks for embedding index")

    model = load_embedding_model(model_path)

    source_texts = source_df[text_column].tolist()

    print("Encoding source chunks...")
    source_embeddings = model.encode(
        source_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"Embedding matrix shape: {source_embeddings.shape}")

    dimension = source_embeddings.shape[1]

    # Inner product over normalized embeddings = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(source_embeddings)

    metadata_columns = [
        "chunk_id",
        "doc_id",
        "chunk_index",
        "start_char",
        "end_char",
    ]

    optional_columns = [
        "file_name",
        "relative_path",
        "part",
        "word_count",
    ]

    metadata_columns += [
        col for col in optional_columns
        if col in source_df.columns
    ]

    source_metadata = source_df[metadata_columns].copy()

    print("Saving FAISS index and metadata...")

    faiss.write_index(
        index,
        str(artifact_dir / "faiss.index"),
    )

    source_metadata.to_parquet(
        artifact_dir / "source_embedding_metadata.parquet",
        index=False,
    )

    config = {
        "model_path": str(Path(model_path).resolve()),
        "dimension": dimension,
        "normalized": True,
        "text_column": text_column,
        "source_chunks_path": str(source_chunks_path),
        "max_source_chunks": max_source_chunks,
        "batch_size": batch_size,
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine_similarity_via_normalized_inner_product",
    }

    joblib.dump(
        config,
        artifact_dir / "embedding_config.joblib",
    )

    print(f"Saved embedding artifacts to: {artifact_dir}")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_embedding_index(artifact_dir: Path):
    artifact_dir = Path(artifact_dir)

    index_path = artifact_dir / "faiss.index"
    metadata_path = artifact_dir / "source_embedding_metadata.parquet"
    config_path = artifact_dir / "embedding_config.joblib"

    for path in [index_path, metadata_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding artifact: {path}")

    print("Loading FAISS index and metadata...")

    index = faiss.read_index(str(index_path))
    source_metadata = pd.read_parquet(metadata_path)
    config = joblib.load(config_path)

    print(f"Loaded FAISS index with {index.ntotal} vectors")
    print(f"Loaded source metadata rows: {len(source_metadata)}")

    return index, source_metadata, config


# ============================================================
# QUERY ONE SUSPICIOUS DOCUMENT
# ============================================================

def retrieve_embedding_candidates_for_suspicious_doc(
    suspicious_chunks_path: Path,
    artifact_dir: Path,
    model_path: Path,
    suspicious_doc_id: str,
    output_path: Path,
    text_column: str = "embedding_text",
    top_k: int = 50,
    batch_size: int = 16,
    max_suspicious_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query the FAISS source index using one selected suspicious document.

    Returns top-k source chunks for each suspicious chunk.
    """

    index, source_metadata, config = load_embedding_index(artifact_dir)

    current_model_path = str(Path(model_path).resolve())
    indexed_model_path = str(Path(config["model_path"]).resolve())

    if indexed_model_path != current_model_path:
        raise ValueError(
            f"Model mismatch.\n"
            f"Index was built with: {indexed_model_path}\n"
            f"Current model path:   {current_model_path}"
        )

    model = load_embedding_model(model_path)

    suspicious_df = load_embedding_chunks(
        path=suspicious_chunks_path,
        text_column=text_column,
    )

    suspicious_df = suspicious_df[
        suspicious_df["doc_id"] == suspicious_doc_id
    ].copy()

    if suspicious_df.empty:
        raise ValueError(f"No suspicious chunks found for doc_id: {suspicious_doc_id}")

    if max_suspicious_chunks is not None:
        suspicious_df = suspicious_df.head(max_suspicious_chunks).reset_index(drop=True)

    print(f"Selected suspicious document: {suspicious_doc_id}")
    print(f"Suspicious chunks to query: {len(suspicious_df)}")
    print(f"Searching top-{top_k} source chunks per suspicious chunk")

    suspicious_texts = suspicious_df[text_column].tolist()

    print("Encoding suspicious chunks...")
    suspicious_embeddings = model.encode(
        suspicious_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"Suspicious embedding matrix shape: {suspicious_embeddings.shape}")

    scores, indices = index.search(suspicious_embeddings, top_k)

    results = []

    for suspicious_i, suspicious_row in enumerate(suspicious_df.itertuples(index=False)):
        for rank in range(top_k):
            source_idx = int(indices[suspicious_i, rank])
            score = float(scores[suspicious_i, rank])

            if source_idx < 0:
                continue

            source_row = source_metadata.iloc[source_idx]

            results.append({
                "suspicious_chunk_id": suspicious_row.chunk_id,
                "suspicious_doc_id": suspicious_row.doc_id,
                "suspicious_chunk_index": suspicious_row.chunk_index,
                "suspicious_start_char": suspicious_row.start_char,
                "suspicious_end_char": suspicious_row.end_char,

                "source_chunk_id": source_row["chunk_id"],
                "source_doc_id": source_row["doc_id"],
                "source_chunk_index": source_row["chunk_index"],
                "source_start_char": source_row["start_char"],
                "source_end_char": source_row["end_char"],

                "embedding_score": score,
                "embedding_rank": rank + 1,
            })

    output_df = pd.DataFrame(results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_path, index=False)

    print(f"Saved embedding candidates to: {output_path}")
    print(f"Candidate rows: {len(output_df)}")

    return output_df


# ============================================================
# DOCUMENT-LEVEL MEAN SCORE
# ============================================================

def get_top_source_documents_by_mean_embedding_score(
    candidates_df: pd.DataFrame,
    source_chunks_path: Path,
    suspicious_doc_id: str,
    top_n: int = 50,
    min_match_count: int = 1,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Rank unique source documents by mean embedding score.
    """

    required_columns = {
        "suspicious_doc_id",
        "suspicious_chunk_id",
        "source_doc_id",
        "source_chunk_id",
        "embedding_score",
    }

    missing = required_columns - set(candidates_df.columns)
    if missing:
        raise ValueError(f"Missing columns in candidates_df: {missing}")

    filtered_df = candidates_df[
        candidates_df["suspicious_doc_id"] == suspicious_doc_id
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No candidates found for suspicious_doc_id: {suspicious_doc_id}")

    grouped_df = (
        filtered_df
        .groupby("source_doc_id")
        .agg(
            mean_embedding_score=("embedding_score", "mean"),
            max_embedding_score=("embedding_score", "max"),
            min_embedding_score=("embedding_score", "min"),
            match_count=("embedding_score", "count"),
            unique_source_chunks=("source_chunk_id", "nunique"),
            unique_suspicious_chunks=("suspicious_chunk_id", "nunique"),
        )
        .reset_index()
    )

    grouped_df = grouped_df[
        grouped_df["match_count"] >= min_match_count
    ].copy()

    grouped_df = (
        grouped_df
        .sort_values(
            ["mean_embedding_score", "match_count"],
            ascending=[False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    grouped_df["source_doc_rank"] = range(1, len(grouped_df) + 1)

    grouped_df = grouped_df[
        [
            "source_doc_rank",
            "source_doc_id",
            "mean_embedding_score",
            "max_embedding_score",
            "min_embedding_score",
            "match_count",
            "unique_source_chunks",
            "unique_suspicious_chunks",
        ]
    ]

    source_meta_df = pd.read_parquet(
        source_chunks_path,
        columns=["doc_id", "relative_path"],
    ).drop_duplicates("doc_id")

    source_meta_df = source_meta_df.rename(columns={
        "doc_id": "source_doc_id",
        "relative_path": "source_relative_path",
    })

    grouped_df = grouped_df.merge(
        source_meta_df,
        on="source_doc_id",
        how="left",
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grouped_df.to_parquet(output_path, index=False)

        print(f"Saved top source documents to: {output_path}")

    return grouped_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    SUSPICIOUS_DOC_ID = "part1__suspicious-document00007.txt"
    BUILD_INDEX = True
    if BUILD_INDEX:
        build_embedding_index(
            source_chunks_path=SOURCE_CHUNKS_PATH,
            artifact_dir=ARTIFACT_DIR,
            model_path=MODEL_PATH,
            batch_size=24,
            max_source_chunks=300_000,
        )

    candidates_df = retrieve_embedding_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        model_path=MODEL_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        top_k=500,
        batch_size=16,
        max_suspicious_chunks=None,
    )

    top_sources_df = get_top_source_documents_by_mean_embedding_score(
        candidates_df=candidates_df,
        source_chunks_path=SOURCE_CANONICAL_CHUNKS_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        top_n=20,
        min_match_count=1,
        output_path=OUTPUT_TOP_DOCS_PATH,
    )

    print("\nTOP SOURCE DOCUMENTS BY MEAN EMBEDDING SCORE")
    print(top_sources_df.to_string(index=False))