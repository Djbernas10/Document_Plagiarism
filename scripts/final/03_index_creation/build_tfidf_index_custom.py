"""Build the sharded hashed TF-IDF index (char mode) for the custom_dataset corpus
(datasets/processed/custom_300), mirroring the PAN2011_300 TF-IDF index build, but
writing to a separate artifacts/tfidf_hashing_custom/ directory so the PAN2011
index is untouched.

Run from project root:
    python scripts/final/03_index_creation/build_tfidf_index_custom.py
"""

import gc
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_300"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_hashing_custom"

SOURCE_CHUNKS_PATH = PROCESSED_DIR / "source_chunks_lsa_esa.parquet"

TFIDF_MODE: Literal["char", "word"] = "char"


def cleanup_memory() -> None:
    gc.collect()


def load_tfidf_chunks(path: Path, text_column: str = "lsa_esa_text") -> pd.DataFrame:
    df = pd.read_parquet(path)

    required_columns = {"chunk_id", "doc_id", "chunk_index", "start_char", "end_char", text_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""].reset_index(drop=True)
    return df


def make_hashing_vectorizer(mode: Literal["char", "word"], n_features: int) -> HashingVectorizer:
    if mode == "char":
        return HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(5, 7),
            n_features=n_features,
            lowercase=True,
            strip_accents="unicode",
            alternate_sign=False,
            norm=None,
            binary=False,
            dtype=np.float32,
        )
    if mode == "word":
        return HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            n_features=n_features,
            lowercase=True,
            strip_accents="unicode",
            alternate_sign=False,
            norm=None,
            binary=False,
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported TF-IDF mode: {mode}")


def compute_hashed_document_frequencies(
    texts: list[str],
    vectorizer: HashingVectorizer,
    n_features: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    df_counts = np.zeros(n_features, dtype=np.int64)
    total_docs = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print("Computing hashed document frequencies...")
    for start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="DF batches"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        counts = vectorizer.transform(batch_texts).tocsr()
        counts.data[:] = 1.0
        batch_df = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)

        df_counts += batch_df
        total_docs += len(batch_texts)

        del counts, batch_df
        if total_docs % (batch_size * 50) == 0:
            cleanup_memory()

    return df_counts, total_docs


def compute_sklearn_style_idf(df_counts: np.ndarray, total_docs: int) -> np.ndarray:
    idf = np.log((1.0 + total_docs) / (1.0 + df_counts.astype(np.float32))) + 1.0
    return idf.astype(np.float32)


def transform_texts_to_tfidf(texts: list[str], vectorizer: HashingVectorizer, idf: np.ndarray) -> sparse.csr_matrix:
    counts = vectorizer.transform(texts).tocsr().astype(np.float32)
    if counts.nnz > 0:
        counts.data = 1.0 + np.log(counts.data)
    tfidf = counts.multiply(idf).tocsr().astype(np.float32)
    tfidf = normalize(tfidf, norm="l2", axis=1, copy=False)
    return tfidf.astype(np.float32)


def build_tfidf_index(
    source_chunks_path: Path,
    artifact_dir: Path,
    text_column: str = "lsa_esa_text",
    mode: Literal["char", "word"] = "char",
    n_features: int = 2**20,
    batch_size: int = 2048,
    shard_size_rows: int = 100_000,
    max_source_chunks: Optional[int] = None,
) -> None:
    artifact_dir = Path(artifact_dir) / mode
    shard_dir = artifact_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_tfidf_chunks(path=source_chunks_path, text_column=text_column)
    if max_source_chunks is not None:
        source_df = source_df.head(max_source_chunks).reset_index(drop=True)

    print(f"Loaded {len(source_df)} source chunks")
    print(f"Building HASHED TF-IDF index with mode: {mode}")
    print(f"Hash features: {n_features}")
    print(f"Batch size: {batch_size}")
    print(f"Shard size rows: {shard_size_rows}")

    metadata_columns = ["chunk_id", "doc_id", "chunk_index", "start_char", "end_char"]
    optional_columns = ["file_name", "relative_path", "part", "word_count"]
    metadata_columns += [col for col in optional_columns if col in source_df.columns]

    source_metadata = source_df[metadata_columns].copy()
    source_texts = source_df[text_column].tolist()

    del source_df
    cleanup_memory()

    vectorizer = make_hashing_vectorizer(mode=mode, n_features=n_features)

    df_counts, total_docs = compute_hashed_document_frequencies(
        texts=source_texts, vectorizer=vectorizer, n_features=n_features, batch_size=batch_size,
    )
    idf = compute_sklearn_style_idf(df_counts=df_counts, total_docs=total_docs)

    print("Finished computing IDF")
    print(f"Total documents/chunks: {total_docs}")

    del df_counts
    cleanup_memory()

    print("Building and saving TF-IDF shards...")

    shard_infos = []
    shard_id = 0
    shard_start_row = 0
    current_parts = []
    current_rows = 0

    total_batches = (len(source_texts) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(source_texts), batch_size), total=total_batches, desc="TF-IDF shard batches"):
        end = min(start + batch_size, len(source_texts))
        batch_texts = source_texts[start:end]

        batch_tfidf = transform_texts_to_tfidf(texts=batch_texts, vectorizer=vectorizer, idf=idf)

        current_parts.append(batch_tfidf)
        current_rows += batch_tfidf.shape[0]

        should_save_shard = current_rows >= shard_size_rows
        is_last_batch = end == len(source_texts)

        if should_save_shard or is_last_batch:
            shard_matrix = sparse.vstack(current_parts, format="csr").astype(np.float32)

            # Stored relative to artifact_dir so it resolves correctly regardless of CWD.
            relative_shard_path = f"shards/source_tfidf_shard_{shard_id:05d}.npz"
            shard_path = artifact_dir / relative_shard_path

            print(f"\nSaving TF-IDF shard {shard_id} | rows={shard_matrix.shape[0]} | nnz={shard_matrix.nnz} | path={shard_path}")
            sparse.save_npz(shard_path, shard_matrix, compressed=True)

            shard_infos.append({
                "shard_id": shard_id,
                "shard_path": relative_shard_path,
                "start_row": shard_start_row,
                "num_rows": shard_matrix.shape[0],
                "end_row": shard_start_row + shard_matrix.shape[0],
                "num_features": shard_matrix.shape[1],
                "nnz": shard_matrix.nnz,
            })

            shard_start_row += shard_matrix.shape[0]
            shard_id += 1

            del shard_matrix, current_parts
            current_parts = []
            current_rows = 0
            cleanup_memory()

    del source_texts
    cleanup_memory()

    total_rows = sum(info["num_rows"] for info in shard_infos)
    if len(source_metadata) != total_rows:
        raise ValueError(f"Metadata/shard mismatch: metadata rows={len(source_metadata)}, TF-IDF rows={total_rows}")

    print("Saving TF-IDF artifacts...")
    joblib.dump(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")
    np.save(artifact_dir / "tfidf_idf.npy", idf)
    source_metadata.to_parquet(artifact_dir / "source_tfidf_metadata.parquet", index=False)

    shard_info_df = pd.DataFrame(shard_infos)
    shard_info_df.to_parquet(artifact_dir / "tfidf_shards.parquet", index=False)

    config = {
        "mode": mode,
        "text_column": text_column,
        "n_features": n_features,
        "batch_size": batch_size,
        "shard_size_rows": shard_size_rows,
        "max_source_chunks": max_source_chunks,
        "source_chunks_path": str(source_chunks_path),
        "num_shards": len(shard_infos),
        "total_rows": total_rows,
        "similarity": "cosine_similarity_via_l2_normalized_dot_product",
        "vectorizer": "HashingVectorizer",
        "idf_formula": "log((1 + n_docs) / (1 + df)) + 1",
    }
    joblib.dump(config, artifact_dir / "tfidf_config.joblib")

    cleanup_memory()
    print(f"Saved TF-IDF artifacts to: {artifact_dir}")
    print(f"Total shards: {len(shard_infos)}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    build_tfidf_index(
        source_chunks_path=SOURCE_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        text_column="lsa_esa_text",
        mode=TFIDF_MODE,
        n_features=2**20,
        batch_size=2048,
        shard_size_rows=100_000,
        max_source_chunks=None,
    )
