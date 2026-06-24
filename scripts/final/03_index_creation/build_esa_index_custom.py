"""Build the ESA-style (collection-based, TF-IDF-as-concept-space) index for the
custom_dataset corpus (datasets/processed/custom_300), mirroring the PAN2011_300
ESA index build, but writing to a separate artifacts/esa_custom/ directory so the
PAN2011 index is untouched.

Run from project root:
    python scripts/final/03_index_creation/build_esa_index_custom.py
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_300"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "esa_custom"

SOURCE_CHUNKS_PATH = PROCESSED_DIR / "source_chunks_lsa_esa.parquet"


def load_lsa_esa_chunks(path: Path, text_column: str = "lsa_esa_text") -> pd.DataFrame:
    df = pd.read_parquet(path)

    required_columns = {"chunk_id", "doc_id", "chunk_index", "start_char", "end_char", text_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""].reset_index(drop=True)
    return df


def build_esa_index(
    source_chunks_path: Path,
    artifact_dir: Path,
    text_column: str = "lsa_esa_text",
    max_features: int = 100_000,
    max_source_chunks: Optional[int] = None,
) -> None:
    """
    Collection-based ESA-style retrieval over source chunks (TF-IDF terms as the
    explicit concept space), not Wikipedia ESA. Matches the PAN2011_300 build.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_lsa_esa_chunks(path=source_chunks_path, text_column=text_column)
    if max_source_chunks is not None:
        source_df = source_df.head(max_source_chunks).reset_index(drop=True)

    source_texts = source_df[text_column].tolist()
    print(f"Loaded {len(source_df)} source chunks")
    print("Fitting ESA TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        norm="l2",
    )

    source_esa = vectorizer.fit_transform(source_texts)
    source_esa = normalize(source_esa, norm="l2", axis=1)
    print(f"ESA source matrix shape: {source_esa.shape}")

    metadata_columns = ["chunk_id", "doc_id", "chunk_index", "start_char", "end_char"]
    optional_columns = ["file_name", "relative_path", "part", "word_count"]
    metadata_columns += [col for col in optional_columns if col in source_df.columns]
    source_metadata = source_df[metadata_columns].copy()

    print("Saving ESA artifacts...")
    joblib.dump(vectorizer, artifact_dir / "esa_tfidf_vectorizer.joblib")
    sparse.save_npz(artifact_dir / "source_esa_vectors.npz", source_esa)
    source_metadata.to_parquet(artifact_dir / "source_esa_metadata.parquet", index=False)

    config = {
        "text_column": text_column,
        "max_features": max_features,
        "max_source_chunks": max_source_chunks,
        "source_chunks_path": str(source_chunks_path),
        "matrix_shape": source_esa.shape,
        "similarity": "cosine_similarity_via_l2_normalized_dot_product",
        "note": "Collection-based ESA-style retrieval over source chunks, not Wikipedia ESA.",
    }
    joblib.dump(config, artifact_dir / "esa_config.joblib")

    print(f"Saved ESA artifacts to: {artifact_dir}")


if __name__ == "__main__":
    build_esa_index(
        source_chunks_path=SOURCE_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        text_column="lsa_esa_text",
        max_features=100_000,
        max_source_chunks=None,
    )
