"""Build the LSA index for the custom_dataset corpus (datasets/processed/custom_300),
mirroring the PAN2011_300 LSA index build (same TF-IDF + TruncatedSVD config), but
writing to a separate artifacts/lsa_custom/ directory so the PAN2011 index is untouched.

Run from project root:
    python scripts/final/03_index_creation/build_lsa_index_custom.py
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "custom_300"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "lsa_custom"

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


def build_lsa_index(
    source_chunks_path: Path,
    artifact_dir: Path,
    text_column: str = "lsa_esa_text",
    n_components: int = 200,
    max_features: int = 100_000,
    max_source_chunks: Optional[int] = None,
) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_lsa_esa_chunks(path=source_chunks_path, text_column=text_column)
    if max_source_chunks is not None:
        source_df = source_df.head(max_source_chunks).reset_index(drop=True)

    source_texts = source_df[text_column].tolist()
    print(f"Loaded {len(source_df)} source chunks")
    print("Fitting TF-IDF...")

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

    source_tfidf = vectorizer.fit_transform(source_texts)
    print(f"TF-IDF matrix shape: {source_tfidf.shape}")

    feature_count = source_tfidf.shape[1]
    actual_components = min(n_components, feature_count - 1)
    if actual_components < 2:
        raise ValueError(f"Not enough TF-IDF features for LSA. Feature count: {feature_count}")

    print(f"Fitting TruncatedSVD with {actual_components} components...")
    svd = TruncatedSVD(n_components=actual_components, random_state=42, n_iter=5)
    source_lsa = svd.fit_transform(source_tfidf)

    print("Normalizing source LSA vectors...")
    source_lsa = normalize(source_lsa, norm="l2", axis=1)

    metadata_columns = ["chunk_id", "doc_id", "chunk_index", "start_char", "end_char"]
    optional_columns = ["file_name", "relative_path", "part", "word_count"]
    metadata_columns += [col for col in optional_columns if col in source_df.columns]
    source_metadata = source_df[metadata_columns].copy()

    print("Saving LSA artifacts...")
    np.save(artifact_dir / "source_lsa_vectors.npy", source_lsa)
    source_metadata.to_parquet(artifact_dir / "source_lsa_metadata.parquet", index=False)
    joblib.dump(svd, artifact_dir / "svd_model.joblib")
    joblib.dump(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")

    print(f"Saved LSA artifacts to: {artifact_dir}")
    print(f"Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")


if __name__ == "__main__":
    build_lsa_index(
        source_chunks_path=SOURCE_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        n_components=200,
        max_features=100_000,
        max_source_chunks=None,
    )
