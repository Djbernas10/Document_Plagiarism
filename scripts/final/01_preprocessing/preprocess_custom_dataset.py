"""Run the same preprocessing pipeline used for PAN2011_300 (chunk_size=300,
overlap=150, min_words=60) against the custom_dataset/ corpus.

Run from project root:
    python scripts/final/01_preprocessing/preprocess_custom_dataset.py
"""

from pathlib import Path

from preprocessing import full_preprocessing_pipeline

if __name__ == "__main__":
    full_preprocessing_pipeline(
        SOURCE_FOLDER=Path("../../../datasets/custom_dataset/source_documents"),
        SUSPICIOUS_FOLDER=Path("../../../datasets/custom_dataset/suspicious_documents"),
        OUTPUT_DIR=Path("../../../datasets/processed/custom_300"),
        batch_size=500,
        chunk_size=300,
        overlap=150,
        min_words=60,
    )
