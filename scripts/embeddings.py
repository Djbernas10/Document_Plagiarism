from pathlib import Path
from typing import Optional

import gc

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse


# ============================================================
# CONFIG
# ============================================================
# NOTE: PROCESSED_DIR/ARTIFACT_DIR/OUTPUT_* are resolved per-dataset in
# run_embedding_lookup() so both CLI and HTTP service calls share one path.

DATASET_PATHS = {
    "pan2011": {
        "processed_dir": Path("datasets/processed/PAN2011_300"),
        "artifact_dir": Path("artifacts/embeddings/embeddings_qwen06b"),
    },
    "custom": {
        "processed_dir": Path("datasets/processed/custom_300"),
        "artifact_dir": Path("artifacts/embeddings_custom/embeddings_qwen06b"),
    },
}

MODEL_PATH = Path("artifacts/models/Qwen3-Embedding-0.6B")

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

    model.max_seq_length = 512

    if device == "cuda":
        model = model.half()

    print("SentenceTransformer device:", model.device)
    print("Model max_seq_length:", model.max_seq_length)

    return model


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory() -> None:
    import torch

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ============================================================
# Validate and resume FAISS Shard
# ============================================================

import re

def inspect_existing_faiss_shards(shard_dir: Path) -> tuple[list[dict], int, int]:
    """
    Validate existing FAISS shards and return:
    - existing shard_infos
    - total already indexed vectors
    - next shard_id

    Raises if any shard is corrupt.
    """
    shard_dir = Path(shard_dir)

    shard_files = sorted(
        shard_dir.glob("faiss_shard_*.index"),
        key=lambda p: int(re.search(r"faiss_shard_(\d+)\.index", p.name).group(1)),
    )

    shard_infos = []
    shard_start_vector_id = 0
    max_shard_id = -1

    if not shard_files:
        return [], 0, 0

    print(f"Found {len(shard_files)} existing FAISS shard(s). Validating...")

    for shard_file in shard_files:
        match = re.search(r"faiss_shard_(\d+)\.index", shard_file.name)
        if not match:
            continue

        shard_id = int(match.group(1))

        try:
            index = faiss.read_index(str(shard_file))
            num_vectors = int(index.ntotal)
        except Exception as e:
            raise RuntimeError(
                f"Corrupt FAISS shard detected: {shard_file}\n"
                f"Delete only this shard, then rerun.\n"
                f"Original error: {e}"
            )

        shard_infos.append({
            "shard_id": shard_id,
            "shard_path": str(shard_file),
            "start_vector_id": shard_start_vector_id,
            "num_vectors": num_vectors,
            "end_vector_id": shard_start_vector_id + num_vectors,
        })

        print(f"OK {shard_file.name} | vectors={num_vectors}")

        shard_start_vector_id += num_vectors
        max_shard_id = max(max_shard_id, shard_id)

    already_indexed_vectors = shard_start_vector_id
    next_shard_id = max_shard_id + 1

    print("\nResume state:")
    print(f"  Existing valid shards: {len(shard_infos)}")
    print(f"  Already indexed vectors: {already_indexed_vectors:,}")
    print(f"  Next shard id: {next_shard_id:05d}")

    return shard_infos, already_indexed_vectors, next_shard_id
# ============================================================
# BUILD SHARDED FAISS INDEX
# ============================================================

def build_embedding_index(
    source_chunks_path: Path,
    artifact_dir: Path,
    model_path: Path,
    text_column: str = "embedding_text",
    batch_size: int = 24,
    max_source_chunks: Optional[int] = None,
    save_every_batches: int = 2000,
) -> None:
    """
    Option A.

    Build sharded FAISS indexes from source chunks.

    Creates:
    - shards/faiss_shard_00000.index
    - shards/faiss_shard_00001.index
    - source_embedding_metadata.parquet
    - faiss_shards.parquet
    - embedding_config.joblib
    """

    artifact_dir = Path(artifact_dir)
    shard_dir = artifact_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_embedding_chunks(
        path=source_chunks_path,
        text_column=text_column,
    )

    if max_source_chunks is not None:
        source_df = source_df.head(max_source_chunks).reset_index(drop=True)

    print(f"Loaded {len(source_df)} source chunks for embedding index")

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

    existing_shard_infos, already_indexed_vectors, shard_id = inspect_existing_faiss_shards(shard_dir)
    print("\n========== RESUME CHECK ==========")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Shard dir: {shard_dir}")
    print(f"Existing valid shards: {len(existing_shard_infos)}")
    print(f"Already indexed vectors: {already_indexed_vectors:,}")
    print(f"Next shard id to write: {shard_id:05d}")
    print(f"Total source rows: {len(source_df):,}")
    print(f"Remaining rows to encode: {len(source_df) - already_indexed_vectors:,}")
    print("==================================\n")

    if already_indexed_vectors > len(source_df):
        raise ValueError(
            f"Existing FAISS shards contain more vectors than source rows: "
            f"{already_indexed_vectors:,} vectors vs {len(source_df):,} rows. "
            f"Wrong artifact directory or changed source chunks file."
        )

    if already_indexed_vectors == len(source_df):
        print("All source chunks are already indexed. Rebuilding metadata/config only.")

        source_metadata.to_parquet(
            artifact_dir / "source_embedding_metadata.parquet",
            index=False,
        )

        shard_info_df = pd.DataFrame(existing_shard_infos)
        shard_info_df.to_parquet(
            artifact_dir / "faiss_shards.parquet",
            index=False,
        )

        config = {
            "model_path": str(Path(model_path).resolve()),
            "dimension": None,
            "normalized": True,
            "text_column": text_column,
            "source_chunks_path": str(source_chunks_path),
            "max_source_chunks": max_source_chunks,
            "batch_size": batch_size,
            "faiss_index_type": "IndexFlatIP_sharded",
            "similarity": "cosine_similarity_via_normalized_inner_product",
            "save_every_batches": save_every_batches,
            "num_shards": len(existing_shard_infos),
            "total_vectors": already_indexed_vectors,
        }

        joblib.dump(config, artifact_dir / "embedding_config.joblib")
        return

    if already_indexed_vectors > 0:
        print(
            f"Resuming: skipping first {already_indexed_vectors:,} "
            f"already-indexed chunks out of {len(source_df):,}."
        )

    source_texts = source_df[text_column].iloc[already_indexed_vectors:].tolist()

    del source_df
    cleanup_memory()

    model = load_embedding_model(model_path)

    print("Encoding remaining source chunks into FAISS shards...")

    current_index = None
    dimension = None
    shard_infos = existing_shard_infos.copy()

    shard_start_vector_id = already_indexed_vectors

    total_batches = (len(source_texts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(
        tqdm(
            range(0, len(source_texts), batch_size),
            total=total_batches,
            desc="Encoding source batches",
        ),
        start=1,
    ):
        end = min(start + batch_size, len(source_texts))
        batch_texts = source_texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        if current_index is None:
            dimension = batch_embeddings.shape[1]
            current_index = faiss.IndexFlatIP(dimension)

            print(f"Embedding dimension: {dimension}")
            print("FAISS index type: IndexFlatIP_sharded")
            print("Similarity: cosine similarity via normalized inner product")

        current_index.add(batch_embeddings)

        del batch_embeddings

        should_save_shard = batch_number % save_every_batches == 0
        is_last_batch = batch_number == total_batches

        if should_save_shard or is_last_batch:
            shard_path = shard_dir / f"faiss_shard_{shard_id:05d}.index"
            shard_vector_count = current_index.ntotal

            print(
                f"\nSaving shard {shard_id} | "
                f"vectors={shard_vector_count} | "
                f"path={shard_path}"
            )

            faiss.write_index(current_index, str(shard_path))

            shard_infos.append({
                "shard_id": shard_id,
                "shard_path": str(shard_path),
                "start_vector_id": shard_start_vector_id,
                "num_vectors": shard_vector_count,
                "end_vector_id": shard_start_vector_id + shard_vector_count,
            })

            pd.DataFrame(shard_infos).to_parquet(
                artifact_dir / "faiss_shards.parquet",
                index=False,
            )


            shard_start_vector_id += shard_vector_count
            shard_id += 1

            del current_index
            current_index = None

            cleanup_memory()

    del source_texts
    del model
    cleanup_memory()

    total_vectors = sum(info["num_vectors"] for info in shard_infos)

    if len(source_metadata) != total_vectors:
        raise ValueError(
            f"Metadata/index mismatch: metadata rows={len(source_metadata)}, "
            f"FAISS vectors={total_vectors}"
        )

    print("Saving source metadata...")

    source_metadata.to_parquet(
        artifact_dir / "source_embedding_metadata.parquet",
        index=False,
    )

    shard_info_df = pd.DataFrame(shard_infos)

    shard_info_df.to_parquet(
        artifact_dir / "faiss_shards.parquet",
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
        "faiss_index_type": "IndexFlatIP_sharded",
        "similarity": "cosine_similarity_via_normalized_inner_product",
        "save_every_batches": save_every_batches,
        "num_shards": len(shard_infos),
        "total_vectors": total_vectors,
    }

    joblib.dump(
        config,
        artifact_dir / "embedding_config.joblib",
    )

    cleanup_memory()

    print(f"Saved sharded embedding artifacts to: {artifact_dir}")
    print(f"Total shards: {len(shard_infos)}")
    print(f"Total vectors: {total_vectors}")


# ============================================================
# OPTION B: MERGE SHARDS INTO ONE INDEX
# ============================================================

def merge_faiss_shards(
    artifact_dir: Path,
    output_index_name: str = "faiss.index",
) -> None:
    """
    Option B.

    Merge all FAISS shards into one single FAISS IndexFlatIP index.

    Warning:
    This may use a lot of RAM. Only run after shards are already built.
    """

    artifact_dir = Path(artifact_dir)
    shards_path = artifact_dir / "faiss_shards.parquet"
    output_index_path = artifact_dir / output_index_name

    if not shards_path.exists():
        raise FileNotFoundError(f"Missing shard metadata: {shards_path}")

    shard_info_df = pd.read_parquet(shards_path)

    if shard_info_df.empty:
        raise ValueError("No FAISS shards found to merge.")

    merged_index = None
    total_vectors = 0

    print(f"Merging {len(shard_info_df)} FAISS shards...")

    for shard_row in tqdm(
        shard_info_df.itertuples(index=False),
        total=len(shard_info_df),
        desc="Merging FAISS shards",
    ):
        shard_path = Path(shard_row.shard_path)

        if not shard_path.exists():
            raise FileNotFoundError(f"Missing FAISS shard: {shard_path}")

        shard_index = faiss.read_index(str(shard_path))

        if merged_index is None:
            dimension = shard_index.d
            merged_index = faiss.IndexFlatIP(dimension)

        if shard_index.d != merged_index.d:
            raise ValueError(
                f"Shard dimension mismatch: shard={shard_index.d}, "
                f"merged={merged_index.d}"
            )

        vectors = shard_index.reconstruct_n(0, shard_index.ntotal)
        vectors = vectors.astype("float32")

        merged_index.add(vectors)
        total_vectors += shard_index.ntotal

        del shard_index
        del vectors
        cleanup_memory()

        print(f"Total merged vectors so far: {total_vectors}")

    if merged_index is None:
        raise ValueError("Merged index was not created.")

    print(f"Saving merged FAISS index to: {output_index_path}")
    print(f"Final merged vectors: {merged_index.ntotal}")

    faiss.write_index(
        merged_index,
        str(output_index_path),
    )

    del merged_index
    cleanup_memory()

    print("Finished merging FAISS shards.")


# ============================================================
# LOAD SHARDED INDEX METADATA
# ============================================================

def load_embedding_index(artifact_dir: Path):
    """
    Load sharded FAISS metadata.

    Does not load all FAISS shards into RAM.
    Individual shards are loaded during search.
    """

    artifact_dir = Path(artifact_dir)

    metadata_path = artifact_dir / "source_embedding_metadata.parquet"
    shards_path = artifact_dir / "faiss_shards.parquet"
    config_path = artifact_dir / "embedding_config.joblib"

    for path in [metadata_path, shards_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding artifact: {path}")

    print("Loading sharded FAISS metadata...")

    source_metadata = pd.read_parquet(metadata_path)
    shard_info_df = pd.read_parquet(shards_path)
    config = joblib.load(config_path)

    print(f"Loaded source metadata rows: {len(source_metadata)}")
    print(f"Loaded FAISS shards: {len(shard_info_df)}")

    expected_total_vectors = int(shard_info_df["num_vectors"].sum())

    if len(source_metadata) != expected_total_vectors:
        raise ValueError(
            f"Metadata/shard mismatch: metadata rows={len(source_metadata)}, "
            f"FAISS vectors={expected_total_vectors}"
        )

    return shard_info_df, source_metadata, config


# ============================================================
# SHARDED FAISS SEARCH
# ============================================================

def search_sharded_faiss_index(
    query_embeddings: np.ndarray,
    shard_info_df: pd.DataFrame,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search all FAISS shards and return global top-k results.

    Returns:
    - final_scores: shape (num_queries, top_k)
    - final_indices: global source metadata row indices, shape (num_queries, top_k)
    """

    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype("float32")

    all_scores = []
    all_indices = []

    for shard_row in tqdm(
        shard_info_df.itertuples(index=False),
        total=len(shard_info_df),
        desc="Searching FAISS shards",
    ):
        shard_path = Path(shard_row.shard_path)
        shard_start_vector_id = int(shard_row.start_vector_id)

        if not shard_path.exists():
            raise FileNotFoundError(f"Missing FAISS shard: {shard_path}")

        index = faiss.read_index(str(shard_path))

        shard_scores, shard_indices = index.search(query_embeddings, top_k)

        valid_mask = shard_indices >= 0
        shard_global_indices = shard_indices.copy()
        shard_global_indices[valid_mask] += shard_start_vector_id

        all_scores.append(shard_scores)
        all_indices.append(shard_global_indices)

        del index
        cleanup_memory()

    combined_scores = np.concatenate(all_scores, axis=1)
    combined_indices = np.concatenate(all_indices, axis=1)

    actual_top_k = min(top_k, combined_scores.shape[1])

    top_positions = np.argpartition(
        -combined_scores,
        kth=actual_top_k - 1,
        axis=1,
    )[:, :actual_top_k]

    final_scores = np.take_along_axis(combined_scores, top_positions, axis=1)
    final_indices = np.take_along_axis(combined_indices, top_positions, axis=1)

    sorted_positions = np.argsort(-final_scores, axis=1)

    final_scores = np.take_along_axis(final_scores, sorted_positions, axis=1)
    final_indices = np.take_along_axis(final_indices, sorted_positions, axis=1)

    return final_scores, final_indices


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
    top_k: int = 100,
    batch_size: int = 8,
    max_suspicious_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query the sharded FAISS source index using one selected suspicious document.

    Returns top-k source chunks for each suspicious chunk.
    """

    shard_info_df, source_metadata, config = load_embedding_index(artifact_dir)

    current_model_path = Path(model_path).resolve()
    indexed_model_path = Path(config["model_path"])

    # The index config may have been written inside a container or another
    # checkout path. Compare the model identity, not the machine-specific
    # absolute prefix.
    current_model_id = current_model_path.name
    indexed_model_id = indexed_model_path.name

    if indexed_model_id != current_model_id:
        raise ValueError(
            f"Model mismatch.\n"
            f"Index was built with: {config['model_path']}\n"
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
    print(f"Searching global top-{top_k} source chunks per suspicious chunk")

    suspicious_texts = suspicious_df[text_column].tolist()

    print("Encoding suspicious chunks...")
    suspicious_embeddings = model.encode(
        suspicious_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    del suspicious_texts
    del model
    cleanup_memory()

    print(f"Suspicious embedding matrix shape: {suspicious_embeddings.shape}")

    scores, indices = search_sharded_faiss_index(
        query_embeddings=suspicious_embeddings,
        shard_info_df=shard_info_df,
        top_k=top_k,
    )

    del suspicious_embeddings
    cleanup_memory()

    source_records = source_metadata[
        ["chunk_id", "doc_id", "chunk_index", "start_char", "end_char"]
    ].to_dict("records")

    results = []

    for suspicious_i, suspicious_row in enumerate(suspicious_df.itertuples(index=False)):
        for rank in range(scores.shape[1]):
            source_idx = int(indices[suspicious_i, rank])
            score = float(scores[suspicious_i, rank])

            if source_idx < 0:
                continue

            source_row = source_records[source_idx]

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
# DOCUMENT-LEVEL MAX SCORE
# ============================================================

def get_top_source_documents_by_max_embedding_score(
    candidates_df: pd.DataFrame,
    source_chunks_path: Path,
    suspicious_doc_id: str,
    top_n: int = 50,
    min_match_count: int = 1,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Rank unique source documents by maximum embedding similarity score.

    This is useful when a source document contains one very strong matching passage,
    even if the rest of the retrieved chunks are weaker.
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
            max_embedding_score=("embedding_score", "max"),
            mean_embedding_score=("embedding_score", "mean"),
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
            ["max_embedding_score", "mean_embedding_score", "match_count"],
            ascending=[False, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    grouped_df["source_doc_rank"] = range(1, len(grouped_df) + 1)

    grouped_df = grouped_df[
        [
            "source_doc_rank",
            "source_doc_id",
            "max_embedding_score",
            "mean_embedding_score",
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

def run_embedding_lookup(doc_id: str, dataset: str = "pan2011", build_index: bool = False) -> dict:
    """Run one embedding lookup and write the standard parquet outputs."""
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")

    paths = DATASET_PATHS[dataset]
    PROCESSED_DIR = paths["processed_dir"]
    ARTIFACT_DIR = paths["artifact_dir"]

    SOURCE_CHUNKS_PATH = PROCESSED_DIR / "source_chunks_embeddings.parquet"
    SUSPICIOUS_CHUNKS_PATH = PROCESSED_DIR / "suspicious_chunks_embeddings.parquet"
    SOURCE_CANONICAL_CHUNKS_PATH = PROCESSED_DIR / "source_chunks.parquet"

    # Fixed filenames (not per-doc) — run_pipeline.py / source_retrieval_branches.py
    # always read these exact names from PROCESSED_DIR.
    OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / "embedding_candidates_suspicious.parquet"
    OUTPUT_TOP_DOCS_PATH = PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"

    SUSPICIOUS_DOC_ID = doc_id

    # ========================================================
    # OPTION A: Recommended
    # Build sharded index and search shards directly.
    # ========================================================

    BUILD_INDEX = build_index

    # ========================================================
    # OPTION B: Disabled by default
    # Only set this to True after shards already exist.
    # This tries to create one big faiss.index from all shards.
    # ========================================================

    MERGE_SHARDS_AFTER_BUILD = False

    if BUILD_INDEX:
        build_embedding_index(
            source_chunks_path=SOURCE_CHUNKS_PATH,
            artifact_dir=ARTIFACT_DIR,
            model_path=MODEL_PATH,
            batch_size=20,
            max_source_chunks=None,
            save_every_batches=2000,
        )

    if MERGE_SHARDS_AFTER_BUILD:
        merge_faiss_shards(
            artifact_dir=ARTIFACT_DIR,
            output_index_name="faiss.index",
        )

    candidates_df = retrieve_embedding_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        model_path=MODEL_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        top_k=500,
        batch_size=8,
        max_suspicious_chunks=None,
    )

    top_sources_df = get_top_source_documents_by_max_embedding_score(
        candidates_df=candidates_df,
        source_chunks_path=SOURCE_CANONICAL_CHUNKS_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        top_n=20,
        min_match_count=1,
        output_path=OUTPUT_TOP_DOCS_PATH,
    )

    return {
        "doc_id": doc_id,
        "dataset": dataset,
        "candidate_rows": int(len(candidates_df)),
        "top_source_rows": int(len(top_sources_df)),
        "candidates_path": str(OUTPUT_CANDIDATES_PATH),
        "top_sources_path": str(OUTPUT_TOP_DOCS_PATH),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", type=str, default="part1__suspicious-document00007.txt")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_PATHS.keys()), default="pan2011")
    parser.add_argument("--build-index", dest="build_index", action="store_true", default=False,
                         help="(Re)build the source index before querying. Default: skip (index already built).")
    args = parser.parse_args()

    summary = run_embedding_lookup(
        doc_id=args.doc_id,
        dataset=args.dataset,
        build_index=args.build_index,
    )
    print(summary)
