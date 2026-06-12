from pathlib import Path
from typing import Literal, Optional, Union
import joblib
import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import subprocess

# ============================================================
# Global Config
# ============================================================

# All paths are relative to scripts/final/ (the CWD set by the Streamlit app)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300"

SUSPICIOUS_CHUNKS_PATH = PROCESSED_DIR / "suspicious_chunks_lsa_esa.parquet"
SOURCE_CANONICAL_CHUNKS_PATH = PROCESSED_DIR / "source_chunks.parquet"


# Options:
# - "char": best for exact copy-paste and small edits
# - "word": good for word/phrase reuse
TFIDF_MODE: Literal["char", "word"] = "char"


def search_artifact(artifact_type):
    """Set global path variables for the chosen retrieval branch artifact directory."""
    global ARTIFACT_DIR
    global OUTPUT_CANDIDATES_PATH
    global OUTPUT_TOP_DOCS_MAX_PATH
    global OUTPUT_PATH
    global SUSPICIOUS_CHUNKS_PATH

    match artifact_type:
        case "tf-idf":
            ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_hashing"
            OUTPUT_TOP_DOCS_MAX_PATH = PROCESSED_DIR / "tfidf_top_source_documents_by_max_score.parquet"
            OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / "tfidf_candidates_suspicious_doc.parquet"
        case "esa":
            ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "esa"
            OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / "esa_candidates_suspicious_doc.parquet"
            OUTPUT_TOP_DOCS_MAX_PATH = PROCESSED_DIR / "esa_top_source_documents_by_max_score.parquet"
        case "lsa":
            ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "lsa"
            SUSPICIOUS_CHUNKS_PATH = PROCESSED_DIR / "suspicious_chunks_lsa_esa.parquet"
            OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / "lsa_candidates_suspicious_doc.parquet"
        case "emb":
            ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "embeddings"
            OUTPUT_PATH = PROCESSED_DIR / "emb_candidates_suspicious_doc.parquet"
        case _:
            raise "error here search artifact!"


def cleanup_memory() -> None:
    """Force a Python GC cycle to reclaim memory after large matrix operations."""
    gc.collect()

#obsolete for now
def load_embedding_results(doc_id: str = None) -> pd.DataFrame:
    embedding_top_path = PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"

    if not embedding_top_path.exists():
        raise FileNotFoundError(
            f"Missing embedding results:\n{embedding_top_path}\n\n"
            f"Run the ROCm Docker embedding lookup first for:\n{doc_id or SUSPICIOUS_DOC_ID}"
        )

    return pd.read_parquet(embedding_top_path)


def tf_idf_lookup():
    # ============================================================
    # LOAD CHUNKS
    # ============================================================

    def load_tfidf_chunks(
        path: Path,
        text_column: str = "lsa_esa_text",
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
    # TF-IDF TRANSFORM HELPERS
    # ============================================================ss
    def transform_texts_to_tfidf(
        texts: list[str],
        vectorizer: HashingVectorizer,
        idf: np.ndarray,
    ) -> sparse.csr_matrix:
        """
        Transform texts into L2-normalized hashed TF-IDF vectors.
        """

        counts = vectorizer.transform(texts).tocsr().astype(np.float32)

        # Apply sublinear TF: 1 + log(tf)
        if counts.nnz > 0:
            counts.data = 1.0 + np.log(counts.data)

        tfidf = counts.multiply(idf).tocsr().astype(np.float32)

        # L2 normalization makes dot product = cosine similarity.
        tfidf = normalize(tfidf, norm="l2", axis=1, copy=False)

        return tfidf.astype(np.float32)

    # ============================================================
    # LOAD SHARDED TF-IDF INDEX
    # ============================================================

    def load_tfidf_index(
        artifact_dir: Path,
        mode: Literal["char", "word"] = "char",
    ):
        artifact_dir = Path(artifact_dir) / mode

        vectorizer_path = artifact_dir / "tfidf_vectorizer.joblib"
        idf_path = artifact_dir / "tfidf_idf.npy"
        metadata_path = artifact_dir / "source_tfidf_metadata.parquet"
        shards_path = artifact_dir / "tfidf_shards.parquet"
        config_path = artifact_dir / "tfidf_config.joblib"

        for path in [vectorizer_path, idf_path, metadata_path, shards_path, config_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing TF-IDF artifact: {path}")

        print("Loading sharded TF-IDF artifacts...")

        vectorizer = joblib.load(vectorizer_path)
        idf = np.load(idf_path).astype(np.float32)
        source_metadata = pd.read_parquet(metadata_path)
        shard_info_df = pd.read_parquet(shards_path)
        shard_info_df["shard_path"] = shard_info_df["shard_path"].apply(
            lambda p: str((artifact_dir / p).resolve())
        )
        config = joblib.load(config_path)

        expected_rows = int(shard_info_df["num_rows"].sum())

        if len(source_metadata) != expected_rows:
            raise ValueError(
                f"Metadata/shard mismatch: metadata rows={len(source_metadata)}, "
                f"TF-IDF rows={expected_rows}"
            )

        print(f"Loaded source metadata rows: {len(source_metadata)}")
        print(f"Loaded TF-IDF shards: {len(shard_info_df)}")

        return vectorizer, idf, shard_info_df, source_metadata, config


    # ============================================================
    # SEARCH SHARDED TF-IDF
    # ============================================================

    def search_tfidf_shards(
        query_tfidf: sparse.csr_matrix,
        shard_info_df: pd.DataFrame,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search all TF-IDF shards and return global top-k source rows per query row.

        Returns:
        - final_scores: shape (num_queries, top_k)
        - final_indices: global source metadata row indices, shape (num_queries, top_k)
        """

        num_queries = query_tfidf.shape[0]

        per_query_scores = [[] for _ in range(num_queries)]
        per_query_indices = [[] for _ in range(num_queries)]

        for shard_row in tqdm(
            shard_info_df.itertuples(index=False),
            total=len(shard_info_df),
            desc="Searching TF-IDF shards",
        ):
            shard_path = Path(shard_row.shard_path)
            shard_start_row = int(shard_row.start_row)

            if not shard_path.exists():
                raise FileNotFoundError(f"Missing TF-IDF shard: {shard_path}")

            source_tfidf_shard = sparse.load_npz(shard_path).tocsr()

            # Sparse cosine similarity.
            # Since both matrices are L2-normalized, dot product = cosine similarity.
            similarities = query_tfidf @ source_tfidf_shard.T

            for query_i in range(num_queries):
                sims_sparse = similarities.getrow(query_i)

                if sims_sparse.nnz == 0:
                    continue

                candidate_indices = sims_sparse.indices
                candidate_scores = sims_sparse.data

                safe_top_k = min(top_k, len(candidate_scores))

                top_local_indices = np.argpartition(
                    -candidate_scores,
                    safe_top_k - 1,
                )[:safe_top_k]

                top_local_indices = top_local_indices[
                    np.argsort(-candidate_scores[top_local_indices])
                ]

                global_indices = candidate_indices[top_local_indices] + shard_start_row
                top_scores = candidate_scores[top_local_indices]

                per_query_indices[query_i].extend(global_indices.tolist())
                per_query_scores[query_i].extend(top_scores.tolist())

            del source_tfidf_shard
            del similarities
            cleanup_memory()

        final_scores = np.full((num_queries, top_k), -np.inf, dtype=np.float32)
        final_indices = np.full((num_queries, top_k), -1, dtype=np.int64)

        for query_i in range(num_queries):
            scores = np.asarray(per_query_scores[query_i], dtype=np.float32)
            indices = np.asarray(per_query_indices[query_i], dtype=np.int64)

            if len(scores) == 0:
                continue

            safe_top_k = min(top_k, len(scores))

            top_positions = np.argpartition(
                -scores,
                safe_top_k - 1,
            )[:safe_top_k]

            top_positions = top_positions[np.argsort(-scores[top_positions])]

            final_scores[query_i, :safe_top_k] = scores[top_positions]
            final_indices[query_i, :safe_top_k] = indices[top_positions]

        return final_scores, final_indices


    # ============================================================
    # QUERY ONE SUSPICIOUS DOCUMENT
    # ============================================================

    def retrieve_tfidf_candidates_for_suspicious_doc(
        suspicious_chunks_path: Path,
        artifact_dir: Path,
        suspicious_doc_id: str,
        output_path: Path,
        text_column: str = "lsa_esa_text",
        mode: Literal["char", "word"] = "char",
        top_k: int = 100,
        batch_size: int = 8,
        max_suspicious_chunks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query the sharded hashed TF-IDF source index using one selected suspicious document.

        Returns top-k source chunks for each suspicious chunk.
        """

        vectorizer, idf, shard_info_df, source_metadata, config = load_tfidf_index(
            artifact_dir=artifact_dir,
            mode=mode,
        )

        suspicious_df = load_tfidf_chunks(
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
        print(f"TF-IDF mode: {mode}")

        results = []

        for start in range(0, len(suspicious_df), batch_size):
            end = min(start + batch_size, len(suspicious_df))
            batch_df = suspicious_df.iloc[start:end]

            batch_texts = batch_df[text_column].tolist()

            suspicious_tfidf = transform_texts_to_tfidf(
                texts=batch_texts,
                vectorizer=vectorizer,
                idf=idf,
            )

            scores, indices = search_tfidf_shards(
                query_tfidf=suspicious_tfidf,
                shard_info_df=shard_info_df,
                top_k=top_k,
            )

            print(f"Processing suspicious chunks {start} to {end}")

            for local_i, suspicious_row in enumerate(batch_df.itertuples(index=False)):
                for rank in range(top_k):
                    source_idx = int(indices[local_i, rank])
                    score = float(scores[local_i, rank])

                    if source_idx < 0 or not np.isfinite(score):
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

                        "TFIDF_score": score,
                        "TFIDF_rank": rank + 1,
                        "TFIDF_mode": mode,
                        "TFIDF_vectorizer": "hashing",
                    })

            del suspicious_tfidf
            del scores
            del indices
            cleanup_memory()

            print(f"Processed suspicious chunks {start} to {end}")

        output_df = pd.DataFrame(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path = output_path.with_name(
            output_path.stem + f"_{mode}_hashing" + output_path.suffix
        )

        output_df.to_parquet(output_path, index=False)

        print(f"Saved TF-IDF candidates to: {output_path}")
        print(f"Candidate rows: {len(output_df)}")

        return output_df


    # ============================================================
    # DOCUMENT-LEVEL MAX SCORE
    # ============================================================

    def get_top_source_documents_by_max_tfidf_score(
        candidates_df: pd.DataFrame,
        source_chunks_path: Path,
        suspicious_doc_id: str,
        top_n: int = 20,
        min_match_count: int = 1,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Rank unique source documents by strongest TF-IDF chunk match.

        Best for copy-paste retrieval because plagiarism is often local.
        """

        required_columns = {
            "suspicious_doc_id",
            "suspicious_chunk_id",
            "source_doc_id",
            "source_chunk_id",
            "TFIDF_score",
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
                max_TFIDF_score=("TFIDF_score", "max"),
                mean_TFIDF_score=("TFIDF_score", "mean"),
                min_TFIDF_score=("TFIDF_score", "min"),
                match_count=("TFIDF_score", "count"),
                unique_source_chunks=("source_chunk_id", "nunique"),
                unique_suspicious_chunks=("suspicious_chunk_id", "nunique"),
            )
            .reset_index()
        )

        grouped_df = grouped_df[grouped_df["match_count"] >= min_match_count].copy()

        grouped_df = (
            grouped_df
            .sort_values(
                [
                    "max_TFIDF_score",
                    "unique_suspicious_chunks",
                    "match_count",
                    "mean_TFIDF_score",
                ],
                ascending=[False, False, False, False],
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        grouped_df["source_doc_rank"] = range(1, len(grouped_df) + 1)

        grouped_df = grouped_df[
            [
                "source_doc_rank",
                "source_doc_id",
                "max_TFIDF_score",
                "mean_TFIDF_score",
                "min_TFIDF_score",
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

            mode = candidates_df["TFIDF_mode"].iloc[0] if "TFIDF_mode" in candidates_df.columns else "unknown"
            output_path = output_path.with_name(
                output_path.stem + f"_{mode}_hashing" + output_path.suffix
            )

            grouped_df.to_parquet(output_path, index=False)
            print(f"Saved top source documents to: {output_path}")

        return grouped_df

    candidates_df = retrieve_tfidf_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        text_column="lsa_esa_text",
        mode=TFIDF_MODE,
        top_k=500,
        batch_size=8,
        max_suspicious_chunks=None,
    )

    top_sources_max_df = get_top_source_documents_by_max_tfidf_score(
        candidates_df=candidates_df,
        source_chunks_path=SOURCE_CANONICAL_CHUNKS_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        top_n=50,
        min_match_count=1,
        output_path=OUTPUT_TOP_DOCS_MAX_PATH,
    )

    return top_sources_max_df

def lsa_lookup():
    # ============================================================
    # LOAD CHUNKS
    # ============================================================

    def load_lsa_esa_chunks(path: Path, text_column: str = "lsa_esa_text",) -> pd.DataFrame:
        
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
    # LOAD LSA INDEX
    # ============================================================

    def load_lsa_index(artifact_dir: Path):
        artifact_dir = Path(artifact_dir)

        vectorizer_path = artifact_dir / "tfidf_vectorizer.joblib"
        svd_path = artifact_dir / "svd_model.joblib"
        vectors_path = artifact_dir / "source_lsa_vectors.npy"
        metadata_path = artifact_dir / "source_lsa_metadata.parquet"

        for path in [vectorizer_path, svd_path, vectors_path, metadata_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing LSA artifact: {path}")

        print("Loading LSA artifacts...")

        vectorizer = joblib.load(vectorizer_path)
        svd = joblib.load(svd_path)
        source_lsa = np.load(vectors_path)
        source_metadata = pd.read_parquet(metadata_path)

        return vectorizer, svd, source_lsa, source_metadata


    # ============================================================
    # QUERY ONE SUSPICIOUS DOCUMENT
    # ============================================================

    def retrieve_lsa_candidates_for_suspicious_doc(
        suspicious_chunks_path: Path,
        artifact_dir: Path,
        suspicious_doc_id: str,
        output_path: Path,
        text_column: str = "lsa_esa_text",
        top_k: int = 20,
        batch_size: int = 8,
        max_suspicious_chunks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query the LSA source index using only one selected suspicious document.

        Returns top-k source chunks for each suspicious chunk.
        """

        vectorizer, svd, source_lsa, source_metadata = load_lsa_index(artifact_dir)

        suspicious_df = load_lsa_esa_chunks(
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

        results = []

        for start in range(0, len(suspicious_df), batch_size):
            end = min(start + batch_size, len(suspicious_df))
            batch_df = suspicious_df.iloc[start:end]

            batch_texts = batch_df[text_column].tolist()

            suspicious_tfidf = vectorizer.transform(batch_texts)
            suspicious_lsa = svd.transform(suspicious_tfidf)
            suspicious_lsa = normalize(suspicious_lsa, norm="l2", axis=1)

            similarities = suspicious_lsa @ source_lsa.T

            for local_i, suspicious_row in enumerate(batch_df.itertuples(index=False)):
                sims = similarities[local_i]

                safe_top_k = min(top_k, len(sims))
                top_indices = np.argpartition(-sims, safe_top_k - 1)[:safe_top_k]
                top_indices = top_indices[np.argsort(-sims[top_indices])]

                for rank, source_idx in enumerate(top_indices, start=1):
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

                        "LSA_score": float(sims[source_idx]),
                        "LSA_rank": rank,
                    })

            print(f"Processed suspicious chunks {start} to {end}")

        output_df = pd.DataFrame(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, index=False)

        print(f"Saved LSA candidates to: {output_path}")

        return output_df


    # ============================================================
    # DOCUMENT-LEVEL Max SCORE
    # ============================================================

    def get_top_source_documents_by_max_score(
        candidates_df: pd.DataFrame,
        source_chunks_path: Path,
        suspicious_doc_id: str,
        top_n: int = 20,
        min_match_count: int = 1,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Rank unique source documents by strongest LSA chunk match.

        This is better for plagiarism retrieval because plagiarism is often local:
        one strong matching passage can identify the true source.
        """

        required_columns = {
            "suspicious_doc_id",
            "suspicious_chunk_id",
            "source_doc_id",
            "source_chunk_id",
            "LSA_score",
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
                mean_LSA_score=("LSA_score", "mean"),
                max_LSA_score=("LSA_score", "max"),
                min_LSA_score=("LSA_score", "min"),
                match_count=("LSA_score", "count"),
                unique_source_chunks=("source_chunk_id", "nunique"),
                unique_suspicious_chunks=("suspicious_chunk_id", "nunique"),
            )
            .reset_index()
        )

        grouped_df = grouped_df[grouped_df["match_count"] >= min_match_count].copy()

        grouped_df = (
            grouped_df
            .sort_values(
                ["max_LSA_score", "match_count", "mean_LSA_score"],
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
                "mean_LSA_score",
                "max_LSA_score",
                "min_LSA_score",
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

    candidates_df = retrieve_lsa_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        top_k=500,              
        batch_size=8,
        max_suspicious_chunks=None,
    )

    top_sources_df = get_top_source_documents_by_max_score(
        candidates_df=candidates_df,
        source_chunks_path=PROCESSED_DIR / "source_chunks.parquet",
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        top_n=50,              
        min_match_count=1,     
        output_path=PROCESSED_DIR / "lsa_top_source_documents_by_max_score.parquet",
    )

    return top_sources_df

def esa_lookup():
    # ============================================================
    # LOAD CHUNKS
    # ============================================================

    def load_lsa_esa_chunks(
        path: Path,
        text_column: str = "lsa_esa_text",
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
    # LOAD ESA INDEX
    # ============================================================

    def load_esa_index(artifact_dir: Path):
        artifact_dir = Path(artifact_dir)

        vectorizer_path = artifact_dir / "esa_tfidf_vectorizer.joblib"
        vectors_path = artifact_dir / "source_esa_vectors.npz"
        metadata_path = artifact_dir / "source_esa_metadata.parquet"
        config_path = artifact_dir / "esa_config.joblib"

        for path in [vectorizer_path, vectors_path, metadata_path, config_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing ESA artifact: {path}")

        print("Loading ESA artifacts...")

        vectorizer = joblib.load(vectorizer_path)
        source_esa = sparse.load_npz(vectors_path)
        source_metadata = pd.read_parquet(metadata_path)
        config = joblib.load(config_path)

        return vectorizer, source_esa, source_metadata, config


    # ============================================================
    # QUERY ONE SUSPICIOUS DOCUMENT
    # ============================================================

    def retrieve_esa_candidates_for_suspicious_doc(
        suspicious_chunks_path: Path,
        artifact_dir: Path,
        suspicious_doc_id: str,
        output_path: Path,
        text_column: str = "lsa_esa_text",
        top_k: int = 20,
        batch_size: int = 8,
        max_suspicious_chunks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query the ESA source index using one selected suspicious document.

        Returns top-k source chunks for each suspicious chunk.
        """

        vectorizer, source_esa, source_metadata, config = load_esa_index(artifact_dir)

        suspicious_df = load_lsa_esa_chunks(
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

        results = []

        for start in range(0, len(suspicious_df), batch_size):
            end = min(start + batch_size, len(suspicious_df))
            batch_df = suspicious_df.iloc[start:end]

            batch_texts = batch_df[text_column].tolist()

            suspicious_esa = vectorizer.transform(batch_texts)
            suspicious_esa = normalize(suspicious_esa, norm="l2", axis=1)

            # Sparse cosine similarity:
            # because both matrices are L2-normalized, dot product = cosine similarity.
            similarities = suspicious_esa @ source_esa.T

            for local_i, suspicious_row in enumerate(batch_df.itertuples(index=False)):
                sims_sparse = similarities.getrow(local_i)

                if sims_sparse.nnz == 0:
                    continue

                candidate_indices = sims_sparse.indices
                candidate_scores = sims_sparse.data

                safe_top_k = min(top_k, len(candidate_scores))

                top_local_indices = np.argpartition(
                    -candidate_scores,
                    safe_top_k - 1,
                )[:safe_top_k]

                top_local_indices = top_local_indices[
                    np.argsort(-candidate_scores[top_local_indices])
                ]

                for rank, local_idx in enumerate(top_local_indices, start=1):
                    source_idx = int(candidate_indices[local_idx])
                    score = float(candidate_scores[local_idx])

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

                        "ESA_score": score,
                        "ESA_rank": rank,
                    })

            print(f"Processed suspicious chunks {start} to {end}")

        output_df = pd.DataFrame(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, index=False)

        print(f"Saved ESA candidates to: {output_path}")

        return output_df

    # ============================================================
    # DOCUMENT-LEVEL MAX SCORE
    # ============================================================

    def get_top_source_documents_by_max_esa_score(
        candidates_df: pd.DataFrame,
        source_chunks_path: Path,
        suspicious_doc_id: str,
        top_n: int = 20,
        min_match_count: int = 1,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Rank unique source documents by strongest ESA chunk match.

        This is usually better for plagiarism source retrieval because plagiarism
        is often local: one strong passage-level match can identify the true source.
        """

        required_columns = {
            "suspicious_doc_id",
            "suspicious_chunk_id",
            "source_doc_id",
            "source_chunk_id",
            "ESA_score",
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
                mean_ESA_score=("ESA_score", "mean"),
                max_ESA_score=("ESA_score", "max"),
                min_ESA_score=("ESA_score", "min"),
                match_count=("ESA_score", "count"),
                unique_source_chunks=("source_chunk_id", "nunique"),
                unique_suspicious_chunks=("suspicious_chunk_id", "nunique"),
            )
            .reset_index()
        )

        grouped_df = grouped_df[grouped_df["match_count"] >= min_match_count].copy()

        grouped_df = (
            grouped_df
            .sort_values(
                ["max_ESA_score", "match_count", "mean_ESA_score"],
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
                "mean_ESA_score",
                "max_ESA_score",
                "min_ESA_score",
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

    candidates_df = retrieve_esa_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        text_column="lsa_esa_text",
        top_k=500,
        batch_size=8,
        max_suspicious_chunks=None,
    )

    top_sources_max_df = get_top_source_documents_by_max_esa_score(
        candidates_df=candidates_df,
        source_chunks_path=SOURCE_CANONICAL_CHUNKS_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        top_n=50,
        min_match_count=1,
        output_path=OUTPUT_TOP_DOCS_MAX_PATH,
    )

    return top_sources_max_df

#obsolete for now
def embeddings_lookup(suspicious_doc_id:str):


    # ============================================================
    # CONFIG
    # ============================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "PAN2011_300"
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "embeddings" / "embeddings_qwen06b"

    SUSPICIOUS_CHUNKS_PATH = PROCESSED_DIR / "suspicious_chunks_embeddings.parquet"
    SOURCE_CANONICAL_CHUNKS_PATH = PROCESSED_DIR / "source_chunks.parquet"

    MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "Qwen3-Embedding-0.6B"

    SUSPICIOUS_DOC_ID = suspicious_doc_id

    safe_doc_id = suspicious_doc_id.replace("/", "__").replace(".txt", "")
    #safe_doc_id=suspicious_doc_id

    OUTPUT_CANDIDATES_PATH = PROCESSED_DIR / f"embedding_candidates_{safe_doc_id}.parquet"
    OUTPUT_TOP_DOCS_PATH = PROCESSED_DIR / f"embedding_top_source_documents_{safe_doc_id}.parquet"
    
    # ============================================================
    # LOAD CHUNKS
    # ============================================================

    def load_embedding_chunks(path: Path,text_column: str = "embedding_text",) -> pd.DataFrame:
        
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
        print("torch cuda available:", torch.cuda.is_available())
        print("torch device count:", torch.cuda.device_count())
        print("torch num threads:", torch.get_num_threads())

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SentenceTransformer(str(model_path), device=device)

        # Important: do not let the model try to support huge context for this task.
        model.max_seq_length = 512

        print("SentenceTransformer device:", model.device)
        print("Model max_seq_length:", model.max_seq_length)

        return model
    
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
        top_k: int = 20,
        batch_size: int = 8,
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

        return output_df


    # ============================================================
    # DOCUMENT-LEVEL MEAN SCORE
    # ============================================================

    def get_top_source_documents_by_max_embedding_score(
        candidates_df: pd.DataFrame,
        source_chunks_path: Path,
        suspicious_doc_id: str,
        top_n: int = 20,
        min_match_count: int = 4,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Rank unique source documents by mean embedding score.

        If the same source document appears multiple times because of different
        source chunks, this computes the mean score for that source document.
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

        grouped_df = grouped_df[grouped_df["match_count"] >= min_match_count].copy()

        grouped_df = (
            grouped_df
            .sort_values(["max_embedding_score", "match_count"], ascending=[False, False])
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

    candidates_df = retrieve_embedding_candidates_for_suspicious_doc(
        suspicious_chunks_path=SUSPICIOUS_CHUNKS_PATH,
        artifact_dir=ARTIFACT_DIR,
        model_path=MODEL_PATH,
        suspicious_doc_id=SUSPICIOUS_DOC_ID,
        output_path=OUTPUT_CANDIDATES_PATH,
        top_k=500,
        batch_size=24,
        max_suspicious_chunks=None,
    )

    top_sources_df = get_top_source_documents_by_max_embedding_score(
    candidates_df=candidates_df,
    source_chunks_path=SOURCE_CANONICAL_CHUNKS_PATH,
    suspicious_doc_id=SUSPICIOUS_DOC_ID,
    top_n=50,
    min_match_count=1,
    output_path=OUTPUT_TOP_DOCS_PATH,
    )

    return top_sources_df

def embedding_run(suspicious_doc_id: str):
    """Trigger the embedding lookup inside the ROCm Docker container for one suspicious doc."""
    result = subprocess.run(
        ["docker", "exec", "docplag-rocm", "python", "scripts/embeddings.py", "--doc_id", suspicious_doc_id],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result



def mean_doc_score_aggreg(
    top_tf_idf: Union[pd.DataFrame, str, Path],
    top_esa: Union[pd.DataFrame, str, Path],
    top_lsa: Union[pd.DataFrame, str, Path],
    top_emb: Union[pd.DataFrame, str, Path],
    weights: Optional[dict] = None,
    top_n_per_branch: int = 20,
    final_top_n: int = 50,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    # Default weights were tuned empirically; embeddings carry the most signal
    """
    Fuse already-aggregated top-50 source-document results from:
    - TF-IDF
    - ESA
    - LSA
    - Embeddings

    Each input should already be document-level aggregated by max score.

    Expected possible score columns:
    TF-IDF:
        mean_TFIDF_score, max_TFIDF_score
    ESA:
        mean_ESA_score, max_ESA_score
    LSA:
        mean_LSA_score, max_LSA_score
    Embeddings:
        mean_embedding_score, max_embedding_score

    Logic:
    1. For each branch, take top N by mean score from its top-50 max list.
    2. Merge all source documents.
    3. Fill missing branch scores with 0.
    4. Compute weighted fusion score.
    """

    def load_df(x):
        if isinstance(x, pd.DataFrame):
            return x.copy()
        return pd.read_parquet(x)

    def prep_branch(
        df: pd.DataFrame,
        branch_name: str,
        mean_col: str,
        max_col: str,
    ) -> pd.DataFrame:
        required = {"source_doc_id", mean_col, max_col}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"{branch_name}: missing columns: {missing}")

        df = df.copy()

        df = (
            df.sort_values(
                [max_col, mean_col],
                ascending=[False, False],
            )
            .head(top_n_per_branch)
            .copy()
        )

        df[f"{branch_name}_mean_rank"] = range(1, len(df) + 1)

        keep_cols = [
            "source_doc_id",
            mean_col,
            max_col,
            f"{branch_name}_mean_rank",
        ]

        optional_cols = [
            "match_count",
            "unique_source_chunks",
            "unique_suspicious_chunks",
            "source_relative_path",
        ]

        keep_cols += [col for col in optional_cols if col in df.columns]

        df = df[keep_cols].copy()

        rename_map = {
            mean_col: f"{branch_name}_mean_score",
            max_col: f"{branch_name}_max_score",
            "match_count": f"{branch_name}_match_count",
            "unique_source_chunks": f"{branch_name}_unique_source_chunks",
            "unique_suspicious_chunks": f"{branch_name}_unique_suspicious_chunks",
            "source_relative_path": f"{branch_name}_source_relative_path",
        }

        df = df.rename(columns=rename_map)

        return df

    if weights is None:
        weights = {
            "tfidf": 0.10,
            "esa":   0.15,
            "lsa":   0.25,
            "emb":   0.50,
        }

    if top_tf_idf is None:
        weights = {k: v for k, v in weights.items() if k != "tfidf"}

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("All weights are zero")
    weights = {k: v / weight_sum for k, v in weights.items()}

    tfidf_df = (
        prep_branch(
            load_df(top_tf_idf),
            branch_name="tfidf",
            mean_col="mean_TFIDF_score",
            max_col="max_TFIDF_score",
        )
        if top_tf_idf is not None
        else None
    )

    esa_df = prep_branch(
        load_df(top_esa),
        branch_name="esa",
        mean_col="mean_ESA_score",
        max_col="max_ESA_score",
    )

    lsa_df = prep_branch(
        load_df(top_lsa),
        branch_name="lsa",
        mean_col="mean_LSA_score",
        max_col="max_LSA_score",
    )

    emb_df = prep_branch(
        load_df(top_emb),
        branch_name="emb",
        mean_col="mean_embedding_score",
        max_col="max_embedding_score",
    )

    branch_dfs = [b for b in [tfidf_df, esa_df, lsa_df, emb_df] if b is not None]
    fused_df = branch_dfs[0]
    for b in branch_dfs[1:]:
        fused_df = fused_df.merge(b, on="source_doc_id", how="outer")

    fused_df["found_by_tfidf"] = fused_df["tfidf_max_score"].notna() if tfidf_df is not None else False
    fused_df["found_by_esa"]   = fused_df["esa_max_score"].notna()
    fused_df["found_by_lsa"]   = fused_df["lsa_max_score"].notna()
    fused_df["found_by_emb"]   = fused_df["emb_max_score"].notna()

    fused_df["methods_found_count"] = fused_df[
        ["found_by_tfidf", "found_by_esa", "found_by_lsa", "found_by_emb"]
    ].sum(axis=1).astype(int)

    score_cols = [
        "esa_mean_score", "lsa_mean_score", "emb_mean_score",
        "esa_max_score",  "lsa_max_score",  "emb_max_score",
    ]
    if tfidf_df is not None:
        score_cols += ["tfidf_mean_score", "tfidf_max_score"]
    else:
        fused_df["tfidf_mean_score"] = 0.0
        fused_df["tfidf_max_score"]  = 0.0
    for col in score_cols:
        fused_df[col] = fused_df[col].fillna(0.0)

    tfidf_weight = weights.get("tfidf", 0.0)
    fused_df["weighted_mean_score"] = (
        tfidf_weight * fused_df["tfidf_mean_score"]
        + weights["esa"] * fused_df["esa_mean_score"]
        + weights["lsa"] * fused_df["lsa_mean_score"]
        + weights["emb"] * fused_df["emb_mean_score"]
    )

    fused_df["weighted_max_score"] = (
        tfidf_weight * fused_df["tfidf_max_score"]
        + weights["esa"] * fused_df["esa_max_score"]
        + weights["lsa"] * fused_df["lsa_max_score"]
        + weights["emb"] * fused_df["emb_max_score"]
    )

    # max score weighted more heavily (0.80) because one strong local match
    # is a better signal for plagiarism than a high average across all chunks
    fused_df["final_score"] = (
        0.20 * fused_df["weighted_mean_score"]
        + 0.80 * fused_df["weighted_max_score"]
        #+ 0.10 * (fused_df["methods_found_count"] / 4) removed as this penalized my final score in this part as recall is the most imporant in source retrieval
    )

    fused_df = (
        fused_df.sort_values(
            ["final_score", "methods_found_count", "weighted_mean_score", "weighted_max_score"],
            ascending=[False, False, False, False],
        )
        .head(final_top_n)
        .reset_index(drop=True)
    )

    fused_df["final_rank"] = range(1, len(fused_df) + 1)

    first_cols = [
        "final_rank",
        "source_doc_id",
        "final_score",
        "weighted_mean_score",
        "weighted_max_score",
        "methods_found_count",
        "found_by_tfidf",
        "found_by_esa",
        "found_by_lsa",
        "found_by_emb",
        "tfidf_mean_score",
        "esa_mean_score",
        "lsa_mean_score",
        "emb_mean_score",
        "tfidf_max_score",
        "esa_max_score",
        "lsa_max_score",
        "emb_max_score",
    ]

    remaining_cols = [col for col in fused_df.columns if col not in first_cols]
    fused_df = fused_df[first_cols + remaining_cols]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fused_df.to_parquet(output_path, index=False)
        print(f"Saved fused results to: {output_path}")

    return fused_df

def lookup_pipeline(suspicious_doc_id: str, run_embeddings: bool = False, run_tfidf: bool = True, top_n=20, min_top1_score: float = 0.60, relative_gap: float = 0.70):
    """
    Full source-retrieval pipeline for one suspicious document.

    Runs ESA and LSA lookups sequentially (CPU), optionally TF-IDF (slow),
    then optionally triggers the GPU embedding lookup via Docker, and finally
    fuses all branch scores into a ranked list of candidate source documents.
    """
    global SUSPICIOUS_DOC_ID
    SUSPICIOUS_DOC_ID = suspicious_doc_id

    tf_df = None
    methods = ["tf-idf", "esa", "lsa"] if run_tfidf else ["esa", "lsa"]

    for method in methods:
        search_artifact(method)
        match method:
            case "tf-idf":
                print("Working on TF-IDF...")
                tf_df = tf_idf_lookup()
            case "esa":
                print("Working on ESA...")
                esa_df = esa_lookup()
            case "lsa":
                print("Working on LSA...")
                lsa_df = lsa_lookup()
            case _:
                raise "error here!"

    if run_embeddings:
        print("Working on embeddings with GPU...")
        result = embedding_run(suspicious_doc_id)
    
        try:
            if result.returncode != 0:
                raise RuntimeError(f"Script failed:\n{result.stderr}")

            emb_df = pd.read_parquet(
                PROCESSED_DIR / "embedding_top_source_documents_by_max_score.parquet"
            )

        except FileNotFoundError as e:
            print(f"[File Error] {e}")
            raise e
        except RuntimeError as e:
            print(f"[Runtime Error] {e}")
            raise e
        except Exception as e:
            print(f"[Unexpected Error] {e}")
            raise e

    else:
        print("Loading existing embedding results...")
        emb_df = load_embedding_results(suspicious_doc_id)

    def _top_score(df, *cols):
        for c in cols:
            if c in df.columns:
                return float(df[c].iloc[0])
        return 0.0

    branch_top1 = {
        "_branch_lsa_top1":    lsa_df["source_doc_id"].iloc[0],
        "_branch_lsa_score":   _top_score(lsa_df, "max_LSA_score", "lsa_max_score"),
        "_branch_esa_top1":    esa_df["source_doc_id"].iloc[0],
        "_branch_esa_score":   _top_score(esa_df, "max_ESA_score", "esa_max_score"),
        "_branch_emb_top1":    emb_df["source_doc_id"].iloc[0],
        "_branch_emb_score":   _top_score(emb_df, "max_embedding_score", "emb_max_score", "embedding_max_score"),
        "_branch_tfidf_top1":  tf_df["source_doc_id"].iloc[0] if tf_df is not None else "",
        "_branch_tfidf_score": _top_score(tf_df, "max_tfidf_score", "tfidf_max_score") if tf_df is not None else 0.0,
    }

    if tf_df is not None:
        print("TF-IDF:", tf_df["source_doc_id"].iloc[0])
    print("LSA:", lsa_df["source_doc_id"].iloc[0])
    print("ESA:", esa_df["source_doc_id"].iloc[0])
    print("EMB:", emb_df["source_doc_id"].iloc[0])

    mean_doc_df = mean_doc_score_aggreg(top_tf_idf=tf_df, top_esa=esa_df, top_lsa=lsa_df, top_emb=emb_df, final_top_n=top_n)

    top1_score = mean_doc_df["final_score"].iloc[0]

    # Gate 1 — absolute floor: if the best fusion candidate isn't credible, treat as clean
    branch_scores = [
        branch_top1["_branch_lsa_score"],
        branch_top1["_branch_esa_score"],
        branch_top1["_branch_emb_score"],
    ]
    if tf_df is not None:
        branch_scores.append(branch_top1["_branch_tfidf_score"])
    best_branch_score = max(branch_scores)

    if top1_score < min_top1_score:
        print(f"  Gate 1 FAILED (fusion={top1_score:.4f} < {min_top1_score}, best_branch={best_branch_score:.4f}) — no credible source found")
        return pd.DataFrame({"_top1_score": [top1_score], **{k: [v] for k, v in branch_top1.items()}})

    # Gate 2 — relative gap: keep only candidates close to the top-1
    min_score = top1_score * relative_gap
    filtered = mean_doc_df[mean_doc_df["final_score"] >= min_score].reset_index(drop=True)
    print(f"  Gate 1 passed (fusion={top1_score:.4f}, best_branch={best_branch_score:.4f}, threshold={min_top1_score})")
    print(f"  Gate 2 relative gap (top1 * {relative_gap} = {min_score:.4f}): {len(mean_doc_df)} → {len(filtered)} candidates")

    # Branch union: guarantee each branch's top-3 reaches the LLM even if fusion buried them.
    # This prevents cases where one branch (e.g. embeddings) finds the correct source at rank 1
    # but the other branches score it low, causing the fusion score to drop it out of the
    # relative gap window.
    BRANCH_TOP_K = 3
    branch_top_ids = {}  # doc_id -> list of branch names that contributed it
    for branch_name, branch_df in [("LSA", lsa_df), ("ESA", esa_df), ("EMB", emb_df)] + ([("TFIDF", tf_df)] if tf_df is not None else []):
        if branch_df is not None and not branch_df.empty:
            for doc_id in branch_df["source_doc_id"].iloc[:BRANCH_TOP_K].tolist():
                branch_top_ids.setdefault(doc_id, []).append(branch_name)

    already_included = set(filtered["source_doc_id"].tolist())
    extra_ids = set(branch_top_ids.keys()) - already_included
    if extra_ids:
        extra_rows = mean_doc_df[mean_doc_df["source_doc_id"].isin(extra_ids)].copy()
        filtered = pd.concat([filtered, extra_rows], ignore_index=True)
        filtered = filtered.sort_values("final_score", ascending=False).reset_index(drop=True)
        added = {doc: branch_top_ids[doc] for doc in extra_rows["source_doc_id"].tolist()}
        added_str = ", ".join(f"{doc} ({'+'.join(branches)})" for doc, branches in sorted(added.items()))
        print(f"  Branch union added {len(extra_rows)} candidate(s) not in Gate 2 window: [{added_str}]")

    # Attach branch top-1 info as metadata columns for downstream logging
    for k, v in branch_top1.items():
        filtered[k] = v
    return filtered

    
if __name__ == "__main__":
    result_df = lookup_pipeline(
        "part1__suspicious-document00007.txt",
        run_embeddings=True,
        run_tfidf=False,
        top_n=20,
        min_final_score=0.70
    )

    #print(result_df.to_string(index=False))


    result_df.to_parquet("top20_df.parquet",index=False) # top 20 of most likely sources
