from typing import Dict
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


from preprocessing import clean_pan_source_text, chunk_clean_text


class LSABranch:
    """
    Branch A:
    Word chunks -> TF-IDF -> TruncatedSVD -> LSA vectors -> cosine similarity
    """

    def __init__(
        self,
        chunk_size: int = 180,
        overlap: int = 50,
        min_words: int = 40,
        n_components: int = 200,
        top_k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.n_components = n_components
        self.top_k = top_k

        self.vectorizer = None
        self.lsa_model = None
        self.source_vectors = None
        self.source_metadata = None

    def build_index(self, source_docs: Dict[str, str]) -> pd.DataFrame:
        """
        Build LSA index from source documents.

        source_docs format:
        {
            "source-document00001.txt": "raw text...",
            "source-document00002.txt": "raw text..."
        }
        """

        rows = []
        source_chunks_text = []

        for source_doc_id, raw_text in source_docs.items():
            clean_text = light_clean_source_text(raw_text)

            chunks = chunk_clean_text(
                clean_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_words=self.min_words,
            )

            for chunk_id, chunk in enumerate(chunks):
                source_chunks_text.append(chunk["chunk_text"])

                rows.append({
                    "source_doc": source_doc_id,
                    "source_chunk_id": chunk_id,
                    "source_start_char": chunk["start_char"],
                    "source_end_char": chunk["end_char"],
                    "source_text": chunk["chunk_text"],
                })

        if not source_chunks_text:
            raise ValueError("No source chunks created for LSA.")

        self.source_metadata = pd.DataFrame(rows)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            max_features=100_000,
        )

        tfidf_matrix = self.vectorizer.fit_transform(source_chunks_text)

        safe_components = min(
            self.n_components,
            tfidf_matrix.shape[0] - 1,
            tfidf_matrix.shape[1] - 1,
        )

        safe_components = max(2, safe_components)

        self.lsa_model = make_pipeline(
            TruncatedSVD(n_components=safe_components, random_state=42),
            Normalizer(copy=False),
        )

        self.source_vectors = self.lsa_model.fit_transform(tfidf_matrix)

        return self.source_metadata

    def retrieve(
        self,
        suspicious_doc_id: str,
        suspicious_text: str,
        threshold: float = 0.30,
    ) -> pd.DataFrame:
        """
        Retrieve candidate source chunks for one suspicious document.
        """

        if self.vectorizer is None or self.lsa_model is None:
            raise RuntimeError("LSA index has not been built yet.")

        clean_text = light_clean_source_text(suspicious_text)

        suspicious_chunks = chunk_clean_text(
            clean_text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            min_words=self.min_words,
        )

        if not suspicious_chunks:
            return pd.DataFrame()

        suspicious_chunk_texts = [chunk["chunk_text"] for chunk in suspicious_chunks]

        suspicious_tfidf = self.vectorizer.transform(suspicious_chunk_texts)
        suspicious_vectors = self.lsa_model.transform(suspicious_tfidf)

        similarity_matrix = cosine_similarity(suspicious_vectors, self.source_vectors)

        results = []

        for suspicious_chunk_id, similarities in enumerate(similarity_matrix):
            best_source_indices = np.argsort(similarities)[::-1][:self.top_k]

            for source_index in best_source_indices:
                score = float(similarities[source_index])

                if score < threshold:
                    continue

                source_row = self.source_metadata.iloc[source_index]
                suspicious_chunk = suspicious_chunks[suspicious_chunk_id]

                results.append({
                    "method": "LSA",

                    "suspicious_doc": suspicious_doc_id,
                    "suspicious_chunk_id": suspicious_chunk_id,
                    "suspicious_start_char": suspicious_chunk["start_char"],
                    "suspicious_end_char": suspicious_chunk["end_char"],
                    "suspicious_text": suspicious_chunk["chunk_text"],

                    "source_doc": source_row["source_doc"],
                    "source_chunk_id": int(source_row["source_chunk_id"]),
                    "source_start_char": int(source_row["source_start_char"]),
                    "source_end_char": int(source_row["source_end_char"]),
                    "source_text": source_row["source_text"],

                    "score": round(score, 4),
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).sort_values("score", ascending=False)