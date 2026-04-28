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
    
    
    
#REMOVE   
from typing import Dict, Optional
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from preprocessing import clean_pan_source_text, chunk_clean_text


class ESABranch:
    """
    Branch B:
    Word chunks -> TF-IDF -> Explicit Semantic Analysis-style vectors -> cosine similarity.

    True ESA normally uses an external concept corpus, for example Wikipedia articles.
    This implementation supports:

    1. External concept docs:
       concept_docs = {
           "concept_world_war_i": "article text...",
           "concept_trench_warfare": "article text..."
       }

    2. Dataset-derived concept space:
       If concept_docs is None, source chunks are used as explicit concepts.
       This is easier for MVP, but weaker than true Wikipedia-based ESA.
    """

    def __init__(
        self,
        chunk_size: int = 180,
        overlap: int = 50,
        min_words: int = 40,
        top_k: int = 5,
        max_features: int = 100_000,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.top_k = top_k
        self.max_features = max_features

        self.vectorizer = None
        self.concept_matrix = None
        self.source_vectors = None
        self.source_metadata = None
        self.concept_metadata = None

    def _build_source_chunks(self, source_docs: Dict[str, str]) -> tuple[list[str], pd.DataFrame]:
        rows = []
        source_chunk_texts = []

        for source_doc_id, raw_text in source_docs.items():
            clean_text = clean_pan_source_text(raw_text)

            chunks = chunk_clean_text(
                clean_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_words=self.min_words,
            )

            for chunk_id, chunk in enumerate(chunks):
                source_chunk_texts.append(chunk["chunk_text"])

                rows.append({
                    "source_doc": source_doc_id,
                    "source_chunk_id": chunk_id,
                    "source_start_char": chunk["start_char"],
                    "source_end_char": chunk["end_char"],
                    "source_text": chunk["chunk_text"],
                })

        return source_chunk_texts, pd.DataFrame(rows)

    def _prepare_concepts(
        self,
        source_chunk_texts: list[str],
        concept_docs: Optional[Dict[str, str]] = None,
    ) -> tuple[list[str], pd.DataFrame]:
        if concept_docs:
            concept_rows = []
            concept_texts = []

            for concept_id, concept_text in concept_docs.items():
                clean_text = clean_pan_source_text(concept_text)

                if not clean_text.strip():
                    continue

                concept_texts.append(clean_text)
                concept_rows.append({
                    "concept_id": concept_id,
                    "concept_type": "external_concept",
                })

            return concept_texts, pd.DataFrame(concept_rows)

        concept_texts = source_chunk_texts

        concept_metadata = pd.DataFrame([
            {
                "concept_id": f"source_chunk_concept_{i}",
                "concept_type": "source_chunk_concept",
            }
            for i in range(len(source_chunk_texts))
        ])

        return concept_texts, concept_metadata

    def build_index(
        self,
        source_docs: Dict[str, str],
        concept_docs: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Build ESA retrieval index.

        source_docs:
        {
            "source-document00001.txt": "raw source text..."
        }

        concept_docs optional:
        {
            "World War I": "concept article text...",
            "Trench warfare": "concept article text..."
        }
        """

        source_chunk_texts, self.source_metadata = self._build_source_chunks(source_docs)

        if not source_chunk_texts:
            raise ValueError("No source chunks created for ESA.")

        concept_texts, self.concept_metadata = self._prepare_concepts(
            source_chunk_texts=source_chunk_texts,
            concept_docs=concept_docs,
        )

        if not concept_texts:
            raise ValueError("No ESA concept texts available.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            max_features=self.max_features,
        )

        all_texts = concept_texts + source_chunk_texts
        all_tfidf = self.vectorizer.fit_transform(all_texts)

        concept_tfidf = all_tfidf[:len(concept_texts)]
        source_tfidf = all_tfidf[len(concept_texts):]

        concept_tfidf = normalize(concept_tfidf, norm="l2", axis=1)
        source_tfidf = normalize(source_tfidf, norm="l2", axis=1)

        # ESA representation:
        # each text is represented by similarities to explicit concepts.
        self.concept_matrix = concept_tfidf

        self.source_vectors = source_tfidf @ self.concept_matrix.T
        self.source_vectors = normalize(self.source_vectors, norm="l2", axis=1)

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

        if self.vectorizer is None or self.concept_matrix is None or self.source_vectors is None:
            raise RuntimeError("ESA index has not been built yet.")

        clean_text = clean_pan_source_text(suspicious_text)

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
        suspicious_tfidf = normalize(suspicious_tfidf, norm="l2", axis=1)

        # Convert suspicious chunks into ESA concept-space vectors
        suspicious_vectors = suspicious_tfidf @ self.concept_matrix.T
        suspicious_vectors = normalize(suspicious_vectors, norm="l2", axis=1)

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
                    "method": "ESA",

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
    
    
#REMOVE
from typing import Dict
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_pan_source_text, chunk_clean_text


class EmbeddingBranch:
    """
    Branch C:
    Clean text -> chunks -> SentenceTransformer embeddings -> cosine similarity.

    This branch retrieves candidate source passages/documents using dense embeddings.
    It does not call the LLM.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 180,
        overlap: int = 50,
        min_words: int = 40,
        top_k: int = 5,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.top_k = top_k
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)
        self.source_embeddings = None
        self.source_metadata = None

    def build_index(self, source_docs: Dict[str, str]) -> pd.DataFrame:
        """
        Build embedding index from source documents.

        source_docs:
        {
            "source-document00001.txt": "raw text...",
            "source-document00002.txt": "raw text..."
        }
        """

        rows = []
        source_chunk_texts = []

        for source_doc_id, raw_text in source_docs.items():
            clean_text = clean_pan_source_text(raw_text)

            chunks = chunk_clean_text(
                clean_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_words=self.min_words,
            )

            for chunk_id, chunk in enumerate(chunks):
                source_chunk_texts.append(chunk["chunk_text"])

                rows.append({
                    "source_doc": source_doc_id,
                    "source_chunk_id": chunk_id,
                    "source_start_char": chunk["start_char"],
                    "source_end_char": chunk["end_char"],
                    "source_text": chunk["chunk_text"],
                })

        if not source_chunk_texts:
            raise ValueError("No source chunks created for embedding branch.")

        self.source_metadata = pd.DataFrame(rows)

        self.source_embeddings = self.model.encode(
            source_chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        return self.source_metadata

    def retrieve(
        self,
        suspicious_doc_id: str,
        suspicious_text: str,
        threshold: float = 0.40,
    ) -> pd.DataFrame:
        """
        Retrieve candidate source chunks using embedding cosine similarity.
        """

        if self.source_embeddings is None:
            raise RuntimeError("Embedding index has not been built yet.")

        clean_text = clean_pan_source_text(suspicious_text)

        suspicious_chunks = chunk_clean_text(
            clean_text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            min_words=self.min_words,
        )

        if not suspicious_chunks:
            return pd.DataFrame()

        suspicious_chunk_texts = [chunk["chunk_text"] for chunk in suspicious_chunks]

        suspicious_embeddings = self.model.encode(
            suspicious_chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        similarity_matrix = cosine_similarity(
            suspicious_embeddings,
            self.source_embeddings,
        )

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
                    "method": "EMBEDDING",

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