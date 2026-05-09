import re
import unicodedata
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# 1. General text cleaning
# ---------------------------------------------------------------------

def clean_pan_source_text(text: str) -> str:
    """
    Light PAN text preprocessing.

    This cleaned text is used as the canonical readable text for:
    - chunking
    - retrieval
    - later passage alignment

    Important:
    start_char/end_char offsets are based on this cleaned text,
    not the original raw file.
    """

    if text is None:
        return ""

    text = text.replace("\x00", " ")
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    text = re.sub(r"\[Illustrated:.*?\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\bCHAPTER\s+[IVXLCDM]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[-=_]{3,}", " ", text)

    # "communi-\ncation" -> "communication"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)

    text = text.replace("\ufeff", "")

    return text.strip()


def chunk_clean_text(
    clean_text: str,
    chunk_size: int = 180,
    overlap: int = 50,
    min_words: int = 40,
) -> List[Dict]:
    """
    Split cleaned text into overlapping word chunks.

    This is the canonical chunking function used before:
    - LSA
    - ESA
    - embeddings / vector DB indexing
    """

    tokens = list(re.finditer(r"\S+", clean_text))

    if not tokens:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start_word in range(0, len(tokens), step):
        end_word = min(start_word + chunk_size, len(tokens))

        if end_word - start_word < min_words:
            continue

        start_char = tokens[start_word].start()
        end_char = tokens[end_word - 1].end()

        chunks.append({
            "chunk_text": clean_text[start_char:end_char],
            "start_char": start_char,
            "end_char": end_char,
            "word_start": start_word,
            "word_end": end_word,
            "word_count": end_word - start_word,
        })

        if end_word == len(tokens):
            break

    return chunks


# ---------------------------------------------------------------------
# 2. Read raw documents and save document-level Parquet
# ---------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    """
    Read PAN text files safely.
    Tries UTF-8 first, then Latin-1 fallback.
    """
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def make_safe_doc_id(relative_path: str) -> str:
    """
    Create stable doc_id from relative path.

    Better than path.name because PAN folders can contain files
    with repeated names in different subfolders.
    """
    return (
        relative_path
        .replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
    )


def collect_documents_to_parquet(
    folder: Path,
    output_path: Path,
    batch_size: int = 250,
    log_every: int = 500,
) -> int:
    """
    Recursively collect all .txt files inside a PAN folder
    and write them directly to Parquet in batches.

    This avoids OOM because it does not store all documents in RAM.
    """

    folder = Path(folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    txt_files = sorted(folder.rglob("*.txt"))
    total_files = len(txt_files)

    print(f"Found {total_files} .txt files in {folder}")

    writer = None
    batch = []
    written_rows = 0

    try:
        for i, path in enumerate(txt_files, start=1):
            relative_path = path.relative_to(folder).as_posix()
            doc_id = make_safe_doc_id(relative_path)

            if i == 1 or i % log_every == 0 or i == total_files:
                print(f"[{i}/{total_files}] Reading: {relative_path}")

            parts = path.relative_to(folder).parts
            part_name = parts[0] if len(parts) > 1 else None

            raw_text = read_text_file(path)
            clean_text = clean_pan_source_text(raw_text)

            batch.append({
                "doc_id": doc_id,
                "file_name": path.name,
                "relative_path": relative_path,
                "part": part_name,

                # Keep both.
                "raw_text": raw_text,
                "clean_text": clean_text,

                "raw_char_count": len(raw_text),
                "clean_char_count": len(clean_text),
                "raw_word_count": len(raw_text.split()),
                "clean_word_count": len(clean_text.split()),
            })

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)

                writer.write_table(table)
                written_rows += len(batch)
                batch.clear()

        if batch:
            df = pd.DataFrame(batch)
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
            written_rows += len(batch)
            batch.clear()

    finally:
        if writer is not None:
            writer.close()

    print(f"Saved {written_rows} rows to: {output_path}")
    return written_rows


# ---------------------------------------------------------------------
# 3. Build canonical chunk Parquet
# ---------------------------------------------------------------------

def build_chunks_from_document_parquet(
    input_parquet: Path,
    output_parquet: Path,
    text_column: str = "clean_text",
    chunk_size: int = 180,
    overlap: int = 50,
    min_words: int = 40,
    batch_size: int = 500,
) -> int:
    """
    Build canonical chunk table from document-level Parquet.

    This output is the shared source of truth for:
    - ESA
    - LSA
    - embeddings
    - LLM alignment

    start_char/end_char are offsets in the cleaned text.
    """

    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not input_parquet.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_parquet}")

    parquet_file = pq.ParquetFile(input_parquet)

    writer = None
    total_chunks = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            docs_df = batch.to_pandas()
            chunk_rows = []

            for _, row in docs_df.iterrows():
                doc_id = row["doc_id"]
                relative_path = row.get("relative_path", None)
                file_name = row.get("file_name", None)
                part = row.get("part", None)

                text = row[text_column]

                chunks = chunk_clean_text(
                    clean_text=text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    min_words=min_words,
                )

                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = f"{Path(doc_id).stem}_c{chunk_index:04d}"

                    chunk_rows.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "relative_path": relative_path,
                        "part": part,

                        "chunk_index": chunk_index,
                        "chunk_text": chunk["chunk_text"],

                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "word_start": chunk["word_start"],
                        "word_end": chunk["word_end"],
                        "word_count": chunk["word_count"],

                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "min_words": min_words,
                    })

            if not chunk_rows:
                continue

            chunk_df = pd.DataFrame(chunk_rows)
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)

            writer.write_table(table)
            total_chunks += len(chunk_rows)

            print(f"Written chunks so far: {total_chunks}")

    finally:
        if writer is not None:
            writer.close()

    print(f"Saved {total_chunks} chunks to: {output_parquet}")
    return total_chunks


# ---------------------------------------------------------------------
# 4. Branch-specific preprocessing: LSA / ESA / TF-IDF
# ---------------------------------------------------------------------

BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "as", "by",
    "this", "that", "these", "those", "it", "its",
}


def normalize_for_lsa_esa(text: str) -> str:
    """
    Stronger lexical normalization for LSA/ESA.

    This is NOT used for embeddings.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Replace punctuation/symbols with spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Optional number normalization.
    text = re.sub(r"\d+", " NUM ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [token for token in tokens if token not in BASIC_STOPWORDS]

    return " ".join(tokens)


def prepare_lsa_esa_chunk_parquet(
    input_chunks_parquet: Path,
    output_parquet: Path,
) -> int:
    """
    Create LSA/ESA-specific text while preserving chunk_id and metadata.
    """

    input_chunks_parquet = Path(input_chunks_parquet)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_chunks_parquet)

    output = df[[
        "chunk_id",
        "doc_id",
        "file_name",
        "relative_path",
        "part",
        "chunk_index",
        "start_char",
        "end_char",
        "word_start",
        "word_end",
        "word_count",
    ]].copy()

    output["lsa_esa_text"] = df["chunk_text"].apply(normalize_for_lsa_esa)
    output["lsa_esa_word_count"] = output["lsa_esa_text"].apply(lambda x: len(x.split()))

    output.to_parquet(output_parquet, index=False)

    print(f"Saved {len(output)} LSA/ESA chunks to: {output_parquet}")
    return len(output)


# ---------------------------------------------------------------------
# 5. Branch-specific preprocessing: embeddings
# ---------------------------------------------------------------------

def normalize_for_embeddings(text: str) -> str:
    """
    Light normalization for transformer embeddings.

    Keeping natural sentence structure.

    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Rough approximation.
    Later, replace with actual embedding model tokenizer if needed.
    """
    if not text:
        return 0

    return int(len(text.split()) * 1.3)


def prepare_embedding_chunk_parquet(
    input_chunks_parquet: Path,
    output_parquet: Path,
) -> int:
    """
    Create embedding-specific text while preserving chunk_id and metadata.
    """

    input_chunks_parquet = Path(input_chunks_parquet)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_chunks_parquet)

    output = df[[
        "chunk_id",
        "doc_id",
        "file_name",
        "relative_path",
        "part",
        "chunk_index",
        "start_char",
        "end_char",
        "word_start",
        "word_end",
        "word_count",
    ]].copy()

    output["embedding_text"] = df["chunk_text"].apply(normalize_for_embeddings)
    output["estimated_token_count"] = output["embedding_text"].apply(estimate_token_count)

    output.to_parquet(output_parquet, index=False)

    print(f"Saved {len(output)} embedding chunks to: {output_parquet}")
    return len(output)


# ---------------------------------------------------------------------
# 6. Preview helpers
# ---------------------------------------------------------------------

def preview_document_parquet(output_path: Path, n: int = 1) -> None:
    output_path = Path(output_path)

    if not output_path.exists():
        print(f"Parquet file not found: {output_path}")
        return

    df = pd.read_parquet(output_path, columns=[
        "doc_id",
        "relative_path",
        "part",
        "raw_char_count",
        "clean_char_count",
        "raw_word_count",
        "clean_word_count",
        "clean_text",
    ])

    if df.empty:
        print("Parquet file is empty.")
        return

    example = df.head(n).iloc[0]

    print(example["relative_path"])
    print("raw chars:", example["raw_char_count"])
    print("clean chars:", example["clean_char_count"])
    print("raw words:", example["raw_word_count"])
    print("clean words:", example["clean_word_count"])
    print(example["clean_text"][:500])


def preview_chunk_parquet(output_path: Path, n: int = 3) -> None:
    output_path = Path(output_path)

    if not output_path.exists():
        print(f"Parquet file not found: {output_path}")
        return

    df = pd.read_parquet(output_path)

    if df.empty:
        print("Parquet file is empty.")
        return

    print(df.head(n).to_string())


# ---------------------------------------------------------------------
# 7. Full processing pipeline
# ---------------------------------------------------------------------

def source_doc_processing(
    SOURCE_FOLDER,
    SUSPICIOUS_FOLDER,
    SOURCE_OUTPUT,
    SUSPICIOUS_OUTPUT,
    batch_size: int = 250,
):
    print("Collecting source documents...")
    source_count = collect_documents_to_parquet(
        SOURCE_FOLDER,
        SOURCE_OUTPUT,
        batch_size=batch_size,
    )

    print("\nCollecting suspicious documents...")
    suspicious_count = collect_documents_to_parquet(
        SUSPICIOUS_FOLDER,
        SUSPICIOUS_OUTPUT,
        batch_size=batch_size,
    )

    print(f"\nSource documents found: {source_count}")
    print(f"Suspicious documents found: {suspicious_count}")

    print(f"\nSaved source Parquet to: {SOURCE_OUTPUT}")
    print(f"Saved suspicious Parquet to: {SUSPICIOUS_OUTPUT}")

    print("\nExample source document:")
    preview_document_parquet(SOURCE_OUTPUT, n=1)


def full_preprocessing_pipeline(
    SOURCE_FOLDER,
    SUSPICIOUS_FOLDER,
    OUTPUT_DIR,
    batch_size: int = 250,
    chunk_size: int = 180,
    overlap: int = 50,
    min_words: int = 40,
):
    """
    Full preprocessing pipeline.

    Creates:
    - source_documents.parquet
    - suspicious_documents.parquet
    - source_chunks.parquet
    - suspicious_chunks.parquet
    - source_chunks_lsa_esa.parquet
    - suspicious_chunks_lsa_esa.parquet
    - source_chunks_embeddings.parquet
    - suspicious_chunks_embeddings.parquet
    """

    OUTPUT_DIR = Path(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_documents_path = OUTPUT_DIR / "source_documents.parquet"
    suspicious_documents_path = OUTPUT_DIR / "suspicious_documents.parquet"

    source_chunks_path = OUTPUT_DIR / "source_chunks.parquet"
    suspicious_chunks_path = OUTPUT_DIR / "suspicious_chunks.parquet"

    source_lsa_esa_path = OUTPUT_DIR / "source_chunks_lsa_esa.parquet"
    suspicious_lsa_esa_path = OUTPUT_DIR / "suspicious_chunks_lsa_esa.parquet"

    source_embeddings_path = OUTPUT_DIR / "source_chunks_embeddings.parquet"
    suspicious_embeddings_path = OUTPUT_DIR / "suspicious_chunks_embeddings.parquet"

    # 1. Raw documents -> document Parquet
    source_doc_processing(
        SOURCE_FOLDER=SOURCE_FOLDER,
        SUSPICIOUS_FOLDER=SUSPICIOUS_FOLDER,
        SOURCE_OUTPUT=source_documents_path,
        SUSPICIOUS_OUTPUT=suspicious_documents_path,
        batch_size=batch_size,
    )

    # 2. Document Parquet -> canonical chunks
    print("\nBuilding source chunks...")
    build_chunks_from_document_parquet(
        input_parquet=source_documents_path,
        output_parquet=source_chunks_path,
        chunk_size=chunk_size,
        overlap=overlap,
        min_words=min_words,
    )

    print("\nBuilding suspicious chunks...")
    build_chunks_from_document_parquet(
        input_parquet=suspicious_documents_path,
        output_parquet=suspicious_chunks_path,
        chunk_size=chunk_size,
        overlap=overlap,
        min_words=min_words,
    )

    # 3. Canonical chunks -> LSA/ESA text
    print("\nPreparing source LSA/ESA chunks...")
    prepare_lsa_esa_chunk_parquet(
        input_chunks_parquet=source_chunks_path,
        output_parquet=source_lsa_esa_path,
    )

    print("\nPreparing suspicious LSA/ESA chunks...")
    prepare_lsa_esa_chunk_parquet(
        input_chunks_parquet=suspicious_chunks_path,
        output_parquet=suspicious_lsa_esa_path,
    )

    # 4. Canonical chunks -> embedding text
    print("\nPreparing source embedding chunks...")
    prepare_embedding_chunk_parquet(
        input_chunks_parquet=source_chunks_path,
        output_parquet=source_embeddings_path,
    )

    print("\nPreparing suspicious embedding chunks...")
    prepare_embedding_chunk_parquet(
        input_chunks_parquet=suspicious_chunks_path,
        output_parquet=suspicious_embeddings_path,
    )

    print("\nPreprocessing complete.")
    print(f"Output folder: {OUTPUT_DIR}")

    print("\nExample source chunk:")
    preview_chunk_parquet(source_chunks_path, n=1)