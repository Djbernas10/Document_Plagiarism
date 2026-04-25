import re
import unicodedata
from typing import List, Dict
import pandas as pd
from pathlib import Path



def clean_pan_source_text(text: str) -> str:
    """
    Simple NLP text preprocessing

    Keeps the text readable and useful for:
    - chunking
    - LSA / ESA retrieval
    - later passage alignment
    """

    if text is None:
        return ""

    # Remove null bytes / broken control chars
    text = text.replace("\x00", " ")

    # Normalize unicode forms
    text = unicodedata.normalize("NFKC", text)

    # Normalize quote and dash variants
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Remove illustration markers, common in old book-style sources
    text = re.sub(r"\[Illustrated:.*?\]", " ", text, flags=re.DOTALL)

    # Remove standalone chapter headings but keep the chapter title text if useful
    # Example: CHAPTER II
    text = re.sub(r"\bCHAPTER\s+[IVXLCDM]+\b", " ", text, flags=re.IGNORECASE)

    # Remove excessive separator-like lines
    text = re.sub(r"[-=_]{3,}", " ", text)

    # Fix hyphenated line-breaks if they exist
    # Example: "communi-\ncation" -> "communication"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Convert all whitespace/newlines/tabs into single spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Add missing space after punctuation if followed by a letter
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)

    text = text.replace("\ufeff", "")

    return text.strip()



def chunk_clean_text( clean_text: str, chunk_size: int = 180, overlap: int = 50, min_words: int = 40) -> List[Dict]:
    """
    Split cleaned text into overlapping word chunks.

    This is used before:
    - LSA
    - ESA
    - embedding/vector database indexing
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
        })

    return chunks

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_text_file(path: Path) -> str:
    """
    Read PAN text files safely.
    Tries UTF-8 first, then Latin-1 fallback.
    """
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def collect_documents_to_parquet(
    folder: Path,
    output_path: Path,
    batch_size: int = 250,
    log_every: int = 500,
) -> int:
    """
    Recursively collect all .txt files inside a PAN folder
    and write them directly to Parquet in batches.

    This avoids OOM because it does NOT store all documents in RAM.
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

            if i == 1 or i % log_every == 0 or i == total_files:
                print(f"[{i}/{total_files}] Reading: {relative_path}")

            parts = path.relative_to(folder).parts
            part_name = parts[0] if len(parts) > 1 else None

            text = read_text_file(path)

            batch.append({
                "doc_id": path.name,
                "relative_path": relative_path,
                "part": part_name,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
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


def preview_parquet(output_path: Path, n: int = 1) -> None:
    """
    Preview saved Parquet without loading the whole dataset.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        print(f"Parquet file not found: {output_path}")
        return

    df = pd.read_parquet(output_path, columns=[
        "doc_id",
        "relative_path",
        "part",
        "char_count",
        "word_count",
        "text",
    ])

    if df.empty:
        print("Parquet file is empty.")
        return

    example = df.head(n).iloc[0]

    print(example["relative_path"])
    print("chars:", example["char_count"])
    print("words:", example["word_count"])
    print(example["text"][:500])


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
    preview_parquet(SOURCE_OUTPUT, n=1)
