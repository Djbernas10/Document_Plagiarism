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
import json





def read_text_file(path: Path) -> str:
    """
    Read PAN text files safely.
    Tries UTF-8 first, then Latin-1 fallback.
    """

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def collect_documents(folder: Path) -> dict:
    """
    Recursively collect all .txt files inside a PAN folder.

    Example output:
    {
        "part1/source-document00001.txt": {
            "doc_id": "source-document00001.txt",
            "relative_path": "part1/source-document00001.txt",
            "part": "part1",
            "text": "..."
        }
    }
    """

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    documents = {}

    txt_files = sorted(folder.rglob("*.txt"))
    total_files = len(txt_files)

    print(f"Found {total_files} .txt files in {folder}")

    for i, path in enumerate(txt_files, start=1):
        relative_path = path.relative_to(folder).as_posix()
        
        if i == 1 or i % 500 == 0 or i == total_files:
            print(f"[{i}/{total_files}] Reading: {relative_path}")

        # Example: part1/source-document00001.txt
        parts = path.relative_to(folder).parts
        part_name = parts[0] if len(parts) > 1 else None

        text = read_text_file(path)

        documents[relative_path] = {
            "doc_id": path.name,
            "relative_path": relative_path,
            "part": part_name,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    return documents


def save_parquet(data: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index(drop=True)

    df.to_parquet(output_path, index=False)


def source_doc_processing(SOURCE_FOLDER,SUSPICIOUS_FOLDER,SOURCE_OUTPUT,SUSPICIOUS_OUTPUT):
    print("Collecting source documents...")
    source_documents = collect_documents(SOURCE_FOLDER)

    print("Collecting suspicious documents...")
    suspicious_documents = collect_documents(SUSPICIOUS_FOLDER)

    print(f"Source documents found: {len(source_documents)}")
    print(f"Suspicious documents found: {len(suspicious_documents)}")

    print("\nSaving Parquet files...")
    save_parquet(source_documents, SOURCE_OUTPUT)
    save_parquet(suspicious_documents, SUSPICIOUS_OUTPUT)

    print(f"Saved source Parquet to: {SOURCE_OUTPUT}")
    print(f"Saved suspicious Parquet to: {SUSPICIOUS_OUTPUT}")

    print("\nExample source document:")
    first_source_key = next(iter(source_documents), None)
    if first_source_key:
        example = source_documents[first_source_key]
        print(example["relative_path"])
        print("chars:", example["char_count"])
        print("words:", example["word_count"])
        print(example["text"][:500])
