import re
import unicodedata
from typing import List, Dict



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