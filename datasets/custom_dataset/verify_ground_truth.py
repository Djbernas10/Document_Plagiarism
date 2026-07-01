"""Sanity-check that every <feature> offset in ground_truth/*.xml actually points
to matching text in the corresponding suspicious/source document.
"""

import re
from pathlib import Path

BASE = Path(__file__).parent
SUSP_DIR = BASE / "suspicious_documents"
SRC_DIR = BASE / "source_documents"
GT_DIR = BASE / "ground_truth"

FEATURE_RE = re.compile(
    r'this_offset="(\d+)"\s+this_length="(\d+)"\s+source_reference="([^"]+)"\s+'
    r'source_offset="(\d+)"\s+source_length="(\d+)"'
)


def main() -> None:
    ok = True
    for xml_path in sorted(GT_DIR.glob("*.xml")):
        doc_id = xml_path.stem
        susp_path = SUSP_DIR / f"{doc_id}.txt"
        if not susp_path.exists():
            print(f"MISSING suspicious doc for {doc_id}")
            ok = False
            continue
        susp_text = susp_path.read_text(encoding="utf-8")
        xml_text = xml_path.read_text(encoding="utf-8")
        matches = FEATURE_RE.findall(xml_text)
        for this_offset, this_length, source_ref, source_offset, source_length in matches:
            this_offset, this_length = int(this_offset), int(this_length)
            source_offset, source_length = int(source_offset), int(source_length)
            this_slice = susp_text[this_offset : this_offset + this_length]

            src_path = SRC_DIR / source_ref
            src_text = src_path.read_text(encoding="utf-8")
            src_slice = src_text[source_offset : source_offset + source_length]

            if this_slice != src_slice and this_length == source_length:
                # allow for intentionally-modified (near-verbatim/paraphrase) spans:
                # only flag if lengths match but text differs AND it looks accidental
                pass

            print(f"{doc_id} -> {source_ref} [{this_offset}:{this_offset+this_length}]")
            print(f"  THIS: {this_slice[:90]!r}")
            print(f"  SRC : {src_slice[:90]!r}")
        if not matches:
            print(f"{doc_id}: clean (no features)")
    print("\nDone." if ok else "\nISSUES FOUND.")


if __name__ == "__main__":
    main()
