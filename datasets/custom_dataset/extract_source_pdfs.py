"""Extract text from the 30 thesis-bibliography PDFs in source_documents/
into plain UTF-8 .txt files named source-document00001.txt ... source-document00030.txt,
matching the convention in datasets/custom_dataset/README.md.

Run from project root:
    python datasets/custom_dataset/extract_source_pdfs.py
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

SOURCE_DIR = Path(__file__).parent / "source_documents"


def main() -> None:
    pdf_paths = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {SOURCE_DIR}")
        return

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        images_scale=1.0,
        generate_page_images=False,
        generate_picture_images=False,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    manifest_lines = ["doc_id\toriginal_filename"]

    for i, pdf_path in enumerate(pdf_paths, start=1):
        doc_id = f"source-document{i:05d}"
        txt_path = SOURCE_DIR / f"{doc_id}.txt"

        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        txt_path.write_text(text, encoding="utf-8")
        manifest_lines.append(f"{doc_id}\t{pdf_path.name}")
        print(f"[{i:02d}/{len(pdf_paths)}] {pdf_path.name} -> {txt_path.name} ({len(text)} chars)")

    (SOURCE_DIR / "manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nWrote manifest.tsv mapping doc_id -> original filename in {SOURCE_DIR}")


if __name__ == "__main__":
    main()
