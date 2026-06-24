from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path(__file__).parent / "suspicious_documents_pdf"

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    n_pages = len(PdfReader(str(pdf_path)).pages)
    print(f"{pdf_path.name}: {n_pages} pages")
