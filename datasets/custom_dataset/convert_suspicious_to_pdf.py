"""Render suspicious_documents/*.txt as paginated PDFs for end-to-end pipeline testing
(the pipeline's preprocessing step expects .txt, but we also want PDF versions to
exercise PDF ingestion / docling extraction in the Streamlit app).

Run from project root:
    python datasets/custom_dataset/convert_suspicious_to_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

BASE = Path(__file__).parent
SUSP_DIR = BASE / "suspicious_documents"
PDF_DIR = BASE / "suspicious_documents_pdf"
PDF_DIR.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
TITLE_STYLE = styles["Heading1"]
HEADING_STYLE = styles["Heading2"]
BODY_STYLE = styles["BodyText"]
BODY_STYLE.spaceAfter = 10


def build_pdf(txt_path: Path, pdf_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    flowables = []
    for i, para in enumerate(paragraphs):
        style = TITLE_STYLE if i == 0 else (HEADING_STYLE if len(para) < 80 and "\n" not in para and para.count(" ") < 8 else BODY_STYLE)
        flowables.append(Paragraph(escape(para).replace("\n", "<br/>"), style))
        flowables.append(Spacer(1, 8))

    doc.build(flowables)


def main() -> None:
    txt_paths = sorted(SUSP_DIR.glob("*.txt"))
    for txt_path in txt_paths:
        pdf_path = PDF_DIR / f"{txt_path.stem}.pdf"
        build_pdf(txt_path, pdf_path)
        print(f"{txt_path.name} -> {pdf_path.name}")


if __name__ == "__main__":
    main()
