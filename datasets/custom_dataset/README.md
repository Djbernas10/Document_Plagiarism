# Custom evaluation dataset

A small dataset of real academic papers, built to test the pipeline outside the synthetic PAN 2011
benchmark.

---

## What is in the repository

```
custom_dataset/
├── suspicious_documents/         # 10 test documents (5 clean + 5 plagiarised), plain text
│   └── suspicious-document00001.txt  ...  suspicious-document00010.txt
├── ground_truth/                 # PAN-style XML annotations, one per suspicious document
│   └── suspicious-document00001.xml  ...  suspicious-document00010.xml
├── extract_source_pdfs.py        # docling: source_documents/*.pdf -> source-documentNNNNN.txt
├── generate_test_docs.py         # builds the 10 suspicious documents and their ground-truth XML
├── verify_ground_truth.py        # checks every ground-truth offset against the document text
├── convert_suspicious_to_pdf.py  # renders suspicious_documents/*.txt as PDF
├── count_pdf_pages.py            # prints the page count of each rendered suspicious PDF
├── EXPERIMENTS_SUMMARY.md        # results and notes from the custom-dataset runs
└── README.md
```

The source corpus is not in the repository. `source_documents/` (the 30 original PDFs, their
extracted `.txt` files and `manifest.tsv`) stays local because the papers are third-party
copyrighted publications. The PDF versions of the suspicious documents are also not committed,
since `*.pdf` is git-ignored. To rebuild the dataset you need to get the source papers from the
project report bibliography yourself. The suspicious-document PDFs can be regenerated from the committed
`.txt` files with `convert_suspicious_to_pdf.py`.

---

## Document conventions

### Source documents (local only)
- 30 academic papers cited in the project report bibliography, placed as PDFs in `source_documents/`
- `extract_source_pdfs.py` converts them with docling into plain UTF-8 files named
  `source-document00001.txt` ... `source-document00030.txt` and writes `manifest.tsv`, which maps
  each `doc_id` to its original PDF filename. OCR and table-structure recovery are turned off
  because the papers are digitally generated PDFs.

> **Known issue (2026-09-22): the 30 files contain 29 distinct papers.**
> `source-document00020` (`CLEF2011wn-PAN-PotthastEt2011a.pdf`) and
> `source-document00027` (`potthast_2011e.pdf`) are two PDF exports of the same paper:
> Potthast, Eiselt, Barrón-Cedeño, Stein & Rosso, "Overview of the 3rd International Competition
> on Plagiarism Detection" (CLEF 2011). Their extracted text has the same title, authors and
> abstract. The files are not byte-identical, and both are 10 pages long. This is also the paper
> the project cites for the official Potthast et al. plagdet formula
> (`scripts/final/compute_plagdet_official.py`). Neither document is a ground-truth source for any
> suspicious document, so no reported metric changes. If the corpus size is cited, use "30 files,
> 29 distinct source papers".

### Suspicious documents (`suspicious_documents/`)
- 10 documents of about 2000-2300 words each (5-6 pages when rendered as PDF):
  - 5 clean (`suspicious-document00001`–`00005`): original essays with no plagiarism
  - 5 plagiarised (`suspicious-document00006`–`00010`): contain passages taken from source documents
- `generate_test_docs.py` builds each document from original "wrapper" prose plus passages copied
  or adapted from the source documents. It records the character offset of every plagiarised span
  as it builds the text, so no offsets are counted by hand.
- The `.txt` files feed the text pipeline. The PDF versions, generated locally, are for testing PDF
  ingestion and docling extraction in the Streamlit app.

### Plagiarism types in the 5 plagiarised documents
| Doc | Type | Description | Source(s) |
|-----|------|-------------|-----------|
| 00006 | Verbatim | Paragraphs copied directly from one source | source-document00019 |
| 00007 | Verbatim | Paragraphs copied directly from several sources | source-document00018, 00016 |
| 00008 | Near-verbatim | Minor word substitutions, same structure | source-document00029 |
| 00009 | Near-verbatim | Sentences reordered within a paragraph | source-document00018 |
| 00010 | Paraphrase / idea-level / mixed | Several sources, paraphrase plus idea-level reuse | source-document00019, 00029, 00016 |

---

## Ground truth format (PAN-style XML)

`generate_test_docs.py` writes one `.xml` file per suspicious document into `ground_truth/`.
`type` is the obfuscation category (`verbatim`, `near-verbatim`, `paraphrase`, `idea-level`), and
`obfuscation` is a finer tag such as `synonym-substitution`, `sentence-reordering`, `heavy-rewrite`
or `concept-reuse`. `source_length` is the length of the original passage at `source_offset`. It can
differ from `this_length` when the copied text was paraphrased or edited.

```xml
<document reference="suspicious-document00006.txt">
  <feature
    name="plagiarism"
    type="verbatim"
    obfuscation="none"
    this_offset="590"
    this_length="866"
    source_reference="source-document00019.txt"
    source_offset="2594"
    source_length="866"
  />
</document>
```

Clean documents get an empty XML:
```xml
<document reference="suspicious-document00001.txt">
</document>
```

After any edit, run `python datasets/custom_dataset/verify_ground_truth.py` to confirm that every
`this_offset`/`source_offset` pair still points at matching text. It needs the local source `.txt`
files.

---

## Regenerating the dataset

```bash
# 1. Extract text from the 30 source PDFs (needs the PDFs in source_documents/, not in the repo)
python datasets/custom_dataset/extract_source_pdfs.py

# 2. Build the 10 suspicious documents and the ground-truth XML
python datasets/custom_dataset/generate_test_docs.py

# 3. Check every ground-truth offset
python datasets/custom_dataset/verify_ground_truth.py

# 4. Optional: render the suspicious documents as PDF (written to suspicious_documents_pdf/)
python datasets/custom_dataset/convert_suspicious_to_pdf.py
```

---

## Pipeline integration

With the source and suspicious documents in place, build the processed inputs and indexes with the
custom-dataset scripts:

```bash
python scripts/final/01_preprocessing/preprocess_custom_dataset.py
python scripts/final/02_XML_analytical_parser/build_custom_ground_truth.py
python scripts/final/03_index_creation/build_esa_index_custom.py
python scripts/final/03_index_creation/build_lsa_index_custom.py
python scripts/final/03_index_creation/build_tfidf_index_custom.py
```

Then run the pipeline on the custom dataset:

```bash
python scripts/final/run_pipeline.py --dataset custom --docs 10 --skip-tfidf --run-embeddings
```
