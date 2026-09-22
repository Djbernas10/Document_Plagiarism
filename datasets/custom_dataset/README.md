# Custom Evaluation Dataset

A small real-world dataset created to validate the pipeline beyond the PAN 2011 synthetic benchmark.

---

## Structure

```
custom_dataset/
├── source_documents/             # 30 real source documents (the "corpus" to search against)
│   ├── *.pdf                     # original PDFs (thesis bibliography)
│   ├── source-document00001.txt  ...  source-document00030.txt   (docling-extracted text)
│   └── manifest.tsv              # doc_id -> original PDF filename mapping
├── suspicious_documents/         # 10 test documents (5 clean + 5 plagiarised), plain text
│   └── suspicious-document00001.txt  ...  suspicious-document00010.txt
├── suspicious_documents_pdf/     # same 10 documents rendered as PDF (5-6 pages each)
│   └── suspicious-document00001.pdf  ...
├── ground_truth/                 # GT XML files (PAN-style char offsets), one per suspicious doc
│   └── suspicious-document00001.xml  ...
├── extract_source_pdfs.py        # docling: source_documents/*.pdf -> source-documentNNNNN.txt
├── generate_test_docs.py         # builds all 10 suspicious docs + ground truth XML from scratch
├── convert_suspicious_to_pdf.py  # renders suspicious_documents/*.txt -> suspicious_documents_pdf/*.pdf
├── verify_ground_truth.py        # sanity-checks every GT offset against the actual document text
└── processed/                    # Pipeline artefacts (chunks, indices, embeddings) — auto-generated
```

---

## Document conventions

### Source documents (`source_documents/`)
- 30 real documents: the academic papers cited in the thesis bibliography
- Original PDFs kept alongside their docling-extracted plain UTF-8 `.txt` counterpart
- Text files named `source-document00001.txt` ... `source-document00030.txt`
- `manifest.tsv` records which original PDF filename each `doc_id` came from
- Re-run extraction with `python datasets/custom_dataset/extract_source_pdfs.py` (uses docling with
  OCR and table-structure recovery disabled, since these are digitally-generated academic PDFs)

> **Known issue (2026-09-22): only 29 distinct documents, not 30.**
> `source-document00020` (`CLEF2011wn-PAN-PotthastEt2011a.pdf`) and
> `source-document00027` (`potthast_2011e.pdf`) are two different PDF exports
> of the same paper: Potthast, Eiselt, Barrón-Cedeño, Stein & Rosso,
> "Overview of the 3rd International Competition on Plagiarism Detection"
> (CLEF 2011). Comparing extracted text confirmed this (identical title,
> authors, and abstract, though the files are not byte-identical and both
> run to 10 pages). This is also the paper this project cites for the
> official Potthast et al. plagdet formula
> (`scripts/final/compute_plagdet_official.py`). Neither doc is used as a
> ground-truth plagiarism source for any suspicious document, so no reported
> metric is affected. If this corpus is cited, use "30 files, 29 distinct
> source papers" rather than "30 distinct documents."

### Suspicious documents (`suspicious_documents/`, `suspicious_documents_pdf/`)
- 10 documents total, ~2000-2300 words / 5-6 pages each:
  - **5 clean** (`suspicious-document00001`–`00005`) — original essays, no plagiarism
  - **5 plagiarised** (`suspicious-document00006`–`00010`) — contain passages derived from source documents
- Built from scratch by `generate_test_docs.py`, which assembles each document from original
  "wrapper" prose plus passages copied/derived from `source_documents/`, tracking every plagiarised
  span's exact character offset programmatically (no manual offset counting)
- Available both as plain `.txt` (for the text pipeline) and as paginated `.pdf` (for testing PDF
  ingestion / docling extraction in the Streamlit app)

### Plagiarism types covered (5 plagiarised docs)
| Doc | Type | Description | Source(s) |
|-----|------|-------------|-----------|
| 00006 | Verbatim | Direct copy-paste of paragraphs from a single source | source-document00019 |
| 00007 | Verbatim | Direct copy-paste, multiple sources | source-document00018, 00016 |
| 00008 | Near-verbatim | Minor word substitutions, same structure | source-document00029 |
| 00009 | Near-verbatim | Sentence reordering within paragraph | source-document00018 |
| 00010 | Paraphrase / idea-level / mixed | Multi-source, paraphrase plus idea-level reuse | source-document00019, 00029, 00016 |

---

## Ground truth format (PAN-style XML)

Each suspicious document has a matching `.xml` file in `ground_truth/`, generated automatically by
`generate_test_docs.py`. `type` is the obfuscation category (`verbatim`, `near-verbatim`,
`paraphrase`, `idea-level`); `obfuscation` is a finer-grained tag (e.g. `synonym-substitution`,
`sentence-reordering`, `heavy-rewrite`, `concept-reuse`). `source_length` is the length of the
**original** passage at `source_offset`, which can differ from `this_length` when the copied text
was paraphrased or edited.

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

Run `python datasets/custom_dataset/verify_ground_truth.py` after any edit to confirm every
`this_offset`/`source_offset` pair still points at matching text.

---

## Regenerating the dataset

```bash
# 1. Extract text from the 30 source PDFs (only needed once, or if PDFs change)
python datasets/custom_dataset/extract_source_pdfs.py

# 2. Build the 10 suspicious documents + ground truth XML
python datasets/custom_dataset/generate_test_docs.py

# 3. Verify every ground-truth offset is correct
python datasets/custom_dataset/verify_ground_truth.py

# 4. Render the suspicious documents as PDF
python datasets/custom_dataset/convert_suspicious_to_pdf.py
```

---

## Pipeline integration

Once documents and GT are ready, run the preprocessing + index creation scripts
(same as PAN 2011 pipeline) pointing to `custom_dataset/` paths, then run:

```bash
python scripts/final/run_pipeline.py --dataset custom --docs 10 --skip-tfidf --run-embeddings
```

(A `--dataset custom` flag will need to be added to run_pipeline.py when ready.)
