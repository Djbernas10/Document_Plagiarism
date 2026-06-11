# Custom Evaluation Dataset

A small real-world dataset created to validate the pipeline beyond the PAN 2011 synthetic benchmark.

---

## Structure

```
custom_dataset/
├── source_documents/        # 30 real source documents (the "corpus" to search against)
│   └── source-document00001.txt  ...
├── suspicious_documents/    # 20 test documents (10 clean + 10 plagiarised)
│   ├── suspicious-document00001.txt  ...
│   └── suspicious-document00001.xml  (ground truth — same folder or in ground_truth/)
├── ground_truth/            # GT XML files (PAN-style char offsets)
│   └── suspicious-document00001.xml  ...
└── processed/               # Pipeline artefacts (chunks, indices, embeddings) — auto-generated
```

---

## Document conventions

### Source documents (`source_documents/`)
- 30 real documents: academic papers, Wikipedia articles, or news articles
- Plain UTF-8 `.txt`, one document per file
- Named `source-document00001.txt` ... `source-document00030.txt`
- Keep to a single domain (e.g. computer science, history) for fair retrieval

### Suspicious documents (`suspicious_documents/`)
- 20 documents total:
  - **10 clean** — original writing with no plagiarism (docs 00001–00010)
  - **10 plagiarised** — contain passages derived from source documents (docs 00011–00020)
- Plain UTF-8 `.txt`

### Plagiarism types to cover (10 plagiarised docs)
| Doc | Type | Description |
|-----|------|-------------|
| 00011 | Verbatim | Direct copy-paste of 1–3 paragraphs |
| 00012 | Verbatim | Direct copy-paste, multiple sources |
| 00013 | Near-verbatim | Minor word substitutions, same structure |
| 00014 | Near-verbatim | Sentence reordering within paragraph |
| 00015 | Paraphrase | Reworded sentences, same meaning |
| 00016 | Paraphrase | Heavy rewrite, key facts/names preserved |
| 00017 | Paraphrase | Mixed — some verbatim, some paraphrased |
| 00018 | Idea-level | Same argument completely rewritten |
| 00019 | Idea-level | Concept reuse, different vocabulary |
| 00020 | Mixed | Multi-source, mixed obfuscation types |

---

## Ground truth format (PAN-style XML)

Each plagiarised suspicious document needs a matching `.xml` file in `ground_truth/`.
Record the exact character offsets of each plagiarised passage when you create it.

```xml
<document reference="suspicious-document00011.txt">
  <feature
    name="plagiarism"
    type="verbatim"
    obfuscation="none"
    this_offset="1200"
    this_length="450"
    source_reference="source-document00003.txt"
    source_offset="8700"
    source_length="450"
  />
</document>
```

Clean documents get an empty XML:
```xml
<document reference="suspicious-document00001.txt">
</document>
```

---

## Pipeline integration

Once documents and GT are ready, run the preprocessing + index creation scripts
(same as PAN 2011 pipeline) pointing to `custom_dataset/` paths, then run:

```bash
python scripts/final/run_pipeline.py --dataset custom --docs 20 --skip-tfidf --run-embeddings
```

(A `--dataset custom` flag will need to be added to run_pipeline.py when ready.)
