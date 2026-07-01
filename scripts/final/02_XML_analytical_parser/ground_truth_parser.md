# 02 — XML Analytical Parser (Ground Truth)

Parses the PAN 2011 XML annotation files to produce two Parquet tables that serve as the gold-standard ground truth for evaluating retrieval and alignment quality.

## What it does

```
dataset_root/
  suspicious-document/partX/suspicious-documentNNNNN.xml
                                       │
                                       ▼
              pan2011_plagiarism_spans.parquet   ← one row per plagiarised passage pair
                                       │
                                       ▼
         pan2011_source_doc_validation.parquet  ← one row per (suspicious doc, source doc)
```

## Files

| File | Purpose |
|---|---|
| `source_parser.py` | All XML parsing and table-building logic |

## Key functions

| Function | Description |
|---|---|
| `build_source_part_lookup` | Scans the filesystem to map each source filename to its `partX` folder. Required because a source document can live in a different part than the suspicious document that references it. |
| `parse_pan_xml_file` | Parses one XML file. Returns one dict per `<feature name="plagiarism">` element, containing both the suspicious-side and source-side character offsets. |
| `collect_pan_plagiarism_spans` | Iterates all XML files and concatenates their rows into the span-level Parquet. |
| `build_source_doc_validation_table` | Groups the span table by `(suspicious_doc_id, source_doc_id)` and aggregates span lists into JSON columns, producing the document-level validation table. |
| `get_ground_truth_for_suspicious_doc` | Convenience filter: returns all true source documents for one suspicious document. |
| `validate_retrieved_source_documents` | Left-joins a retrieval result DataFrame against the ground truth to flag which retrieved documents are true sources. |

## Output Parquet files (written to `datasets/processed/PAN2011_ground_truth/`)

### `pan2011_plagiarism_spans.parquet` — span level

One row per plagiarised passage pair.

| Column | Description |
|---|---|
| `suspicious_doc_id` | `partX__suspicious-documentNNNNN.txt` |
| `source_doc_id` | `partY__source-documentNNNNN.txt` |
| `suspicious_offset` / `suspicious_length` / `suspicious_end` | Char span in the suspicious document (raw file offsets) |
| `source_offset` / `source_length` / `source_end` | Char span in the source document (raw file offsets) |
| `plagiarism_type` | PAN type annotation (e.g. `artificial`, `simulated`) |
| `obfuscation` | Obfuscation strategy (e.g. `none`, `random`, `translation`) |

### `pan2011_source_doc_validation.parquet` — document level

One row per `(suspicious_doc, source_doc)` pair.

| Column | Description |
|---|---|
| `is_true_source_document` | Always `True` in this table (negatives are identified by absence) |
| `true_plagiarism_passage_count` | Number of individual plagiarised spans in this pair |
| `suspicious_ranges_json` | JSON list of suspicious-side span dicts |
| `source_ranges_json` | JSON list of source-side span dicts |
| `aligned_passages_json` | JSON list of matched `(suspicious_span, source_span)` pairs — used in stage 05 |
| `suspicious_total_plagiarized_chars` | Sum of all `suspicious_length` values |
| `source_total_plagiarized_chars` | Sum of all `source_length` values |

## Important design notes

- **Char offsets are raw-file offsets**, not offsets in the cleaned text produced by `01_preprocessing`. Do not mix these up when comparing against chunk `start_char`/`end_char` values.
- The `source_part_lookup` filesystem scan is done once and reused for all XML files, making cross-part resolution O(1) per feature.
- JSON string serialisation for list columns (instead of native Parquet lists) keeps the output compatible with every downstream tool without requiring schema negotiation.
- `validate_retrieved_source_documents` uses a left-join, so every retrieved document is kept in the output; unmatched ones get `is_true_source_document=False` automatically.

## Running

```bash
python scripts/final/02_XML_analytical_parser/source_parser.py
```

Set `DATASET_ROOT` and `OUTPUT_DIR` at the bottom of the file before running.
