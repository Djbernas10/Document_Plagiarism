# 06 — Analytics

Evaluates retrieval and alignment quality against the PAN 2011 ground truth. Computes standard IR metrics (Recall@K, MRR, MAP) broken down by retrieval branch and obfuscation type.

## What it does

```
pan2011_source_doc_validation.parquet  (ground truth from 02)
        +
top-N retrieval results per branch    (from 04)
        │
        ▼
Recall@K, MRR, MAP per branch
        +
Per-obfuscation-type breakdown
        +
Fusion vs single-branch comparison
```

## Files

| File | Purpose |
|---|---|
| `analyze_results.ipynb` | Main evaluation notebook — loads all retrieval results, joins with ground truth, computes metrics |
| `kb_analysis.py` | Placeholder / scratch file (currently empty) |

## Metrics computed

| Metric | Description |
|---|---|
| **Recall@K** | Fraction of suspicious documents whose true source appears in the top-K retrieved results. Primary metric for source retrieval. |
| **MRR** | Mean Reciprocal Rank — average of 1/rank for the first correct source document. |
| **MAP** | Mean Average Precision — area under the precision-recall curve across queries. |

## Evaluation breakdown

Results are reported:
- **Per branch**: TF-IDF, ESA, LSA, Embeddings, Fused
- **Per obfuscation type**: `none` (copy-paste), `random` (word substitution), `translation` (cross-language), etc.
- **Across K values**: K = 1, 5, 10, 20, 50

## Notes

- The ground truth used is `pan2011_source_doc_validation.parquet` from `02_XML_analytical_parser`. Each suspicious document may have multiple true source documents (multi-source plagiarism).
- A retrieval result is considered a hit if **any** true source document for that suspicious document appears within rank K.
- The fused branch is expected to outperform any single branch on recall, especially for obfuscated cases where different methods complement each other.
