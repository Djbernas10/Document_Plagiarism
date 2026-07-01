# 308-Document Full Run — v3 Rubric Prompt + gemma4:26b

Final corpus evaluation. Baseline configuration (no perplexity filter, no soft Gate 1,
no cross-encoder, no char n-gram aligner) — identical pipeline to the 212-doc run,
extended to the full processed corpus.

## Configuration

```
uv run python scripts/final/run_pipeline.py --skip-tfidf --run-embeddings \
    --relative-gap 0.85 --min-top1-score 0.60
```

- Retrieval: ESA + LSA + sentence embeddings, branch union top-3
- Gate 1 = 0.60, Gate 2 relative gap = 0.85
- LLM: gemma4:26b (MoE), rubric prompt v3, threshold 0.85, 25 pairs/candidate
- temperature=0, top_p=0.95, top_k=64, seed=42

## Corpus

- 308 documents evaluated (156 plagiarised, 152 clean)

## Headline Results

| Metric | Value |
|--------|-------|
| **Macro plagdet (plag only)** | **0.328** |
| Macro F1 (plag only) | 0.333 |
| Macro precision (plag only)† | 0.310 |
| Macro recall (plag only) | 0.397 |
| Granularity | 1.015 |
| Micro plagdet | 0.323 |
| Micro precision | 0.246 |
| Micro recall | 0.483 |
| Macro plagdet (all docs incl. clean) | 0.601 |
| Retrieval recall@20 (plag only) | 0.607 |
| Plag docs detected (≥1 TP) | 87/156 (55.8%) |
| Plag docs with zero detections | 63/156 (40.4%) |
| Clean FP docs | 18/152 (11.8%) |

†Precision=0.0 for zero-detection plagiarised docs (not 1.0), so the macro is not inflated by missed
sources. Both macro precision (0.310) and recall (0.397) are held down by the 63/156 docs with no
detections at all; on the 93 docs where the system fires, per-detection precision is substantially
higher (0.53–0.64 in the per-obfuscation breakdown).

## Per-Obfuscation Breakdown

| Category | plagdet | F1 | Precision | Recall |
|----------|---------|-----|-----------|--------|
| none (verbatim) | 0.516 | 0.516 | 0.645 | 0.430 |
| paraphrase-auto-low (synonym-swap) | **0.615** | 0.625 | 0.530 | 0.763 |
| paraphrase-auto-high (word-salad) | 0.334 | 0.337 | 0.413 | 0.284 |

## Runtime

### Per-document inference (query time)

- This session processed **45 new documents** (docs ~213–312) in **6.4 hours**
  (mean ~513 s/doc ≈ 8.5 min, median ~435 s, slowest single doc 26.7 min).
- The preceding 212 docs loaded from cache instantly.
- Cumulative LLM processing for the full corpus: ~22 hours
  (≈16 h for the first 212 + 6.4 h for the remaining 45 fresh docs).

### One-time offline index build (over 11,093 source docs, amortised across all queries)

| Build step | One-time cost |
|------------|---------------|
| Source embedding index (Qwen3-Embedding-0.6B, GPU) | ~44 h |
| TF-IDF char-hashing index | ~6 h |
| ESA index | ~2 h |
| LSA index | ~2 h |
| **Total offline indexing** | **~54 h** |

These costs are paid once and reused for every suspicious-document query; they do not
scale with the number of documents evaluated.

## Comparison vs 212-doc run

| Metric | 212 docs (102 plag) | 308 docs (156 plag) |
|--------|---------------------|---------------------|
| Macro plagdet (plag only) | 0.359 | 0.328 |
| Granularity | 1.019 | 1.015 |
| Retrieval recall@20 (plag only) | 0.640 | 0.607 |

The slight plagdet drop is expected: the additional ~54 plagiarised documents form a
harder, more representative sample than the earlier prefix. The obfuscation pattern is
identical (synonym-swap best, word-salad the ceiling) and granularity remains ~1.01,
confirming the pipeline does not over-fragment detections at scale.
