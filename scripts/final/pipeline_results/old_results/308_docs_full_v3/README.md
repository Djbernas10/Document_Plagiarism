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

## Update (2026-09-16): official Potthast et al. metric + data completeness fix

The original headline numbers below were computed before 2 of the 308 documents'
per-document detection cache (`suspicious-document00010`, `00012`) had been
overwritten by later, unrelated experiment runs. Both docs were rerun under the
exact configuration above (`--embeddings-backend docker-exec` used instead of the
HTTP path, which was hitting a dead port) and `plagdet_summary.parquet` was
regenerated for all 308 docs — see `scripts/final/compute_plagdet_official.py`.

This also introduced the **official Potthast et al. (2011) plagdet formula**
(`scripts/final/compute_plagdet_official.py`), which pools every GT case and every
detection across the *entire* corpus into two sets S (cases) and R (detections),
then averages one equally-weighted overlap-fraction term per case (recall) / per
detection (precision) — see the PAN 2011 overview paper (`CLEF2011wn-PAN-PotthastEt2011a.pdf`),
eq. 1–2. This is neither the "macro" (per-document average) nor "micro" (raw
pooled-character ratio) figures already below, both of which predate this fix.

| Metric | Old (stale, 306/308 docs) | Regenerated (full 308 docs, same macro/micro convention) | Official Potthast et al. (plag-only pooling) |
|--------|------|------|------|
| Precision | 0.310 | 0.3165 (macro) / 0.2482 (micro) | **0.3516** |
| Recall | 0.397 | 0.4028 (macro) / 0.4872 (micro) | **0.3440** |
| Granularity | 1.015 | 1.0152 (macro) | **1.0428** |
| **Plagdet** | **0.328** (macro) / 0.323 (micro) | **0.3342** (macro) / 0.3253 (micro) | **0.3374** |

Per-category official plagdet (pooled per-case, not per-document; replaces the
Per-Obfuscation Breakdown table below for any claim about the *official* metric):

| Category | Official precision | Official recall | **Official plagdet** |
|----------|---------------------|------------------|------------------------|
| none (verbatim) | 0.8091 | 0.1763 | **0.2895** |
| paraphrase-auto-low (synonym-swap) | 0.7190 | 0.5302 | **0.5978** |
| paraphrase-auto-high (word-salad) | 0.8197 | 0.2355 | **0.3489** |
| translation-auto | 0.5976 | 0.2336 | **0.3359** |
| translation-manual | 0.0000 | 0.0000 | **0.0000** |

Note the category ranking changes under the official (per-case, corpus-pooled)
metric: verbatim drops from the best category (0.516–0.702 under the old
per-document conventions) to the worst non-zero category (0.2895). This is not
a sample-size artifact — only 3 documents in the entire 308-doc corpus contain
any verbatim GT case at all (`suspicious-document00032`, 6 cases;
`suspicious-document00126`, 4 cases; `suspicious-document00228`, 7 cases), and
of those, only `00032` was detected by the pipeline (4 of its 6 cases,
contributing all 3 case-level detections behind the 0.1763 recall above);
`00126` and `00228` each produced **zero detections of any kind**, not just a
verbatim-specific miss. Since `00032` falls in the 1–80 tuning range and both
zero-detection documents fall in the untuned 81–308 tail, this also explains
why the tail's own verbatim category score is exactly 0.0000 (see below).
Synonym-swap (paraphrase-auto-low) remains the strongest category either way.

Conditional precision ("precision on documents where the pipeline fires at all")
is **not a distinct quantity under the official formula** — official precision is
already an average over individual detections, and every detection by definition
lives inside a firing document, so official precision computed over all 308 docs
(0.2357, all-docs pooling) and over just the 112 firing docs are numerically
identical. The old macro precision's document-level "fires vs. doesn't" split
(0.310 → ~0.53) is an artifact specific to per-document averaging and has no
equivalent under the corpus-pooled official metric.

### Official plagdet split by tuning subset vs. held-out tail

Docs 1–80 of this corpus were used during development for parameter tuning
(Gate 1/Gate 2 thresholds, pairs-per-doc, prompt iteration — see
`EXPERIMENTS_SUMMARY.md`); docs 81–308 were never touched during tuning and
are the closest thing this evaluation has to a held-out set. Reporting them
separately, rather than blending them into one 308-doc figure, makes clear
how much (if any) of the headline score is inflated by having been tuned on
part of the same data it's evaluated on.

Computed with `compute_plagdet_official.py --run 308 --doc-range <range>`.
A print-formatted, three-table PDF version of everything below is at
`scripts/final/pipeline_results/official_plagdet_annex.pdf`
(built by `scripts/final/build_official_plagdet_annex.py`).

**Docs 1–80 (tuning subset, 43 plag / 37 clean):**

| Variant | Precision | Recall | F1 | Granularity | **Plagdet** | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2145 | 0.3737 | 0.2725 | 1.0342 | 0.2660 | 290 | 261 | 80 |
| **Plagiarised docs only, pooled** | **0.3059** | **0.3737** | **0.3364** | **1.0342** | **0.3284** | 290 | 183 | 43 |
| Conditional precision (firing docs only) | 0.2145 | — | — | — | — | — | 261 | 33 |
| Category: none (verbatim) | 0.8091 | 0.4995 | 0.6177 | 1.0000 | 0.6177 | 6 | 3 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8625 | 0.2141 | 0.3430 | 1.0286 | 0.3361 | 155 | 12 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7098 | 0.6201 | 0.6619 | 1.0417 | 0.6428 | 107 | 54 | — |
| Category: translation-auto | 0.6968 | 0.3432 | 0.4599 | 1.0000 | 0.4599 | 17 | 7 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 5 | 0 | — |

**Docs 81–308 (held-out tail, 113 plag / 115 clean):**

| Variant | Precision | Recall | F1 | Granularity | **Plagdet** | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2426 | 0.3330 | 0.2807 | 1.0461 | 0.2718 | 781 | 798 | 228 |
| **Plagiarised docs only, pooled** | **0.3674** | **0.3330** | **0.3494** | **1.0461** | **0.3382** | 781 | 527 | 113 |
| Conditional precision (firing docs only) | 0.2426 | — | — | — | — | — | 798 | 79 |
| Category: none (verbatim) | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 11 | 0 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8149 | 0.2435 | 0.3749 | 1.0794 | 0.3550 | 413 | 107 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7225 | 0.4989 | 0.5902 | 1.0238 | 0.5803 | 308 | 140 | — |
| Category: translation-auto | 0.5281 | 0.1772 | 0.2653 | 1.0000 | 0.2653 | 33 | 10 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 16 | 0 | — |

**Docs 1–308 (overall corpus, 156 plag / 152 clean):**

| Variant | Precision | Recall | F1 | Granularity | **Plagdet** | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2357 | 0.3440 | 0.2797 | 1.0428 | 0.2715 | 1071 | 1059 | 308 |
| **Plagiarised docs only, pooled** | **0.3516** | **0.3440** | **0.3477** | **1.0428** | **0.3374** | 1071 | 710 | 156 |
| Conditional precision (firing docs only) | 0.2357 | — | — | — | — | — | 1059 | 112 |
| Category: none (verbatim) | 0.8091 | 0.1763 | 0.2895 | 1.0000 | 0.2895 | 17 | 3 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8197 | 0.2355 | 0.3658 | 1.0683 | 0.3489 | 568 | 119 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7190 | 0.5302 | 0.6103 | 1.0292 | 0.5978 | 415 | 194 | — |
| Category: translation-auto | 0.5976 | 0.2336 | 0.3359 | 1.0000 | 0.3359 | 50 | 17 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 21 | 0 | — |

The "conditional precision (firing docs only)" row is identical to the "all
docs pooled" precision in every range above — not a computation artifact, but
a structural property of the official formula: precision is already an
average over individual detections, and every detection by definition exists
inside a document that fired at least once, so restricting to firing
documents changes nothing about the set of detections being averaged.

The held-out tail (0.3382) scores marginally *higher* than the tuning subset
(0.3284) under the official metric, and both are close to the overall figure
(0.3374). This is reassuring: it means the tuning subset is not an easy subset
that inflates the headline number, and the pipeline's parameters generalise to
documents that were never used to choose them. Per-category official plagdet
also holds a similar shape across both ranges — synonym-swap (paraphrase-auto-low)
is the strongest category in both (0.6428 on 1–80, 0.5803 on 81–308).

**Verbatim ("none") is 0.6177 on 1–80 but 0.0000 on 81–308 — this is not
sampling noise, it is a genuine, complete detection failure.** The 81–308
tail's 11 verbatim GT cases live in exactly 2 documents
(`suspicious-document00126.txt`, 4 cases; `suspicious-document00228.txt`, 7
cases), and both documents have **zero detections of any kind** (`det_spans =
0` in `plagdet_summary.parquet`) — the pipeline did not merely miss the
verbatim spans specifically, it produced no output at all for either
document (consistent with a Gate 1 retrieval block or a full LLM rejection
across all candidates; not investigated further here). |S|=11 in the table
above is correct; |R|=0 reflects that no detection exists to pool into that
category, not a filtering bug. This should be flagged in the thesis text as
a real gap — the verbatim/best-case category is entirely unrepresented by
correct detections in the untuned two-thirds of the corpus, in contrast to
the 1–80 tuning subset where it is the second-strongest category.

Raw output:
`scripts/final/pipeline_results/official_plagdet/plagdet_official_summary.parquet` (overall 308),
`scripts/final/pipeline_results/official_plagdet_docs_1_80/plagdet_official_summary.parquet` (1–80),
`scripts/final/pipeline_results/official_plagdet_docs_81_308/plagdet_official_summary.parquet` (81–308).

## Update (2026-09-19): retrieval recall@20 data-completeness fix

**This section is about the retrieval stage only, not plagdet.** Recall@20
measures whether the true source document appeared anywhere in the top-20
candidates *before* the LLM confirmation stage even runs — it is a diagnostic
for the retrieval branches (ESA/LSA/embeddings), and is computed independently
of, and does not feed into, the official Potthast et al. plagdet formula
above (that formula only uses confirmed *detections*, not retrieval
candidates). The two are reported side by side in this file because they
come from the same 308-doc run, not because one is derived from the other.

`retrieval_recall.parquet` (the file backing the "Retrieval recall@20" row
below and in Table 6.4) was archived with only 257 of 308 rows. All 26
missing rows fell inside the 81–308 held-out tail, and every one of them was
a document where Gate 1 failed on its real fusion score — the retrieval
code path discards the full top-20 candidate list once Gate 1 rejects a
document (`lookup_pipeline` returns only the top-1 score in that branch), so
recall@20 was never recorded for these 26 docs rather than being wrong.

All 26 documents genuinely contain GT plagiarism spans (1–28 spans each, none
are clean docs), so recall@20 could not default to any trivial value for
them. They were rerun with Gate 1 disabled (`--min-top1-score 0.0`, otherwise
identical configuration) purely to force the full candidate list to survive
so recall@20 could be computed independently — this diagnostic rerun does
not change the pipeline's real Gate-1-failed status already recorded for
these 26 docs in `plagdet_summary.parquet`/`analytics_summary.parquet`.

| Range | Plag docs with data (before → after) | Retrieval recall@20 (before, partial) | **Retrieval recall@20 (after, complete)** |
|-------|----------------------------------------|-----------------------------------------|----------------------------------------------|
| 1–80 (tuning subset) | 43 → 43 (no gap) | 0.5604 | **0.5604** (unchanged) |
| 81–308 (held-out tail) | 87 → 113 | 0.6307 | **0.6028** |
| 1–308 (overall) | 130 → 156 | ~0.607 (see note below) | **0.5911** |

The 26 previously-missing tail docs average recall@20 = 0.5097 on their own —
meaningfully worse than the 87 tail docs that already had data — so the
earlier partial 81–308 figure (0.6307) was quietly optimistic: it silently
excluded the harder Gate-1-failed cases. The complete, corrected figures
(0.6028 for the tail, 0.5911 overall) are the ones that should be used in
Table 6.4 and anywhere else "Retrieval recall@20" is cited; the original
0.607 figure quoted below and in the "Headline Results (original)" table
predates this fix (it was itself computed over an incomplete 257/308 base,
not the full corpus) and should not be quoted going forward.

Merged, complete retrieval-recall data (283 rows: 257 archived + 26 backfilled):
`scripts/final/pipeline_results/retrieval_recall_merged_308.parquet`.

### Retrieval recall@20 by subset (final, complete)

| Range | Plag docs | Docs with true source in top-20 | **Retrieval recall@20 (macro)** |
|-------|-----------|-----------------------------------|-------------------------------------|
| 1–80 (tuning subset) | 43 | 27/43 | **0.5604** |
| 81–308 (held-out tail) | 113 | 82/113 | **0.6028** |
| 1–308 (overall) | 156 | 109/156 | **0.5911** |

("Docs with true source in top-20" counts documents with recall@20 > 0, i.e.
at least one of the document's true source(s) appeared in the retrieval
system's top-20 candidates; multi-source documents can score a fractional
recall@20 between 0 and 1 if only some of their true sources were retrieved,
which is why this count and the macro average are reported separately.)

Recall@1 per subset/category was not computed in this pass — `retrieval_recall.parquet`
only stores whichever single `recall_at_k` was requested at run time (k=20 here), not
the full ranked candidate list, and the scratch file holding per-doc candidate rankings
(`embedding_top_source_documents_by_max_score.parquet`) is overwritten on every document
processed rather than kept per-doc. Getting recall@1 would need a separate rerun with
`--retrieval-recall-k 1`.

## Update (2026-09-21): plagiarized-detected and clean-FP rates, by scope, TP-definition fix

Computed with `scripts/final/compute_detection_fp_rates.py` from the current, complete
`plagdet_summary.parquet` only (no pipeline rerun). Two definitions of "detected" exist
in this codebase and they disagree by exactly the same 2 documents this file has already
flagged twice above (`suspicious-document00010`, `00012`):

- `tp_chars > 0` (character-overlap TP) — from the current, regenerated
  `plagdet_summary.parquet` (308 rows, doc00010/12 fixed) — gives **85/156**.
- `tp > 0` (discrete span-count TP) — from the archived
  `old_results/308_docs_full_v3/analytics_summary.parquet` (308 rows, but doc00010/12
  are **stale**: `det_spans` = 0 and 9 there, not the corrected 5 and 4) — gives the
  previously-cited **87/156**.

The 87/156 figure already in the thesis was computed on the stale, pre-regeneration
detection data for those 2 documents — it is not an independently valid alternative
metric, it is the same character-vs-span discrepancy applied to uncorrected data.
**87/156 (55.8%) should be replaced with 85/156 (54.5%) everywhere it is cited.**

| Scope | Plag docs | Plag detected | Detection rate | Clean docs | Clean FP | FP rate |
|-------|-----------|----------------|-----------------|------------|----------|---------|
| 81–308 | 113 | 60 | 53.1% | 115 | 12 | 10.4% |
| 1–80 | 43 | 25 | 58.1% | 37 | 6 | 16.2% |
| 1–308 | 156 | 85 | 54.5% | 152 | 18 | 11.8% |

Sanity checks: in every scope, (plagiarized docs that fired) + (clean docs with a false
positive) exactly equals (total firing documents) — 33 for 1–80, 79 for 81–308, 112 for
1–308 — confirming the arithmetic is internally consistent. The clean-FP figure (18/152,
11.8%) matches the thesis exactly and required no correction.

The same computation was also run on the 10-document custom dataset
(`scripts/final/pipeline_results_custom/plagdet_summary.parquet`, already fully
regenerated earlier this session — no stale docs there):

| Scope | Plag docs | Plag detected | Detection rate | Clean docs | Clean FP | FP rate |
|-------|-----------|----------------|-----------------|------------|----------|---------|
| custom (10 docs) | 5 | 4 | 80.0% | 5 | 1 | 20.0% |

This matches the custom-dataset narrative already established (Table 6.9: 1/5 clean
docs with a false alarm; `suspicious-document00010` is the one zero-detection
plagiarised doc, a known Gate-1/short-span failure mode, not a data-completeness issue)
— no correction needed for the custom dataset's figures.

Raw output: `scripts/final/pipeline_results/detection_fp_rates_by_scope.csv` (all 4 scopes,
including custom-10doc).

## Headline Results (original, 2026-06-15 — kept for historical reference; superseded above)

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
| Retrieval recall@20 (plag only) | ~~0.607~~ **0.5911** (corrected 2026-09-19, see Update above) |
| Plag docs detected (≥1 TP) | ~~87/156 (55.8%)~~ **85/156 (54.5%)** (corrected 2026-09-21, see note below) |
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
