# 308-document full run (v3 rubric prompt + gemma4:26b)

Final corpus evaluation. It uses the baseline configuration (no perplexity filter, no soft Gate 1,
no cross-encoder, no char n-gram aligner), the same as the 212-document run, extended to the full
processed corpus.

The run's result files (per-document detections, plagdet summaries, retrieval recall and the
official-metric outputs) stay local and are not committed. This README records the results. The
scripts that compute them are in `scripts/final/`.

## Configuration

```
uv run python scripts/final/run_pipeline.py --skip-tfidf --run-embeddings \
    --relative-gap 0.85 --min-top1-score 0.60
```

- Retrieval: ESA + LSA + sentence embeddings, branch union top-3
- Gate 1 = 0.60, Gate 2 relative gap = 0.85
- LLM: gemma4:26b (MoE), rubric prompt v3, threshold 0.85, 25 pairs per candidate
- temperature=0, top_p=0.95, top_k=64, seed=42

## Corpus

- 308 documents evaluated (156 plagiarised, 152 clean)

## Update (2026-09-16): official Potthast et al. metric and data completeness fix

The original headline numbers at the end of this file were computed after the cached per-document
detections for 2 of the 308 documents (`suspicious-document00010` and `00012`) had been overwritten
by later, unrelated experiments. Both documents were rerun with the configuration above, using
`--embeddings-backend docker-exec` because the HTTP path was pointing at a dead port. The plagdet
summary was then regenerated for all 308 documents.

This update also added the official Potthast et al. (2011) plagdet formula
(`scripts/final/compute_plagdet_official.py`). It pools every ground-truth case and every detection
across the whole corpus into two sets, S (cases) and R (detections). Recall is the average of one
equally weighted overlap-fraction term per case, and precision is the same per detection (PAN 2011
overview paper, eq. 1-2). This is neither the macro (per-document average) nor the micro (pooled
character ratio) figure reported further down. Both of those predate this fix.

| Metric | Old (stale, 306/308 docs) | Regenerated (all 308 docs, same macro/micro convention) | Official Potthast et al. (plag-only pooling) |
|--------|------|------|------|
| Precision | 0.310 | 0.3165 (macro) / 0.2482 (micro) | 0.3516 |
| Recall | 0.397 | 0.4028 (macro) / 0.4872 (micro) | 0.3440 |
| Granularity | 1.015 | 1.0152 (macro) | 1.0428 |
| Plagdet | 0.328 (macro) / 0.323 (micro) | 0.3342 (macro) / 0.3253 (micro) | **0.3374** |

> **Correction (2026-09-28):** the "none (verbatim)" row and the narrative below it were
> contaminated by a classification bug in `classify_case_category`
> (`scripts/final/compute_plagdet_official.py`) / the duplicated inline copy in
> `compute_plagdet.py`. Ground-truth cases with `type="simulated"` (PAN's manually/simulated
> plagiarism-paraphrase category) carry no `obfuscation` XML attribute at all, so they fell
> through to the `obfuscation == "none"` branch and were silently merged into the true
> no-obfuscation ("verbatim") bucket. A `typ == "simulated"` branch returning a dedicated
> `paraphrase-manual` category, placed before that fallback, fixes this. The table and
> narrative below are the corrected, post-fix numbers; see `git log` on `compute_plagdet.py`
> and `compute_plagdet_official.py` for the fix.

Official plagdet by category (pooled per case, not per document). For any claim about the official
metric, use this table instead of the per-obfuscation breakdown further down:

| Category | Official precision | Official recall | Official plagdet |
|----------|---------------------|------------------|------------------------|
| none (verbatim) | 0.0000 | 0.0000 | 0.0000 |
| paraphrase-manual | 0.8091 | 0.2305 | 0.3588 |
| paraphrase-auto-low (synonym-swap) | 0.7190 | 0.5302 | 0.5978 |
| paraphrase-auto-high (word-salad) | 0.8197 | 0.2355 | 0.3489 |
| translation-auto | 0.5976 | 0.2336 | 0.3359 |
| translation-manual | 0.0000 | 0.0000 | 0.0000 |

The category ranking changes under the official metric, but not in the way the pre-fix table
suggested. Only 1 document in the 308-document corpus contains a true verbatim ground-truth
case: `suspicious-document00126` (4 cases, `type="artificial" obfuscation="none"`), and the
pipeline produced zero detections for it — official verbatim recall/plagdet is **0.0000**, not
0.2895. The other two documents previously cited as "verbatim" are not verbatim at all:
`suspicious-document00032` (6 cases) and `suspicious-document00228` (7 cases) are both entirely
`type="simulated"` (manual paraphrasing), 13 cases total, matching the `paraphrase-manual`
row's `|S|=13`. The pipeline detected 4 of `00032`'s 6 paraphrase-manual cases (all 4
case-level detections behind the 0.2305 recall on that row); `00228` produced no detections at
all. `00032` is in the 1-80 tuning range and `00228` is in the untuned 81-308 tail. Synonym-swap
(paraphrase-auto-low) is the strongest category under both conventions; true verbatim is now the
single worst category (0.0000), not paraphrase-manual.

Conditional precision (precision on documents where the pipeline fires at all) is not a separate
quantity under the official formula. Official precision already averages over individual
detections, and every detection belongs to a document that fired, so official precision over all
308 documents (0.2357, all-docs pooling) equals official precision over the 112 firing documents.
The old macro precision's split between firing and non-firing documents (0.310 to about 0.53) only
exists because of per-document averaging and has no counterpart under the corpus-pooled metric.

### Official plagdet by tuning subset and held-out tail

Documents 1-80 were used during development to tune parameters (Gate 1 and Gate 2 thresholds,
pairs per document, prompt iterations; see `../EXPERIMENTS_SUMMARY.md`). Documents 81-308 were never
used for tuning and are the closest this evaluation has to a held-out set. Reporting them separately
shows whether tuning on part of the evaluation data inflated the headline score.

Computed with `compute_plagdet_official.py --run 308 --doc-range <range>`.

**Docs 1–80 (tuning subset, 43 plag / 37 clean):**

| Variant | Precision | Recall | F1 | Granularity | Plagdet | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2145 | 0.3737 | 0.2725 | 1.0342 | 0.2660 | 290 | 261 | 80 |
| **Plagiarised docs only, pooled** | **0.3059** | **0.3737** | **0.3364** | **1.0342** | **0.3284** | 290 | 183 | 43 |
| Conditional precision (firing docs only) | 0.2145 | — | — | — | — | — | 261 | 33 |
| Category: none (verbatim) | — | — | — | — | — | 0 | 0 | — |
| Category: paraphrase-manual | 0.8091 | 0.4995 | 0.6177 | 1.0000 | 0.6177 | 6 | 3 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8625 | 0.2141 | 0.3430 | 1.0286 | 0.3361 | 155 | 12 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7098 | 0.6201 | 0.6619 | 1.0417 | 0.6428 | 107 | 54 | — |
| Category: translation-auto | 0.6968 | 0.3432 | 0.4599 | 1.0000 | 0.4599 | 17 | 7 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 5 | 0 | — |

**Docs 81–308 (held-out tail, 113 plag / 115 clean):**

| Variant | Precision | Recall | F1 | Granularity | Plagdet | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2426 | 0.3330 | 0.2807 | 1.0461 | 0.2718 | 781 | 798 | 228 |
| **Plagiarised docs only, pooled** | **0.3674** | **0.3330** | **0.3494** | **1.0461** | **0.3382** | 781 | 527 | 113 |
| Conditional precision (firing docs only) | 0.2426 | — | — | — | — | — | 798 | 79 |
| Category: none (verbatim) | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 4 | 0 | — |
| Category: paraphrase-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 7 | 0 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8149 | 0.2435 | 0.3749 | 1.0794 | 0.3550 | 413 | 107 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7225 | 0.4989 | 0.5902 | 1.0238 | 0.5803 | 308 | 140 | — |
| Category: translation-auto | 0.5281 | 0.1772 | 0.2653 | 1.0000 | 0.2653 | 33 | 10 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 16 | 0 | — |

**Docs 1–308 (overall corpus, 156 plag / 152 clean):**

| Variant | Precision | Recall | F1 | Granularity | Plagdet | \|S\| | \|R\| | Docs |
|---|---|---|---|---|---|---|---|---|
| All docs pooled (incl. clean) | 0.2357 | 0.3440 | 0.2797 | 1.0428 | 0.2715 | 1071 | 1059 | 308 |
| **Plagiarised docs only, pooled** | **0.3516** | **0.3440** | **0.3477** | **1.0428** | **0.3374** | 1071 | 710 | 156 |
| Conditional precision (firing docs only) | 0.2357 | — | — | — | — | — | 1059 | 112 |
| Category: none (verbatim) | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 4 | 0 | — |
| Category: paraphrase-manual | 0.8091 | 0.2305 | 0.3588 | 1.0000 | 0.3588 | 13 | 3 | — |
| Category: paraphrase-auto-high (word-salad) | 0.8197 | 0.2355 | 0.3658 | 1.0683 | 0.3489 | 568 | 119 | — |
| Category: paraphrase-auto-low (synonym-swap) | 0.7190 | 0.5302 | 0.6103 | 1.0292 | 0.5978 | 415 | 194 | — |
| Category: translation-auto | 0.5976 | 0.2336 | 0.3359 | 1.0000 | 0.3359 | 50 | 17 | — |
| Category: translation-manual | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 21 | 0 | — |

In every range, the conditional precision row equals the all-docs pooled precision. That follows
from the formula, not from a calculation error: precision averages over individual detections, and
every detection sits in a document that fired at least once, so restricting to firing documents
leaves the set of detections unchanged.

Under the official metric the held-out tail (0.3382) scores slightly higher than the tuning subset
(0.3284), and both are close to the overall figure (0.3374). So the tuning subset is not an easy
subset that inflates the headline number, and the parameters carry over to documents that were
never used to choose them. The per-category scores follow a similar pattern in both ranges, with
synonym-swap (paraphrase-auto-low) the strongest (0.6428 on 1-80, 0.5803 on 81-308).

> **Correction (2026-09-28):** this paragraph originally attributed 11 "verbatim" cases in the
> 81-308 tail to `suspicious-document00126.txt` (4 cases) and `suspicious-document00228.txt` (7
> cases) combined, and separately claimed `suspicious-document00032.txt` (6 cases) was a third
> verbatim document detected in the 1-80 subset. Both claims were wrong, caused by the
> `classify_case_category` bug described above (see the "none (verbatim)" correction note
> earlier in this file). Only `00126` is actually verbatim
> (`type="artificial" obfuscation="none"`, 4 cases). `00032` and `00228` are both entirely
> `type="simulated"` (manual paraphrasing, 6 + 7 = 13 cases, matching the corpus-wide
> `paraphrase-manual` |S|=13). Corrected below.

**True verbatim ("none") has zero detections in every scope: 0/0 cases exist in 1-80, 0/4
detected in 81-308, 0/4 detected overall.** The only document with real verbatim ground truth,
`suspicious-document00126.txt` (4 cases, in the 81-308 tail), produced no detections at all
(`det_spans = 0` in the plagdet summary), which points to a Gate 1 retrieval block or an LLM
rejection of every candidate; this was not investigated further. Separately,
**paraphrase-manual scores 0.6177 on 1-80 but 0.0000 on 81-308** — the tail's 7 paraphrase-manual
cases are entirely in `suspicious-document00228.txt`, which also produced zero detections, while
the 1-80 subset's 6 paraphrase-manual cases (`suspicious-document00032.txt`) had 3 of them
detected. The project report should mention both gaps separately: in the untuned tail, both true verbatim
and manual-paraphrase categories have zero correct detections, while in the 1-80 tuning subset
manual-paraphrase is the second-strongest category (there are no verbatim cases at all in 1-80
to compare against).

## Update (2026-09-19): retrieval recall@20 data completeness fix

This section is about the retrieval stage, not plagdet. Recall@20 measures whether the true source
document appeared anywhere in the top 20 candidates before LLM confirmation runs. It is a diagnostic
for the retrieval branches (ESA, LSA, embeddings), computed separately from the official plagdet
formula and not used in it, since that formula only uses confirmed detections. Both appear in this
file because they come from the same 308-document run.

The saved retrieval recall results (behind the "Retrieval recall@20" row further down and Table
6.4) had only 257 of 308 rows. All 26 missing plagiarised documents were in the 81-308 tail, and
each was a document that failed Gate 1 on its real fusion score. Once Gate 1 rejects a document,
the retrieval code throws away the top-20 candidate list (`lookup_pipeline` returns only the top-1
score in that branch), so recall@20 was never recorded for these 26 documents. The values were
missing, not wrong.

All 26 documents contain ground-truth plagiarism spans (1 to 28 each, none are clean), so recall@20
could not default to a trivial value. They were rerun with Gate 1 turned off
(`--min-top1-score 0.0`, everything else unchanged) only to keep the full candidate list so recall@20
could be computed. This diagnostic rerun does not change the Gate 1 failure already recorded for
these 26 documents in the plagdet and analytics summaries.

| Range | Plag docs with data (before → after) | Retrieval recall@20 (before, partial) | **Retrieval recall@20 (after, complete)** |
|-------|----------------------------------------|-----------------------------------------|----------------------------------------------|
| 1-80 (tuning subset) | 43 to 43 (no gap) | 0.5604 | **0.5604** (unchanged) |
| 81-308 (held-out tail) | 87 to 113 | 0.6307 | **0.6028** |
| 1-308 (overall) | 130 to 156 | ~0.607 (see note below) | **0.5911** |

On their own, the 26 added tail documents average recall@20 = 0.5097, well below the 87 tail
documents that already had data. The earlier partial tail figure (0.6307) was too optimistic because
it left out the harder Gate-1-failed cases. Use the complete figures (0.6028 for the tail, 0.5911
overall) in Table 6.4 and wherever "Retrieval recall@20" is cited. The older 0.607 figure in the
original headline table below was computed on the incomplete 257/308 base and should not be quoted.

The merged recall data has 283 rows: 257 original and 26 backfilled.

### Retrieval recall@20 by subset (final, complete)

| Range | Plag docs | Docs with true source in top-20 | **Retrieval recall@20 (macro)** |
|-------|-----------|-----------------------------------|-------------------------------------|
| 1-80 (tuning subset) | 43 | 27/43 | **0.5604** |
| 81-308 (held-out tail) | 113 | 82/113 | **0.6028** |
| 1-308 (overall) | 156 | 109/156 | **0.5911** |

"Docs with true source in top-20" counts documents with recall@20 > 0, meaning at least one of the
document's true sources was in the top 20 candidates. A multi-source document gets a fractional
recall@20 when only some of its sources were retrieved, which is why the count and the macro
average are reported separately.

Recall@1 per subset and category was not computed. The retrieval recall output stores only the
single `recall_at_k` requested at run time (k=20 here), not the ranked candidate list, and the
scratch file with per-document candidate rankings is overwritten for every document. Recall@1 would
need a separate rerun with `--retrieval-recall-k 1`.

## Update (2026-09-21): detection and clean false-positive rates by scope, TP definition fix

Computed with `scripts/final/compute_detection_fp_rates.py` from the complete, regenerated plagdet
summary, without rerunning the pipeline. The codebase has two definitions of "detected", and they
disagree on the same 2 documents flagged twice above (`suspicious-document00010` and `00012`):

- `tp_chars > 0` (character-overlap TP) on the regenerated plagdet summary (308 rows, 00010 and
  00012 fixed) gives **85/156**.
- `tp > 0` (span-count TP) on the archived analytics summary (308 rows, but 00010 and 00012 are
  stale there, with `det_spans` of 0 and 9 instead of the corrected 5 and 4) gives the previously
  cited **87/156**.

The 87/156 figure in the project report came from the stale detection data for those 2 documents. It is not
a separate valid metric, only the same character-versus-span difference applied to uncorrected
data. **Replace 87/156 (55.8%) with 85/156 (54.5%) wherever it is cited.**

| Scope | Plag docs | Plag detected | Detection rate | Clean docs | Clean FP | FP rate |
|-------|-----------|----------------|-----------------|------------|----------|---------|
| 81-308 | 113 | 60 | 53.1% | 115 | 12 | 10.4% |
| 1-80 | 43 | 25 | 58.1% | 37 | 6 | 16.2% |
| 1-308 | 156 | 85 | 54.5% | 152 | 18 | 11.8% |

In every scope, plagiarised documents that fired plus clean documents with a false positive equals
the total number of firing documents: 33 for 1-80, 79 for 81-308 and 112 for 1-308, so the
arithmetic is consistent. The clean false-positive figure (18/152, 11.8%) matches the project report and
needed no correction.

The same script was run on the 10-document custom dataset, whose plagdet summary had already been
fully regenerated, so it has no stale documents:

| Scope | Plag docs | Plag detected | Detection rate | Clean docs | Clean FP | FP rate |
|-------|-----------|----------------|-----------------|------------|----------|---------|
| custom (10 docs) | 5 | 4 | 80.0% | 5 | 1 | 20.0% |

This matches the existing custom-dataset results (Table 6.9: 1 of 5 clean documents with a false
alarm). `suspicious-document00010` is the one plagiarised document with no detections, a known
Gate 1 / short-span failure and not a data completeness problem. The custom dataset's figures need
no correction.

## Headline results (original, 2026-06-15; superseded by the updates above)

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
| Retrieval recall@20 (plag only) | ~~0.607~~ **0.5911** (corrected 2026-09-19, see update above) |
| Plag docs detected (≥1 TP) | ~~87/156 (55.8%)~~ **85/156 (54.5%)** (corrected 2026-09-21, see update above) |
| Plag docs with zero detections | 63/156 (40.4%) |
| Clean FP docs | 18/152 (11.8%) |

†Precision is 0.0 for plagiarised documents with no detections (not 1.0), so missed sources do not
inflate the macro. Both macro precision (0.310) and recall (0.397) are pulled down by the 63/156
documents with no detections. On the 93 documents where the system fires, per-detection precision
is much higher (0.53–0.64 in the per-obfuscation breakdown).

## Per-obfuscation breakdown

| Category | plagdet | F1 | Precision | Recall |
|----------|---------|-----|-----------|--------|
| none (verbatim) | 0.516 | 0.516 | 0.645 | 0.430 |
| paraphrase-auto-low (synonym-swap) | **0.615** | 0.625 | 0.530 | 0.763 |
| paraphrase-auto-high (word-salad) | 0.334 | 0.337 | 0.413 | 0.284 |

## Runtime

### Per-document inference (query time)

- This session processed 45 new documents (docs ~213–312) in 6.4 hours (mean ~513 s/doc ≈ 8.5
  min, median ~435 s, slowest single doc 26.7 min).
- The preceding 212 docs loaded from cache instantly.
- Cumulative LLM processing for the full corpus: ~22 hours (≈16 h for the first 212 + 6.4 h for
  the remaining 45 fresh docs).
- If the ~8.5 min/doc mean measured on the 45 fresh docs is instead applied uniformly across all
  308 docs (i.e. as if none had been cached), that projects to ≈43.6 hours (8.5 min × 308 ≈ 2618
  min). This is a hypothetical upper-bound extrapolation, not the actual measured runtime — the
  real cumulative figure is ~22 h because the first 212 docs completed faster (cache hits from
  earlier sessions), and the slowest single document remains 26.7 min either way.

### One-time offline index build (over 11,093 source docs, shared by all queries)

| Build step | One-time cost |
|------------|---------------|
| Source embedding index (Qwen3-Embedding-0.6B, GPU) | ~44 h |
| TF-IDF char-hashing index | ~6 h |
| ESA index | ~2 h |
| LSA index | ~2 h |
| **Total offline indexing** | **~54 h** |

These costs are paid once and reused for every suspicious-document query. They do not grow with the
number of documents evaluated.

## Comparison with the 212-document run

| Metric | 212 docs (102 plag) | 308 docs (156 plag) |
|--------|---------------------|---------------------|
| Macro plagdet (plag only) | 0.359 | 0.328 |
| Granularity | 1.019 | 1.015 |
| Retrieval recall@20 (plag only) | 0.640 | 0.607 |

The small plagdet drop is expected, since the ~54 added plagiarised documents are a harder and more
representative sample than the earlier prefix. The obfuscation pattern is the same (synonym-swap
best, word-salad the ceiling), and granularity stays around 1.01, so the pipeline does not
over-fragment detections at scale.
