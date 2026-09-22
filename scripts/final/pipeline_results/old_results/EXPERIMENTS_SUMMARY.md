# Experiments Summary — PAN 2011 Plagiarism Detection Pipeline

This document summarises every experiment run during thesis development, in chronological
order. For each experiment: what was tested, what was found, and what decision was made.
Intended as a reference for writing the thesis chapters.

---

## System Overview

A three-stage pipeline for external plagiarism detection on PAN 2011:

```
Suspicious doc
      ↓
[Stage 1] Multi-branch Retrieval
  ESA + LSA + Sentence Embeddings (+ branch union top-3)
  → Fusion score → Gate 1 (min_top1_score = 0.60)
  → Gate 2 (relative_gap = 0.85) → top candidates
      ↓
[Stage 2] LLM Confirmation
  gemma4:26b, rubric prompt v3
  25 chunk pairs per candidate → score 0.00/0.25/0.50/0.85/0.95/1.00
  → Threshold 0.85 → confirmed source docs + spans
      ↓
[Stage 3] Char N-gram Aligner (experimental, not used in final results)
  Sliding window Jaccard char trigram similarity
  → Extends span coverage on confirmed source docs
```

**Dataset:** PAN 2011 — 300 suspicious documents, 11,093 source documents.
**Corpus processed:** 212 suspicious documents (run stopped at doc 212).
**Evaluation metric:** PAN plagdet = F1 / log2(1 + granularity), plus obfuscation breakdown.

---

## Experiment 1 — Baseline Gate 1=0.70

**Folder:** `70_gate1/`
**Docs:** 4 (early correctness test)
**Change:** Initial pipeline with Gate 1=0.70, pairs=15, no branch union.
**Finding:** Gate 1=0.70 too aggressive — doc00007 (top-1 score 0.715) barely passed.
**Decision:** Lower Gate 1 threshold.

---

## Experiment 2 — Gate 1=0.40

**Folder:** `40_gate1/`
**Docs:** 68
**Change:** Gate 1 lowered to 0.40, pairs=25, Gate 2=0.85.
**Finding:** 64% of plagiarised docs still blocked — but root cause was stale cache from
Run 1 contaminating results. 23 blocked docs had `retrieval_top1_score=0.0` (old cache).
Results unreliable due to cache mixing.
**Decision:** Clear cache, rerun with Gate 1=0.60.

---

## Experiment 3 — Gate 1=0.60

**Folder:** `60_gate1_80max/`
**Docs:** 69
**Change:** Gate 1=0.60, fusion max weight=0.80, fresh cache.

| Metric | Value |
|--------|-------|
| Detected | 25/37 (68%) |
| Macro F1 | 0.366 |
| Clean FP | 1/32 |
| Total FP spans | 143 |

**Finding:** Unlocked 12 more docs vs Run 2. But 3 problem docs drove most FPs:
- doc00049: 38 FP, 0 TP — LLM over-confirmed wrong sources
- doc00068: 38 FP, 0 TP — embeddings found correct source but fusion buried it
- doc00061: 6 FP, 0 TP — same as 00068

Root cause for 00061/00068: branch disagreement. Embeddings found correct source at
score 0.90 but LSA/ESA disagreed, dragging fusion score below Gate 2 window.
**Decision:** Add branch union to guarantee top-3 from each branch reaches LLM.

---

## Experiment 4 — Branch Union Top-3

**Folder:** `top3_gate1/`
**Docs:** 69
**Change:** Top-3 from each branch (ESA, LSA, Embeddings) added to candidate pool
regardless of fusion score.

| Metric | Value |
|--------|-------|
| Detected | 25/37 (68%) |
| Macro F1 | 0.380 |
| Clean FP | 1/32 |
| Total FP spans | 141 |

**Finding:** Fixed doc00061 (TP=0→4, F1=0→0.47). The correct source found by embeddings
at 0.90 was now guaranteed to reach the LLM via branch union.
Doc00068 still had 36 FP — correct source confirmed but 7 wrong sources also confirmed.
This is an LLM prompt problem, not a retrieval problem.
**Decision:** Fix LLM over-confirmation with a stricter prompt.

---

## Experiment 5 — Strict LLM Prompt

**Folder:** `strict_prompt/`
**Docs:** 69
**Change:** Prompt now explicitly states: topical similarity is NOT plagiarism, requires
≥3 pairs with verbatim/near-verbatim overlap before confirming.

| Metric | Value |
|--------|-------|
| Detected | 25/37 (68%) |
| Macro F1 | **0.659** |
| Retrieval recall@20 | 0.471 |
| Clean FP | 1/32 |

**Finding:** Biggest single improvement across all runs: +0.279 F1.
- doc00049: 38 FP → 0 FP ✅
- doc00068: 36 FP → 1 FP ✅ (correct source confirmed at 0.980)

**But new failure mode discovered:** Synonym-swap obfuscation (obf=low) now missed.
Docs 00005 and 00007 scored ~0.50 under strict prompt (below 0.95 threshold) because
synonym-swapped text doesn't look "verbatim" even though it IS plagiarism.
**Decision:** Design rubric prompt with explicit synonym-swap band at 0.85.

---

## Experiment 6 — Rubric Prompt v3 + gemma4:26b (80 docs)

**Folder:** `80_docs_promptv3/`
**Docs:** 80
**Changes:**
- Upgraded LLM: gemma4:e4b → gemma4:26b (MoE, 25.2B total / 3.8B active)
- New rubric prompt with discrete scoring bands:
  - 0.00 = topical similarity only
  - 0.25 = same-author reuse or weak overlap
  - 0.50 = partial overlap, insufficient
  - 0.85 = synonym-swap (coherent text, synonyms substituted, structure preserved)
  - 0.95 = near-verbatim
  - 1.00 = exact verbatim
- Threshold lowered: 0.95 → 0.85 to catch synonym-swap
- temperature=0, top_p=0.95, top_k=64, seed=42 for reproducibility

| Metric | Value |
|--------|-------|
| Detected | 29/43 (67%) |
| Macro F1 | 0.422 |
| Macro charF1 | 0.407 |
| Retrieval recall@20 (plag only) | 0.753 |
| Clean FP docs | 6/37 |

**Finding:** Rubric prompt fixed two Run 5 failures:
1. doc00005 and doc00007 (synonym-swap): now scored 0.850 and 0.950 → confirmed ✅
2. Same-author reuse (Samuel Pepys diary): now explicitly scored 0.25 → rejected ✅

**New failure mode discovered:** Word-salad FP pattern on obf=high docs (doc10, doc12, doc15).
Named entities survive scrambling → LLM sees entity overlap → wrongly fires 0.85 band.
Example: `"she take Braun'mho"` (incoherent) contains "Braun" → scored 0.850.
**Decision:** Attempt prompt v4 with coherence guard.

---

## Experiment 7 — Prompt v4 Coherence Guard — REJECTED

**Change:** Added rule: "if suspicious text is grammatically incoherent/word-salad,
score 0.25 at most — even if named entities match."

**Test on word-salad docs (doc10, doc12, doc15):** Correctly scored 0.25 ✅

**Regression tests:**
- doc05 (synonym-swap obf=low): v3=0.850 → v4=0.250 ❌ REGRESSION
- doc05 wrong (topical): v3=0.000 → v4=0.250 ❌ REGRESSION

**Decision: REJECTED.** LLM cannot reliably distinguish obfuscation=low (coherent
synonym-swap) from obfuscation=high (word-salad) via prompt rules alone. The model's
judgment of "grammatically coherent" is inconsistent at the boundary. Stayed on v3.

**Correct fix identified:** Perplexity pre-filter (code-level, before LLM) rather than
prompt-level instruction.

---

## Experiment 8 — 50 Pairs per Doc — REJECTED

**Change:** Increased chunk pairs sent to LLM from 25 → 50 per candidate.
**Test docs:** doc12 (4/14 GT spans), doc37, doc55.

**Result:** Doc12 at 50 pairs: new FP added (part13__source-document06435 scored 0.950)
that was correctly rejected at 25 pairs. More pairs = more opportunity for wrong confirmations.

**Decision: REJECTED.** More evidence didn't help the correct source; it introduced new
FPs. Reverted to 25 pairs.

---

## Experiment 9 — TF-IDF for Retrieval Misses — NEGATIVE RESULT

**Motivation:** 5 plagiarised docs had recall@20=0 — true source never in top-20.
Hypothesis: TF-IDF char n-gram retrieval might surface these sources.
**Test doc:** doc26 (obfuscation=high, retrieval miss).
**Result:** Gate 1 FAILED. TF-IDF score below 0.60 for true source.

**Finding:** All retrieval miss docs have obfuscation=high. Word-salad destroys char
n-gram signal as much as semantic signal. No retrieval method can match incoherent
scrambled text to coherent source text reliably. Unfixable retrieval ceiling for obf=high.

---

## Experiment 10 — PAN plagdet Metric Implementation

**File:** `scripts/final/compute_plagdet.py`
**Change:** Implemented official PAN 2011 plagdet formula:
- Detection requires: suspicious offset overlap + source offset overlap + same source_doc_id
- Granularity = mean detections per detected GT span (zero-detection spans excluded)
- plagdet = F1 / log2(1 + granularity)
- Bug fixed: macro average must be over plagiarised docs only (clean docs score 1.0 and
  inflate macro if included)

**212-doc results (official thesis numbers):**

| Metric | Value |
|--------|-------|
| Macro plagdet (plag docs only) | **0.359** |
| Macro F1 | 0.364 |
| Granularity | 1.019 |
| Micro plagdet | 0.304 |

| Obfuscation | plagdet | F1 | Recall | Precision |
|-------------|---------|-----|--------|-----------|
| low (synonym-swap) | **0.665** | 0.680 | 0.784 | 0.600 |
| none (verbatim) | 0.570 | 0.570 | 0.511 | 0.645 |
| high (word-salad) | 0.314 | 0.317 | 0.282 | 0.362 |

**Granularity = 1.019:** Near-perfect. Pipeline does not over-fragment detections.
Each plagiarised passage is reported as one clean merged span, not multiple overlapping
fragments. This is a positive thesis result.

---

## Experiment 11 — Char N-gram Aligner — NEGATIVE RESULT (corpus-level)

**File:** `scripts/final/char_ngram_aligner.py`
**Motivation:** After LLM confirms a source doc, some plagiarised passages are missed
because the embedding chunk pairs didn't align well. Char trigrams survive obfuscation
better than embeddings.

**Algorithm:**
- For each confirmed (suspicious, source) doc pair: slide 800-char window, stride 400
- Find best-matching source window via Jaccard char trigram similarity
- If Jaccard ≥ 0.12 and not already covered → add as new span
- Only extend the dominant source (most confirmed chars) to avoid extending FP sources

**Individual doc result (doc12):**
+4 new spans, all 4 hit GT regions not previously covered ✅
(GT spans covered: 4/14 → ~8/14 after aligner)

**Full corpus result (212 docs):**

| Metric | Base | + Aligner |
|--------|------|-----------|
| Macro plagdet | 0.359 | 0.193 ❌ |
| Granularity | 1.019 | 1.265 |
| Precision (obf=low) | 0.600 | 0.185 |
| Recall (obf=low) | 0.784 | 0.842 |

**Decision: NOT USED in final results.** 340 new spans added but precision collapsed
because the aligner extends FP-confirmed source docs (e.g. doc39: 46 FP spans → +10 more).

**Key thesis finding:** The aligner's effectiveness is bounded by LLM confirmation quality.
It works when the LLM confirmed the right source; it amplifies errors when the LLM confirmed
the wrong source. This is a fundamental limitation of any post-processing approach.

**Future fix:** Perplexity pre-filter to reduce LLM FPs before running aligner.

---

## Experiment 12 — Perplexity Pre-filter (GPT-2 small)

**File:** `scripts/final/run_pipeline.py` (`compute_perplexity`, `--perplexity-filter`)
**Motivation:** Failure Mode 2 — word-salad (obf=high) docs trick the LLM into firing the
0.85 band because named entities survive scrambling. A code-level coherence check before
the LLM should cap these.

**Algorithm:**
- GPT-2 small (~117M params) scores the perplexity of each candidate's suspicious chunk text.
- Per-pair scoring (not doc-level): each source candidate's matched suspicious chunks are
  scored individually — docs are mixed coherent + word-salad, so doc-level averaging fails.
- Majority vote: if ≥50% of a candidate's chunks have perplexity > threshold (250) → word-salad.
- Word-salad → cap that candidate's LLM score at 0.25 (below the 0.85 confirm threshold).

**Calibration:** coherent text scores ~54.6, word-salad scores ~3887.9. Threshold=250
cleanly separates the two classes.

**Per-doc verification (with `--run-embeddings`):**
- doc179 candidate `part22__source-document10586`: perplexity mean=454, frac_salad=0.90 →
  capped 0.95 → 0.25 → correctly rejected ✅
- doc102 candidate `part20__source-document09528`: perplexity mean=367, frac_salad=1.00 →
  capped 0.85 → 0.25 → correctly rejected ✅
- Coherent candidates (mean 39–68) passed through to the LLM unchanged ✅

**Decision:** KEEP. The filter behaves exactly as designed — it converts word-salad false
positives into rejections without touching coherent candidates. Corpus-level impact on the
full 212 docs still to be measured.

---

## Experiment 13 — Soft Gate 1 (branch-confidence + perplexity rescue)

**Files:** `scripts/final/04_source_retrieval/source_retrieval_branches.py`,
`scripts/final/run_pipeline.py` (`--soft-gate1`, `--soft-gate1-min-branch`)
**Motivation:** Failure Mode 3 — plagiarised docs blocked at Gate 1 because branches
disagree: one branch finds the correct source at high confidence (e.g. EMB=0.84) but the
others (especially ESA) drag the fusion score below 0.60. The branch union runs *after*
Gate 1, so these docs never reach the LLM at all.

**Algorithm:** When fusion < `min_top1_score` (0.60), allow a *soft pass* if:
1. the best single-branch score ≥ `soft_gate1_min_branch` (default 0.50), AND
2. the suspicious text is coherent (perplexity check, not word-salad).

This rescues coherent docs where one branch is confident, while still blocking word-salad.

**Design iteration:** First version required fusion ≥ 0.50 (a "soft floor") as well. But
fresh runs without TF-IDF put the borderline docs at fusion 0.41–0.45 — below that floor —
so the soft pass never triggered (verified on doc10 and doc179). The soft floor was removed;
the gate now keys off best-branch confidence + perplexity only.

**Per-doc verification (with `--run-embeddings`):**
- doc179: fusion=0.5099, best_branch=0.8424, perplexity=coherent → SOFT PASS ✅
  (reached LLM; true source still a retrieval miss — recall@20=0.50 — so F1=0, an
  obf=high retrieval-ceiling problem, not a gate problem)
- doc102: fusion=0.5961, best_branch=0.7989, perplexity=coherent → SOFT PASS ✅
  (reached LLM; recall@20=1.00 — true source retrieved — but the LLM scored its chunk
  pairs 0.00 and confirmed nothing; see Experiment 14 motivation)

**Decision:** KEEP as an opt-in flag. It correctly unblocks coherent borderline docs and
correctly refuses word-salad ones. It does not, on its own, recover true positives — the
remaining bottleneck is the *quality of the chunk pairs* the LLM scores, not gate access.
Corpus-level impact still to be measured.

---

## Experiment 14 — Cross-encoder Reranking

**Files:** `scripts/final/04_source_retrieval/source_retrieval_branches.py`
(`rerank_with_cross_encoder`), `scripts/final/run_pipeline.py` (`--cross-encoder`)
**Motivation:** Experiments 12–13 showed the gates can be opened, but two structural
problems remain:
1. The chunk pairs sent to the LLM always come from the **embedding** branch. Even when
   LSA/ESA votes the correct source into the candidate list, the LLM gets embedding-derived
   pairs — weak evidence if embeddings didn't align that source well (doc102: true source
   retrieved but LLM scored its pairs 0.00).
2. The fusion score averages three **bi-encoders** that score texts independently. A
   cross-encoder reads suspicious + source text *together*, giving a much stronger relevance
   signal, especially for synonym-swap (obf=low) obfuscation.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB).

### Attempt A — Whole-document pairing — REJECTED

**Algorithm:** Runs after fusion + branch union, on the full candidate pool (~20–29 docs).
For each candidate: concatenate all source chunks into one doc text (truncated to 3000 chars),
pair with truncated suspicious text, score with the cross-encoder, sigmoid → `ce_score`.

**Result on doc102 (true source in pool, recall@20=1.00):**
```
[CE] Top-3 after rerank: part14 (0.016), part4 (0.000), part22 (0.000)
```
All candidates collapsed to ~0.00. The true source (`part22__source-document10513`) scored
0.000; a wrong source (`part14`) "won" at 0.016 — pure noise.

**Why it failed:** `ms-marco-MiniLM-L-6-v2` is trained for **query→passage** relevance
(short query, single short passage, 512-token limit). Feeding it **document→document**
(3000 chars each) is out of distribution — it only sees the first ~512 tokens and has no
notion of "do these two long documents share a plagiarised passage." Scores degenerate to ~0.

**Decision: REJECTED.** Whole-doc pairing is the wrong granularity for this model.

### Attempt B — Chunk-pair pairing — NEGATIVE BUT DIAGNOSTIC

**Algorithm:** Score short chunk pairs (suspicious chunk vs source chunk, ≤400 chars each,
top-40 embedding-ranked pairs per source) — exactly what ms-marco is trained for. Each
candidate source's rerank score = the MAX cross-encoder score over its best chunk pairs.
Pair pool comes from the embedding candidates parquet (already-aligned chunk pairs).

**Result on doc102 (true source `part22` in pool, recall@20=1.00):**
```
[CE] Top-3 (max chunk-pair score): part14 (0.818), part20 (0.032), part22-TRUE (0.023)
```
Unlike Attempt A, the scores are now **discriminative** (0.818 vs 0.03) — the chunk-pair
granularity fixed the model-level problem. But the cross-encoder ranked a **wrong** source
(`part14`) top at 0.818 and the **true** source (`part22`) dead last at 0.023. The LLM
independently agreed: it scored part22's pairs ~0 and called part14 "unrelated fragments."

**Diagnosis (the key finding):** Every independent signal — embeddings, fusion,
cross-encoder, and the LLM — agrees the true source's *embedding-aligned chunk pairs* do
not look like plagiarism. The GT confirms part22 contains the plagiarised passage, so the
failure is **upstream in chunk alignment**: the embedding branch never paired the
suspicious plagiarised chunk to the correct part22 source chunk — it paired unrelated
chunks. A reranker can only reorder the pairs it is handed; it cannot create the missing
correct pair.

**Decision: NOT USED in final results.** The bottleneck is not retrieval (source doc IS
retrieved), not gating (soft Gate 1 lets it through), not confirmation judgment, and not
the reranker — it is **chunk-pair alignment**. The cross-encoder is valuable as a *diagnostic*
that localises the failure, but it does not recover the true positive.

**This marks the limit of what the current local stack can achieve.** Fixing it requires
either (a) a frontier confirmation LLM that can recognise obfuscated plagiarism from weaker
evidence, or (b) a fundamentally better chunk-alignment stage that surfaces the correct
suspicious↔source pairs in the first place. Both are listed in Future Work.

---

## Final Pipeline Configuration (Thesis Results)

```
Stage 1 — Retrieval
  ESA + LSA + Sentence Embeddings
  Branch union: top-3 from each branch guaranteed
  Gate 1: min_top1_score = 0.60
  Gate 2: relative_gap = 0.85
  Top-N: 20 candidates

Stage 2 — LLM Confirmation
  Model: gemma4:26b (Mixture of Experts)
  Sampling: temperature=0, top_p=0.95, top_k=64, seed=42
  Prompt: v3 rubric (discrete bands 0.00/0.25/0.50/0.85/0.95/1.00)
  Pairs per candidate: 25
  Threshold: 0.85
```

**Final results (full 308-doc corpus, PAN 2011):**

| Metric | Value |
|--------|-------|
| Documents evaluated | 308 (156 plag, 152 clean) |
| Detected (≥1 TP) | 87/156 (55.8%) |
| Macro F1 (plag only) | 0.333 |
| Retrieval recall@20 (plag only) | 0.607 |
| Clean FP docs | 18/152 (11.8%) |
| **Macro plagdet (plag only)** | **0.328** |
| Granularity | 1.015 |
| Micro plagdet | 0.323 |

**Per-obfuscation breakdown (308 docs):**

| Category | plagdet | Precision | Recall |
|----------|---------|-----------|--------|
| none (verbatim) | 0.516 | 0.645 | 0.430 |
| paraphrase-auto-low (synonym-swap) | **0.615** | 0.530 | 0.763 |
| paraphrase-auto-high (word-salad) | 0.334 | 0.413 | 0.284 |

**Runtime:** the full corpus required ~22 hours of cumulative LLM processing (~16 h for
the first 212 docs, then 6.4 h for the remaining 45 fresh docs at ~8.5 min/doc mean;
slowest single doc 26.7 min). The earlier 212-doc subset scored macro plagdet 0.359; the
slight drop to 0.328 on the full set reflects the harder, more representative tail of
additional plagiarised documents. The obfuscation pattern and near-unity granularity are
unchanged, confirming the pipeline scales without over-fragmenting detections.
(Full breakdown: `308_docs_full_v3/`.)

> **Update (2026-09-16):** the numbers above were computed before 2/308 docs' detection
> cache had been overwritten by later runs. They have since been regenerated (full 308
> docs), and the **official Potthast et al. corpus-pooled plagdet** has been added
> alongside the existing macro/micro figures. See the "Update" section at the top of
> `308_docs_full_v3/README.md` for the corrected macro/micro values (0.3342/0.3253),
> the official value (**0.3374**, plag-only pooling), and the official per-category
> breakdown, which reorders the categories (verbatim drops from best to worst
> non-zero category under per-case pooling).
>
> **Update (2026-09-19):** the "Retrieval recall@20 (plag only) = 0.607" figure above
> was also computed over an incomplete base (only 257/308 docs had retrieval-recall
> data recorded; the missing 26 were all Gate-1-failed docs in the 81-308 tail, which
> score worse on average). Backfilled and corrected: **recall@20 = 0.5911** overall
> (156/156 plag docs), with a per-subset breakdown (0.5604 on docs 1-80, 0.6028 on
> docs 81-308) in the second "Update" section of `308_docs_full_v3/README.md`. This is
> a retrieval-stage diagnostic, independent of the plagdet correction above.
>
> **Update (2026-09-21):** the "Detected (≥1 TP) = 87/156" figure above also used the
> stale, pre-regeneration detection counts for docs 00010/00012 (span-count TP from the
> archived `analytics_summary.parquet`, not yet corrected for those 2 docs). Recomputed
> from the current, regenerated `plagdet_summary.parquet` (character-overlap TP):
> **85/156 (54.5%)**. The clean-FP figure (18/152, 11.8%) is unaffected and unchanged.
> Full per-scope breakdown (1-80, 81-308, 1-308) is in the third "Update" section of
> `308_docs_full_v3/README.md`.

---

## Failure Mode Analysis

### Failure Mode 1 — Word-salad retrieval miss (33 docs)
**Cause:** obfuscation=high text is machine-scrambled (word-salad). Incoherent text has
no semantic signal for ESA/LSA/Embeddings or lexical signal for TF-IDF to match against
the coherent source. True source never appears in top-20.
**Fix:** Cross-encoder re-ranking after fusion (reads both texts together, more accurate
than bi-encoder comparison). May rescue some cases where source is at rank 21–30.
**Ceiling:** Some obf=high cases are fundamentally unmatchable — even top PAN 2011 teams
scored ~0.30 on this category.

### Failure Mode 2 — Word-salad LLM FP (obf=high named entity FPs)
**Cause:** Named entities survive word-salad scrambling (e.g. "Braun", "Pontresina").
LLM sees entity overlap in incoherent text and fires the 0.85 synonym-swap band.
**Fix:** Perplexity pre-filter (GPT-2 small, ~117M params, runs locally). Score suspicious
text coherence before LLM — high perplexity = word-salad → cap max score at 0.25.
**Impact:** Would reduce clean FP docs and improve precision on obf=high.

### Failure Mode 3 — Gate 1 blocked plagiarised docs (15 docs)
**Cause:** Fusion score < 0.60 for true source. All branches agree the source is
unlikely — genuine retrieval failure, not just ranking issue.
**Fix:** Lower Gate 1 threshold increases recall but increases FPs. Cross-encoder
re-ranking would give a better signal to base the threshold on.

---

## What Worked vs What Didn't

| Experiment | Result | Why |
|------------|--------|-----|
| Branch union top-3 | ✅ +F1 | Guarantees correct source reaches LLM when branches disagree |
| Strict prompt | ✅ +0.279 F1 | Eliminated LLM over-confirmation on topical similarity |
| Rubric prompt v3 | ✅ Catches synonym-swap | Explicit band for obf=low; same-author reuse correctly rejected |
| gemma4:26b | ✅ Better judgment | Wider MoE expert pool vs smaller gemma4:e4b |
| Prompt v4 coherence guard | ❌ Regression | LLM can't reliably judge coherence at obf=low/high boundary |
| 50 pairs per doc | ❌ More FPs | More evidence = more confirmation opportunities for wrong sources |
| TF-IDF for retrieval misses | ❌ No signal | Word-salad destroys lexical signal same as semantic signal |
| Char n-gram aligner | ❌ Corpus-level | Amplifies LLM FPs; works on individual good docs only |
| Perplexity pre-filter | ✅ Works as designed | Cleanly separates word-salad (caps at 0.25) from coherent text |
| Soft Gate 1 | ⚠️ Opens gate, no TP | Correctly unblocks coherent borderline docs, but bottleneck is downstream |
| Cross-encoder (whole-doc) | ❌ Out of distribution | ms-marco is query→passage; doc→doc scores collapse to ~0 |
| Cross-encoder (chunk-pair) | ⚠️ Diagnostic | Discriminative scores, but localises failure to chunk alignment — no TP recovery |

---

## The Local-Hardware Ceiling (Key Conclusion)

Experiments 12–14 systematically eliminated every stage as the bottleneck for the hardest
borderline doc (doc102, obf=low, true source retrieved at recall@20=1.00):

- **Retrieval** — true source IS in the candidate pool ✅
- **Gating** — soft Gate 1 lets it through ✅
- **Word-salad FPs** — perplexity filter correctly caps them ✅
- **Confirmation judgment** — LLM and cross-encoder *independently agree* the evidence is weak

The failure localises to **chunk-pair alignment**: the embedding branch never paired the
genuinely-plagiarised suspicious chunk with the correct source chunk. Every downstream
component (fusion, cross-encoder, LLM) faithfully scored the wrong pairs it was handed.
No reranking or gating change can fix a missing correct pair.

This is the ceiling of the current local stack (cached embeddings + gemma4:26b + small
HF models on CPU/single GPU). Further gains require either a stronger confirmation model
or a fundamentally better alignment stage — both budget/hardware-gated (see Future Work).

---

## Future Work (Prioritised)

1. **Frontier confirmation LLM** (highest impact, API-budget-gated)
   Swap gemma4:26b for a Claude / GPT-4-class API at the confirmation stage. A local 26B
   MoE is the ceiling on recognising obfuscated plagiarism from weak/partial evidence.
   No better PC needed — only API budget. Most likely single change to move plagdet.

2. **Better chunk alignment for the LLM** (high impact, medium effort)
   Decouple the LLM's evidence pairs from the embedding branch alone. Assemble candidate
   chunk pairs from multiple aligners (char n-gram + embedding + cross-encoder) so a source
   is not penalised because one aligner missed its plagiarised passage. Directly targets the
   doc102 root cause (Experiment 14).

3. **Fine-tuned cross-encoder on PAN pairs** (high impact, GPU-VRAM-gated)
   ms-marco is off-domain (web search relevance). Train a cross-encoder on PAN 2011
   obfuscated↔source chunk pairs for a proper in-domain rerank signal. Needs more VRAM
   than the current setup for training.

4. **Obfuscation-aware retrieval for obf=high** (research-scale)
   The ~33 fully-missed word-salad docs need a different signal (fuzzy/edit-distance
   matching, or a model trained on scrambled text). Even top PAN 2011 teams scored ~0.30
   on this category — likely a fundamental ceiling, not a tweak.

5. **Larger local model** (medium impact, hardware-gated)
   gemma4:26b → 70B-class would improve confirmation but needs much more VRAM.

6. **Complete 300-doc run** (no code change, just compute)
   Docs 213–300 still unprocessed. Same pipeline, same config.

7. **Custom dataset** (medium effort)
   Small annotated dataset to show the pipeline generalises beyond PAN 2011.
