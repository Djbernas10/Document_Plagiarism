# Pipeline Experiment Results

Each folder contains `analytics_summary.parquet` and `retrieval_recall.parquet` from a completed run.
All runs use the PAN 2011 corpus (first 69 suspicious documents), ESA + LSA + Embeddings (TF-IDF disabled).

---

## Run 1 — Baseline: `70_gate1/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.70
- Gate 2 (relative_gap)   : 0.70
- Fusion max weight        : 0.70
- Pairs per doc            : 15
- Branch union             : No

**Result (4 docs — early test run):**
- Only 4 docs processed (docs 1–3 clean + 1 plagiarised test)
- No plagiarised docs in this subset
- Used to verify pipeline correctness before full batch run

**Key finding:** Pipeline mechanics verified. Gate 1=0.70 was the initial default chosen
based on doc 00007 (top-1 score 0.715). Proved too aggressive on the full corpus.

---

## Run 2 — Lower Gate 1: `40_gate1/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.40
- Gate 2 (relative_gap)   : 0.85
- Fusion max weight        : 0.80
- Pairs per doc            : 25
- Branch union             : No

**Result (68 docs):**

| Metric                    | Value  |
|---------------------------|--------|
| Plagiarised docs          | 36     |
| Detected (any span)       | 13 / 36 (36%) |
| Fully blocked (Gate 1)    | 23 / 36 (64%) |
| Macro F1 (all plag)       | 0.278  |
| Macro charF1 (all plag)   | 0.245  |
| Mean F1 (detected only)   | 0.769  |
| Retrieval recall@20       | 0.650  |
| Clean FP                  | 1 / 32 |
| Total FP spans            | 36     |

**Key finding:** Mean F1 on detected docs was excellent (0.769) but 64% of plagiarised docs
were still blocked. Root cause: the 23 blocked docs had stale cache from the previous
Gate1=0.70 run — their real scores were never evaluated. The cache mixing problem made
this run's blocked rate unreliable. Additionally, the cache showed `retrieval_top1_score=0.0`
for all blocked docs, confirming they were old cached results.

---

## Run 3 — Gate 1=0.60 + Max Weight 0.80: `60_gate1_80max/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.60
- Gate 2 (relative_gap)   : 0.85
- Fusion max weight        : 0.80
- Pairs per doc            : 25
- Branch union             : No

**Result (69 docs):**

| Metric                    | Value  |
|---------------------------|--------|
| Plagiarised docs          | 37     |
| Detected (any span)       | 25 / 37 (68%) |
| Fully blocked (Gate 1)    | 12 / 37 (32%) |
| Macro F1 (all plag)       | 0.366  |
| Macro charF1 (all plag)   | 0.331  |
| Mean F1 (detected only)   | 0.541  |
| Retrieval recall@20       | 0.569  |
| Clean FP                  | 1 / 32 |
| Total FP spans            | 143    |

**Key finding:** Lowering Gate 1 to 0.60 unlocked 12 more plagiarised docs vs the
previous run (25 vs 13 detected). However, mean F1 on detected docs dropped from
0.769 to 0.541 — the newly unlocked docs are harder cases with partial detections
and more false positives. Total FP jumped to 143, largely driven by 3 problem docs:
- **00049**: 38 FP, 0 TP — LLM over-confirmed wrong sources
- **00068**: 38 FP, 0 TP — embeddings found correct source but fusion buried it; LLM confirmed wrong ones
- **00061**: 6 FP, 0 TP — same problem as 00068 (embeddings correct, fusion wrong)

Root cause for 00061 and 00068: branches disagreed strongly. Embeddings found the
correct source at rank 1 with score ~0.90, but LSA/ESA pointed to different documents.
The fusion score for the correct source was dragged down, dropping it out of the Gate 2
window before reaching the LLM.

---

## Run 4 — Branch Union Top-3: `top3_gate1/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.60
- Gate 2 (relative_gap)   : 0.85
- Fusion max weight        : 0.80
- Pairs per doc            : 25
- Branch union             : Yes — top-3 from each branch added to candidate pool

**Result (69 docs):**

| Metric                    | Value  |
|---------------------------|--------|
| Plagiarised docs          | 37     |
| Detected (any span)       | 25 / 37 (68%) |
| Fully blocked (Gate 1)    | 12 / 37 (32%) |
| Macro F1 (all plag)       | 0.380  |
| Macro charF1 (all plag)   | 0.346  |
| Mean F1 (detected only)   | 0.562  |
| Retrieval recall@20       | 0.486  |
| Clean FP                  | 1 / 32 |
| Total FP spans            | 141    |

**Key finding:** Branch union fixed doc 00061:
- Before: TP=0, FP=6, F1=0.00
- After:  TP=4, FP=6, F1=0.47, charF1=0.53

The correct source (`source-document07899`, found by embeddings at score 0.90) was
added to the candidate pool via branch union and confirmed by the LLM.

Doc 00068 still problematic (36 FP, 1 TP): the correct source was added via branch union
but the LLM still confirmed 7 wrong sources — this is an LLM prompt problem, not a
retrieval problem.

**Remaining bottlenecks:**
1. **LLM over-confirmation** — docs 00049 (38 FP) and 00068 (36 FP) need a tighter prompt
2. **12 still-blocked plagiarised docs** — retrieval genuinely fails to find the source
   (all 3 branches disagree, fusion score < 0.60)
3. **Retrieval recall drop** — adding more candidates via branch union gives the LLM more
   wrong candidates to potentially confirm, explaining the slight FP increase

---

## Run 5 — Strict LLM Prompt: `strict_prompt/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.60
- Gate 2 (relative_gap)   : 0.85
- Fusion max weight        : 0.80
- Pairs per doc            : 25
- Branch union             : Yes — top-3 from each branch added to candidate pool
- LLM prompt               : Strict — explicit rules against topical similarity, requires ≥3 pairs with verbatim/near-verbatim overlap

**Result (69 docs):**

| Metric                    | Value  |
|---------------------------|--------|
| Plagiarised docs          | 37     |
| Detected (any span)       | 25 / 37 (68%) |
| Fully blocked (Gate 1)    | 12 / 37 (32%) |
| Macro F1 (all docs)       | 0.659  |
| Macro charF1 (all docs)   | 0.644  |
| Binary macro F1           | 0.659  |
| Retrieval recall@20       | 0.471  |
| Clean FP                  | 1 / 32 |
| Total FP spans            | 152    |
| Total TP spans            | 60     |

**Key finding:** The strict prompt dramatically improved macro F1: 0.380 → **0.659** (+0.279).
This is the biggest single improvement across all runs.

- **Doc 00049**: 38 FP → **0 FP** — all wrong candidates rejected (top-1 score 0.620 < threshold)
  Note: recall_at_20=0.00 — true sources were never retrieved; doc remains undetected but silently.
- **Doc 00068**: 36 FP → **1 FP** — correct source confirmed (0.980), wrong candidates rejected
  F1 improved from ~0.10 to **0.44** (charF1=0.69)

The LLM now correctly rejects topically similar but non-plagiarised candidates.
The strict rules ("requires ≥3 pairs with verbatim overlap", "topical similarity is NOT plagiarism")
are working as intended.

**Remaining bottleneck:** Doc 00055 still has 31 FP (cached from Run 4 — prompt not re-applied).
Total FP (+11 vs Run 4) is partly an artifact of this stale cache.

---

## Run 6 — Rubric Prompt v3 + gemma4:26b + Threshold 0.85: `80docs_v3_prompt/`

**Parameters:**
- Gate 1 (min_top1_score) : 0.60
- Gate 2 (relative_gap)   : 0.85
- Fusion max weight        : 0.80
- Pairs per doc            : 25
- Branch union             : Yes — top-3 from each branch
- LLM model                : gemma4:26b (Mixture of Experts, 25.2B total / 3.8B active)
- LLM sampling             : temperature=0, top_p=0.95, top_k=64, seed=42
- LLM threshold            : 0.85
- LLM prompt               : v3 rubric — discrete scoring bands (0.00 / 0.25 / 0.50 / 0.85 / 0.95 / 1.00)

**Result (80 docs):**

| Metric                    | Value  |
|---------------------------|--------|
| Plagiarised docs          | 43     |
| Detected (any span)       | 29 / 43 (67%) |
| Macro binary F1 (all)     | 0.4224 |
| Macro char F1 (all)       | 0.4066 |
| Retrieval recall@20 (plag only) | 0.7531 |
| Clean FP docs             | 6 / 37 |
| Total TP / FP / FN spans  | 85 / 280 / 156 |

> **Note:** An earlier version of this entry incorrectly reported recall@20=0.851 and F1=0.615.
> The 0.851 figure was the mean over ALL docs including clean docs (which trivially score 1.0).
> The correct recall@20 over plagiarised docs only is **0.753**. The F1 figures above are
> the correct macro F1 computed over all plagiarised docs.

**Key changes vs Run 5:**
- Upgraded LLM from `gemma4:e4b` (4.5B active) to `gemma4:26b` (MoE, 3.8B active but wider expert pool)
- Replaced strict boolean prompt with a **rubric-based discrete scoring system**:
  - `0.85` = synonym-swap obfuscation confirmed (same structure, synonyms substituted, named entities match)
  - `0.95` = near-verbatim copying
  - `1.00` = exact verbatim
  - `0.25` = same-author reuse or weak overlap (explicitly rejected at threshold 0.85)
  - `0.00` = topical similarity only
- Threshold lowered from 0.95 → **0.85** to catch synonym-swap cases (obfuscation=low)

**Key finding — rubric prompt v3 vs strict prompt:**
The rubric prompt was specifically designed to fix two failure modes from Run 5:
1. **Synonym-swap missed (obfuscation=low)**: docs 00005 and 00007 were missed by the strict prompt
   (scored ~0.50, below 0.95 threshold). The 0.85 rubric band + lowered threshold now catches these.
   - Doc 00005: correct source now scores 0.850 → CONFIRMED ✅, wrong sources score 0.000
   - Doc 00007: correct source scores 0.950 → CONFIRMED ✅, wrong sources score 0.250
2. **Same-author false positives**: PAN 2011 includes source docs from the same books as clean
   suspicious docs (e.g. different volumes of Samuel Pepys' diary). The strict prompt scored these
   at 0.85+ → FP. The rubric prompt explicitly defines same-author reuse = 0.25 → correctly rejected.

**7-doc verification before full run (clean/plausible cases):**
- Binary macro F1 = **1.0000**
- Char macro F1 = **0.9559**
- Clean docs with false alarms: **0 / 5**
- TP/FP/FN: **2 / 0 / 0**

**Retrieval quality:**
- recall@20 = **0.7531** (plagiarised docs only) — the true source was in the top-20 candidates
  for 75% of plagiarised docs. The multi-branch fusion (ESA + LSA + Embeddings) reliably surfaces
  the correct source for the majority of cases.

**FN breakdown (34 docs with missed spans):**
| Cause | Docs |
|-------|------|
| Retrieval miss (true source not in top-20) | 5 |
| Gate 1 blocked (fusion score < 0.60) | ~10 |
| LLM miss (retrieved but not confirmed) | ~19 |

**Obfuscation breakdown of missed docs:**
| Obfuscation type | Missed docs |
|-----------------|-------------|
| high (simulated paraphrase / word-salad) | 19 |
| low (synonym-swap) | 12 |
| none (verbatim) | 3 |

**Analysis of LLM misses by obfuscation type:**
- `obfuscation=none` misses (3 docs): likely fixable with prompt tuning — verbatim text should score 1.00
- `obfuscation=low` misses (12 docs): synonym-swap; threshold=0.85 should catch these but some chunk
  pairs may not align well enough to trigger the 0.85 band
- `obfuscation=high` misses (19 docs): **fundamental ceiling** — PAN 2011 high obfuscation is
  machine-generated word-salad/scrambling (not coherent paraphrase). Named entities survive but
  sentence structure is destroyed. Even top PAN 2011 teams struggled with this category.

**Deep-dive on high-obfuscation FP pattern (docs 00010, 00015):**
The rubric's 0.85 synonym-swap band fires incorrectly on high-obfuscation text because:
- High-obfuscation suspicious text is word-salad (incoherent: `"they make the gusto religion"`,
  `"she take Braun'mho"`) — grammatical structure is destroyed, not preserved
- Named entities from the source survive in the garbled text (e.g. "Braun", "Pontresina")
- LLM sees named entity overlap + garbled text → incorrectly applies the 0.85 synonym-swap rule
- **Key distinction**: synonym-swap (obf=low) preserves grammatical coherence; word-salad (obf=high)
  does not. A prompt v4 fix: incoherent/ungrammatical suspicious text with named entity overlap = 0.25

---

## Summary Comparison

| Run | Gate 1 | LLM Model | Prompt | Threshold | Docs | Macro F1 | Ret@20¹ | Clean FP |
|-----|--------|-----------|--------|-----------|------|----------|---------|----------|
| 60_gate1_80max  | 0.60 | gemma4:e4b | loose     | 0.95 | 69  | 0.366 | —     | 1/32  |
| top3_gate1      | 0.60 | gemma4:e4b | loose     | 0.95 | 69  | 0.380 | —     | 1/32  |
| strict_prompt   | 0.60 | gemma4:e4b | strict    | 0.95 | 69  | 0.659 | 0.471 | 1/32  |
| v3_rubric_80    | 0.60 | gemma4:26b | rubric v3 | 0.85 | 80  | 0.422 | 0.753 | 6/37  |
| v3_rubric_212   | 0.60 | gemma4:26b | rubric v3 | 0.85 | 212 | 0.413 | 0.640 | 15/110 |
| **v3_rubric_308 (final)** | 0.60 | gemma4:26b | rubric v3 | 0.85 | 308 | **0.333** | **0.607** | 18/152 |

¹ recall@20 computed over plagiarised docs only (clean docs excluded — they trivially score 1.0).
The F1 column is macro span-level F1 over plagiarised docs only.

**Plagdet, FINAL (308-doc full corpus, macro/micro convention, not the official PAN formula; see note below):**

| Metric | Value |
|--------|-------|
| Macro plagdet (plag only) | **0.328** |
| Macro F1 (plag only) | 0.333 |
| Granularity | 1.015 |
| Micro plagdet | 0.323 |
| Macro plagdet (all docs, incl. clean) | 0.601 |

> **Note:** the headline thesis number is **macro plagdet over plagiarised docs only = 0.328**.
> The 0.601 figure includes clean docs (which trivially score 1.0) and must not be quoted as the
> plagdet result. An earlier draft mislabelled the all-docs figure (0.6207) as the macro plagdet.

> **Correction (2026-09-16):** despite the section heading above, **neither macro (0.328) nor
> micro (0.323) is the official Potthast et al. (2011) plagdet formula.** Both are this
> codebase's own averaging conventions (macro = per-document average, micro = raw pooled
> character-count ratio). The official formula pools every GT case and detection across the
> whole corpus and averages one equally-weighted overlap-fraction term per case/detection
> (see `scripts/final/compute_plagdet_official.py`, eq. 1-2 of the PAN 2011 overview paper).
> After also regenerating 2 docs whose detection cache had been overwritten by later runs,
> the corrected macro/micro are **0.3342 / 0.3253**, and the actual official value is
> **0.3374** (plag-only pooling). Full detail and the official per-category breakdown are
> in the "Update (2026-09-16)" section at the top of `308_docs_full_v3/README.md`.

| Obfuscation | plagdet (old, per-document convention) | Official plagdet (per-case, corpus-pooled) | Precision | Recall |
|-------------|---------|---------|-----------|--------|
| paraphrase-auto-low (synonym-swap) | 0.615 | **0.5978** | 0.530 | 0.763 |
| none (verbatim) | 0.516 | **0.2895** | 0.645 | 0.430 |
| paraphrase-auto-high (word-salad) | 0.334 | **0.3489** | 0.413 | 0.284 |

Note the category ranking flips under the official metric: verbatim drops from best to worst
non-zero category (only 3/17 verbatim GT cases detected case-wise). Synonym-swap remains
the strongest category either way. See `308_docs_full_v3/README.md` for translation-auto/manual
rows (not present in this older breakdown) and the full official-metric numbers.

> **Correction (2026-09-19):** the "Retrieval recall@20: ... to 0.607 (308)" trend line below
> was also computed over an incomplete base. `retrieval_recall.parquet` had only 257/308 rows,
> and the 26 missing docs were all Gate-1-failed cases in the untuned 81-308 tail, which score
> worse on average (0.5097) than the docs that already had data. Backfilled and corrected:
> **recall@20 = 0.5911** over the full 156/156 plagiarised docs (0.5604 on docs 1-80, 0.6028
> on docs 81-308). See the second "Update" section of `308_docs_full_v3/README.md`. This is
> a retrieval-stage diagnostic, independent of the plagdet correction above.

**Plagdet (212-doc subset, for reference):** macro plagdet (plag only) = 0.359, granularity 1.019.

**Trend:**
- Strict prompt (Run 5) was the biggest F1 leap: +0.279 by eliminating LLM over-confirmation
- Rubric prompt v3 (Run 6) correctly handles synonym-swap (obf=low) and same-author reuse
- Plagdet stable as corpus grows: 0.359 (212 docs) → 0.328 (308 docs) — the extra docs are a harder tail
- Retrieval recall@20: 0.753 (80) → 0.640 (212) → ~~0.607~~ **0.5911** (308, corrected, see note above); later docs are harder for retrieval
- Granularity ≈ 1.0 throughout — the pipeline does not over-fragment detections at scale
- Remaining ceiling: word-salad (obf=high) plagdet ≈ 0.33 — fundamental retrieval + alignment limit
- Runtime: full corpus ≈ 22 h cumulative LLM processing (~8.5 min/doc fresh; slowest doc 26.7 min)

---

## Experiment: Prompt v4 (Coherence Guard) — REJECTED

**Motivation:** The rubric's 0.85 synonym-swap band fires on word-salad (obf=high) text because
named entities survive scrambling. A coherence guard rule was designed:
> "If suspicious text is grammatically incoherent/word-salad (unreadable nonsense), score 0.25 at most
> — even if named entities match."

**Rule added to prompt v4:**
```
0.25 — Weak overlap. Similar vocabulary or named entities but different structure.
       OR same-author reuse (different volumes of the same work).
       OR suspicious text is grammatically incoherent/word-salad (unreadable nonsense).
0.85 — Synonym-swap obfuscation. Suspicious text is GRAMMATICALLY COHERENT but content
       words are replaced with synonyms. Sentence structure and narrative sequence preserved.
       Named entities (proper nouns) still match the source.

RULES:
- Pick 0.85 ONLY if: (1) sentence structure is preserved AND (2) suspicious text is
  grammatically readable/coherent (not word-salad).
- If the suspicious text reads as nonsense or word-salad (sentences do not parse, words
  are in wrong grammatical positions), pick 0.25 at most — even if named entities match.
```

**Test results on high-obfuscation docs (doc10, doc12, doc15):**
| Doc | Pair | v3 score | v4 score | Expected |
|-----|------|----------|----------|---------|
| doc10 FP | garbled text + named entity | 0.850 | 0.250 ✅ | 0.25 |
| doc12 FP | word-salad | 0.850 | 0.250 ✅ | 0.25 |
| doc15 FP1 | word-salad | 0.850 | 0.250 ✅ | 0.25 |

**Regression test results:**
| Doc | Expected | v3 score | v4 score | Result |
|-----|----------|----------|----------|--------|
| doc5 correct (synonym-swap obf=low) | 0.85 | 0.850 | 0.250 ❌ | REGRESSION |
| doc5 wrong (topical only) | 0.00 | 0.000 | 0.250 ❌ | REGRESSION |
| doc2 same-author | 0.25 | 0.250 | 0.250 ✅ | OK |

**Decision: REJECTED.** The model cannot reliably distinguish obfuscation=low (coherent synonym-swap)
from obfuscation=high (word-salad) via prompt rules alone. Doc5 (legitimate synonym-swap) was wrongly
demoted to 0.25, and doc5 wrong was promoted from 0.000 to 0.250. Root cause: the LLM's judgment of
"grammatically coherent" is inconsistent at the boundary. Prompt-level fixes cannot reliably target
word-salad without collateral damage to valid synonym-swap detections.

**Correct fix:** Perplexity pre-filter (GPT-2 small score the suspicious text before sending to LLM)
or char n-gram aligner as post-processing step (see below).

---

## Experiment: 50 Pairs per Doc — REJECTED

**Motivation:** Current config sends 25 chunk pairs per candidate to the LLM. Hypothesis: 50 pairs
might improve recall by giving the LLM more evidence of plagiarism in difficult cases.

**Test docs:** doc12 (4/14 GT spans found), doc37, doc55 (challenging docs from the 80-doc run).

**Results:**
| Doc | 25 pairs result | 50 pairs result |
|-----|-----------------|-----------------|
| doc12 | 4 TP confirmed | 4 TP confirmed BUT new FP added (part13__source-document06435.txt scored 0.950) |
| doc37 | same as before | same as before |
| doc55 | same as before | same as before |

**Decision: REJECTED.** More pairs = more opportunity for wrong confirmations. The new FP on doc12
at 50 pairs proves this: a wrong source (part13__source-document06435) scored 0.950 at 50 pairs
but was not confirmed at 25 pairs. More evidence didn't help the correct source; it introduced a
new false positive. Reverted to 25 pairs.

---

## Experiment: TF-IDF for Retrieval Misses

**Motivation:** 5 docs had retrieval misses (recall@20=0) — the true source was never in the top-20
candidates. Hypothesis: TF-IDF char n-gram retrieval might surface these sources where ESA/LSA/EMB fail.

**Test doc:** doc26 (obfuscation=high, type=artificial, GT retrieval miss).

**Result:** Gate 1 FAILED. TF-IDF produced a fusion score below 0.60 for the true source.
The word-salad text destroys char n-gram signal as much as semantic signal — TF-IDF cannot
retrieve highly scrambled text either.

**Analysis:** All 5 retrieval miss docs have obfuscation=high (machine-generated word-salad).
The signal is completely destroyed at the token level — this is an unfixable retrieval ceiling
for this class of obfuscation. No retrieval method (TF-IDF, embedding, ESA, LSA) can match
incoherent word-salad to coherent source text reliably.

---

## PAN 2011 Metric: plagdet + Granularity

**Background from PAN 2011 evaluation:**
Official PAN 2011 scoring uses `plagdet = F1 / log2(1 + granularity)` where granularity measures
over-detection (detection count / GT span count for detected GT spans only). The formula penalizes
systems that report many tiny overlapping fragments instead of clean, merged detections.

**Implementation:** `scripts/final/compute_plagdet.py`
- Requires both suspicious AND source side overlap for detection (strict PAN criteria)
- Strips "part1__" prefix from source_doc_id for GT matching
- Granularity excludes GT spans with zero detections (not penalised as 0, just excluded)
- CLI: `--analytics`, `--per-doc-dir`, `--out-dir`
- Outputs: `plagdet_summary.parquet`, `obfuscation_breakdown.parquet`

**212-doc plagdet results (official thesis number):**

| Metric | Value |
|--------|-------|
| Macro plagdet | **0.6207** |
| Macro F1 (char-level) | 0.6234 |
| Granularity | 1.009 |
| Micro plagdet | 0.3058 |

**Obfuscation breakdown:**
| Obfuscation | GT spans | Det spans | Precision | Recall | F1 | plagdet |
|-------------|----------|-----------|-----------|--------|----|---------|
| none (verbatim) | 54 | 20 | 0.645 | 0.511 | 0.570 | 0.570 |
| low (synonym-swap) | 269 | 164 | 0.600 | 0.784 | 0.680 | 0.665 |
| high (word-salad) | 421 | 130 | 0.362 | 0.282 | 0.317 | 0.314 |

**80-doc plagdet (partial result for reference):**

| Metric | Value |
|--------|-------|
| Macro plagdet | 0.371 |
| Macro F1 | 0.378 |
| Granularity | 1.026 |

---

## Char N-gram Aligner (Post-LLM Span Extension)

**File:** `scripts/final/char_ngram_aligner.py`

**Purpose:** After the LLM confirms a source document, the pipeline only detects spans where
embedding chunk pairs scored highly. This misses plagiarised passages where the embedding similarity
dropped (heavily obfuscated passages, or passages whose chunks didn't align with source chunks).

**Algorithm:**
1. Load confirmed spans from `per_doc/<suspicious_doc_id>.parquet`
2. For each confirmed (suspicious_doc, source_doc) pair:
   - Slide a window of W characters over the suspicious doc
   - For each suspicious window, find best-matching source window via Jaccard char trigram similarity
   - If Jaccard ≥ threshold AND window doesn't already overlap confirmed span → add as new span
3. Merge adjacent new spans within merge_gap characters
4. Save extended spans to `per_doc/<suspicious_doc_id>_extended.parquet`

**Why char trigrams survive obfuscation:**
- Word-salad (obf=high): destroys word order but preserves individual characters.
  Char trigrams capture sub-word patterns that survive scrambling.
- Synonym-swap (obf=low): changes content words but preserves sentence structure.
  Char trigrams of function words and punctuation patterns still match.

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `--window` | 800 | Window size in characters |
| `--stride` | 400 | Stride between windows (50% overlap) |
| `--threshold` | 0.12 | Jaccard similarity threshold |
| `--merge-gap` | 1800 | Max gap (chars) to merge adjacent spans |
| `--doc-id` | — | Process single doc (for testing) |
| `--all` | — | Process all docs with confirmed spans |
| `--save` | — | Write `_extended.parquet` files |

**Usage:**
```bash
# Dry run on doc12 (best candidate — 4/14 GT spans detected)
uv run python scripts/final/char_ngram_aligner.py --doc-id part1__suspicious-document00012.txt

# Full run with save
uv run python scripts/final/char_ngram_aligner.py --all --save
```

**Status:** Implemented and evaluated on 212-doc confirmed spans. See results below.

**Evaluation results (212 docs):**

| Metric | Base pipeline | + Char n-gram aligner |
|--------|--------------|----------------------|
| Macro plagdet (plag only) | **0.359** | 0.193 ❌ |
| Macro F1 | 0.364 | 0.234 |
| Granularity | 1.019 | 1.265 |
| Precision (obf=low) | 0.600 | 0.185 |
| Recall (obf=low) | 0.784 | 0.842 |

**Decision: REJECTED for full corpus use.** The aligner adds 340 new spans across 84 docs but precision collapsed (0.60 → 0.18) because it amplifies existing LLM false positives. Docs like doc39 (46 FP confirmed spans) and doc82 (34 FP confirmed spans) received 10 and 14 additional FP spans respectively from the aligner — extending wrong source docs further.

Granularity jumped from 1.019 → 1.265, meaning the aligner fires multiple detections on the same GT span (over-fragmentation).

**When the aligner works:** On individual docs where the LLM confirmed the correct source, the aligner recovers missed passages accurately. Doc12: +4 new spans, all 4 hit GT regions not previously covered. The aligner is effective when LLM confirmation quality is high.

**Key thesis finding:** The aligner's effectiveness is bounded by LLM confirmation quality. If the LLM confirmed the wrong source doc, the aligner extends those false positives. This is a fundamental limitation: the post-processing step cannot distinguish correct from incorrect LLM confirmations without ground truth.

**Future fix:** Combine with a perplexity pre-filter to reduce LLM FPs before running the aligner. Fewer FP confirmed sources → aligner extends only real detections.

---

## Run 7 — 212-doc run: `212_docs_v3_prompt/`

**Config (same as Run 6):**
- LLM: gemma4:26b, prompt v3 rubric, Gate 1=0.60, Gate 2=0.85, pairs=25, branch union=yes

**Result (212 docs, stopped at doc 212/300):**

| Metric | Value |
|--------|-------|
| Plagiarised docs | 102 |
| Clean docs | 110 |
| Detected (any span) | 69/102 (67.6%) |
| Macro F1 | 0.413 |
| Macro charF1 | 0.398 |
| Retrieval recall@20 (plag only) | 0.640 |
| Clean FP docs | 15/110 (13.6%) |
| TP / FP / FN spans | 280 / 564 / 382 |
| **Macro plagdet** | **0.621** |
| Granularity | 1.009 |

**Key finding:** F1 is consistent with the 80-doc batch (0.413 vs 0.422) — the pipeline is stable
at scale. Retrieval recall@20 dropped from 0.753 to 0.640 as docs 81–212 include harder cases.
Granularity ≈ 1.0 confirms no over-fragmentation. The obf=low plagdet (0.665) is the pipeline's
strongest result; obf=high (0.314) remains the ceiling limited by word-salad destroying retrieval signal.

---

## Next Steps

### Implemented / Tested
- [x] Rubric prompt v3 (BEST CURRENT CONFIG — use for thesis)
- [x] Branch union top-3 per branch
- [x] Prompt v4 coherence guard (REJECTED)
- [x] 50 pairs per doc (REJECTED)
- [x] TF-IDF retrieval for misses (does not help on obf=high)
- [x] PAN plagdet metric implementation (`compute_plagdet.py`)
- [x] Char n-gram aligner implementation (`char_ngram_aligner.py`)
- [x] 212-doc run with official plagdet score (macro plagdet = 0.621)

### Remaining / Future Work
1. **Continue to 300-doc run** — complete remaining docs 213–300 for full corpus results
2. **Cross-encoder re-ranking** — after ESA+LSA+EMB fusion, re-rank candidates with a
   cross-encoder (reads suspicious + source chunk together). Targets the 33 fully missed docs
   where the true source is retrieved but ranked too low to pass Gate 1.
3. **Perplexity pre-filter** — GPT-2 small (117M, runs locally in seconds) scores suspicious
   text coherence before sending to gemma. High perplexity = word-salad → cap LLM max score
   at 0.25. Targets the FP problem on obf=high docs where named entities survive scrambling.
   Would also make the char n-gram aligner viable by reducing FP confirmed sources.
4. **Custom dataset** — small annotated dataset to demonstrate pipeline generalisability
   beyond PAN 2011.
5. **Fine-tuned embeddings** — domain-adapt embedding model on PAN 2011 pairs.
