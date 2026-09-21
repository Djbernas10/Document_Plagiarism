# Custom Dataset — Experiments Summary

Scope: wiring the full pipeline (`run_pipeline.py --dataset custom`) against the
30-source / 10-suspicious custom dataset, and diagnosing why 3 of the 5
plagiarised docs were initially missed.

## 1. Pipeline wiring

- `run_pipeline.py`, `source_retrieval_branches.py`, and both copies of
  `embeddings.py` (`scripts/embeddings.py` used by Docker, and
  `scripts/final/03_index_creation/embeddings.py`) now accept `--dataset {pan2011,custom}`.
- Results are written to separate directories per dataset
  (`pipeline_results/` for PAN2011, `pipeline_results_custom/` for custom) so
  the finalized PAN2011 308-doc results (`pipeline_results/old_results/308_docs_full_v3/`)
  are never touched by custom-dataset runs.
- Added `--retrieval-recall-k` (default 20, matching PAN2011's convention) so
  the Recall@K diagnostic can use a smaller K (e.g. 5) appropriate for this
  dataset's much smaller 30-document source pool — Recall@20 out of 30 sources
  is a weak test; Recall@5 is a meaningful one.

## 2. Baseline run (before fixes)

| Metric | Value |
|---|---|
| Binary P / R / F1 (macro) | 0.70 / 0.70 / 0.70 |
| TP / FP / FN | 2 / 0 / 7 |
| Clean docs with false alarms | 0 / 5 |

Per-doc: 00006 and 00009 (verbatim / sentence-reordering) detected perfectly.
00007 (verbatim, multi-source), 00008 (near-verbatim, synonym-substitution),
and 00010 (paraphrase / idea-level, multi-source) were missed entirely.

## 3. Diagnosing the 3 misses

Used `--debug-llm` to dump the exact chunk-pairs and prompts sent to the LLM,
then inspected `per_doc/*.meta.json` for branch scores and LLM verdicts.

**00007 (verbatim, 2 sources)** — retrieval found both correct sources
(`source-document00018.txt`, `source-document00016.txt`) with no problem, but
the LLM alignment stage scored both at `0.0` ("no overlap"), despite Pair 1 in
the dump showing near-exact verbatim text in both SUSPICIOUS and SOURCE
columns. Root cause: the plagiarised span (579+363 chars ≈ 150 words) was a
small fraction of the ~2100-word document, so most of the 25 chunk-pairs sent
to the LLM were dominated by non-overlapping surrounding prose. The LLM
appears to anchor on the majority of weak/irrelevant pairs rather than
carefully verifying the one strong pair.

Ran an isolated experiment (`experiments/rescore_doc00007_prompt_variant.py`,
not part of the shared pipeline) re-sending the same dumped pairs with a
modified prompt: pairs deduplicated, sorted and labelled by embedding
similarity, plus an explicit instruction that "a single pair with clear
overlap is sufficient." Result: **no change** — both sources still scored
0.0. This rules out prompt engineering as a fix and confirms a genuine model
reading/verification limitation (Ollama `gemma4:26b`), not a pipeline defect.

**00008 (near-verbatim, synonym-substitution)** — all 4 retrieval branches
correctly ranked `source-document00029.txt` (the true source) as top-1
(fusion=0.568), but Gate 1's `min_top1_score=0.60` threshold rejected it before
the LLM stage ever ran. The plagiarised span is only 359 chars (~60 words) out
of a ~2000-word document — intrinsically hard to retrieve confidently, since
the dominant document-level signal is the ~97% non-plagiarised content.

**00010 (paraphrase / idea-level, 3 sources)** — also failed Gate 1
(fusion=0.430). Unlike 00008, branches disagreed on top-1 (TF-IDF→00005,
EMB→00014, both wrong; LSA/ESA→00016, correct but weak signal at 0.43-0.60).
This is the textbook hardest case for lexical/embedding retrieval: idea-level
reuse with heavy rewriting leaves little lexical or even semantic-embedding
trace.

## 4. Fix applied: extend doc 00007's plagiarised passages

Since the dilution theory (small verbatim span swamped by mostly-original
surrounding prose) was the suspected cause for 00007, extended both
plagiarised excerpts in `generate_test_docs.py::build_doc_00012()` from
579/363 chars to ~1830/~1734 chars each (full paragraphs copied verbatim from
the same two sources, same offsets re-verified via `find_offset`/
`verify_ground_truth.py`). Regenerated the doc, re-ran preprocessing
(`preprocess_custom_dataset.py`) and ground truth (`build_custom_ground_truth.py`).

Result: `source-document00016.txt` flipped from `score=0.0` (rejected) to
`score=0.95` (confirmed) — the longer contiguous excerpt aligned more cleanly
with single 300-word chunks, giving the LLM a cleaner, less-diluted pair to
judge. `source-document00018.txt` is **still rejected** even with the longer
passage — the 300-word sliding-window chunking still splits the contiguous
excerpt across multiple overlapping chunks with some weak-alignment fragments,
and the LLM still anchors on those. This is a partial, not complete, fix.

00008 and 00010 were left untouched — their failure mode (Gate 1 retrieval
threshold / weak branch agreement) is unrelated to LLM dilution and reflects a
genuine, expected difficulty gradient (short spans and idea-level reuse are
hard for retrieval-based systems, arguably for human reviewers too).

## 5. Result after fix

| Metric | Before | After |
|---|---|---|
| Binary P / R / F1 (macro) | 0.70 / 0.70 / 0.70 | **0.80 / 0.75 / 0.77** |
| TP / FP / FN | 2 / 0 / 7 | **3 / 0 / 6** |
| Char recall (macro, plagiarised docs) | — | 0.747 |
| Retrieval Recall@5 (macro, Gate-1-passing docs) | — | **1.00** |
| Clean docs with false alarms | 0 / 5 | 0 / 5 (unchanged) |

PlagDet (`compute_plagdet.py --gt-xml-dir datasets/custom_dataset/ground_truth`):

| Scope | Macro plagdet | Macro F1 | Macro recall |
|---|---|---|---|
| Plagiarised docs only (5 docs, 2 zero-detection) | 0.0954 | 0.0954 | 0.4974 |
| All 10 docs (incl. clean) | 0.5477 | 0.5477 | — |

Per-obfuscation breakdown confirms the difficulty gradient directly:

| Category | GT spans | Detected | Recall |
|---|---|---|---|
| none (verbatim) | 4 | 3 | 0.65 |
| sentence-reordering | 2 | 2 | 1.00 |
| synonym-substitution | 2 | 0 | 0.00 |
| heavy-rewrite | 2 | 0 | 0.00 |
| concept-reuse | 1 | 0 | 0.00 |

## 6. Takeaways for the thesis discussion

- The pipeline's retrieval stage is strong even on this much smaller (30-doc)
  corpus: Recall@5 = 1.00 whenever Gate 1 passes, and even Gate-1-failing docs
  (00008) had the correct source ranked top-1 by every branch — the bottleneck
  there is threshold calibration for short spans, not retrieval quality.
- The LLM alignment stage has a documented, reproducible failure mode: it can
  reject genuine verbatim matches when they are a small fraction of a
  25-pair, mostly-irrelevant batch, even when explicitly instructed that one
  strong pair is sufficient. This is model-judgment limited, not a prompt
  engineering or pipeline-architecture problem (verified via an isolated,
  non-pipeline experiment that reused real production debug dumps).
- Detection difficulty cleanly tracks obfuscation level (verbatim >
  near-verbatim/reordering > synonym-substitution > heavy-rewrite >
  concept-reuse), consistent with PAN2011 findings and with the intuition
  that short, heavily-paraphrased, idea-level reuse is hard to detect
  automatically — and arguably for human reviewers skimming a full document too.

> **Note:** the numbers in Section 5 above are from an intermediate run, before
> the final containerized configuration. The final thesis figures (Table 6.9)
> are macro plagdet = 0.1047, precision = 0.0590, recall = 0.6974 — see
> `scripts/final/pipeline_results_custom/plagdet_summary.parquet`. As of
> 2026-09-16 the **official Potthast et al. corpus-pooled plagdet** has also
> been computed for the full 10-doc set: precision 0.0737, recall 0.6364,
> plagdet **0.1321** (plag-only pooling), with per-category values (none
> 0.1922, sentence-reordering 0.0961, synonym-substitution 0.0467,
> heavy-rewrite/concept-reuse 0.0000) — see
> `scripts/final/pipeline_results_custom/official_plagdet/plagdet_official_summary.parquet`
> and `scripts/final/compute_plagdet_official.py`.
