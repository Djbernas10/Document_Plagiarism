# PAN-PC-11 — TF-IDF Ablation Run (first 20 docs)

First (and so far only) run of the PAN-PC-11 corpus with the TF-IDF branch
enabled and live per-document embeddings, rather than the `--skip-tfidf`
configuration used for every other batch run in this repo
(`old_results/*`, the 308-doc final run, etc.).

```
uv run python scripts/final/run_pipeline.py \
  --dataset pan2011 --docs 20 --relative-gap 0.85 --min-top1-score 0.60 \
  --tfidf-batch-size 64 --fresh \
  --run-embeddings --embeddings-backend local --ollama-model gemma4:26b
```

## Why only 20 docs

TF-IDF's offline index for PAN-PC-11 spans 2.8M source chunks across 29
on-disk shards (vs. 1,590 rows in 1 shard for the 30-source custom dataset).
At PAN-PC-11 scale this run took **15h 8m for 20 documents** — consistent
with the thesis's own claim (Section 5.7 / `generate_thesis_chapter.py`
around line 881) that TF-IDF is the slowest branch "by a wide margin" at
retrieval time on this corpus. A full 308-doc (or 11,093-doc) TF-IDF-enabled
run was not attempted; 20 docs is what was feasible in this session.

## Bug found and fixed during this run: TF-IDF batch size

The TF-IDF shard search (`search_tfidf_shards` in
`04_source_retrieval/source_retrieval_branches.py`) reloads all 29 shard
`.npz` files from disk **per query batch**, not once per document. The
original hardcoded `batch_size=8` meant a suspicious document with, say, 376
chunks re-read all 29 shards ~47 times. Document 6 in the initial attempt at
this run stalled for multiple hours before being killed.

Fix: added `--tfidf-batch-size` as a proper CLI flag (default kept at 8 for
backward compatibility) threaded through `run_pipeline.py` →
`lookup_pipeline()` → a new `TFIDF_BATCH_SIZE` module global in
`source_retrieval_branches.py`, replacing the hardcoded `8` at the one call
site. Re-run with `--tfidf-batch-size 64` completed doc 6 without stalling
and produced identical Gate 1 fusion scores to the pre-fix partial run on
docs 1-5 (confirms the batch size only changes speed, not results). Also
fixed, earlier in this session: a Windows-only crash where redirecting
stdout to a file used the console's legacy codepage instead of UTF-8,
crashing on any non-ASCII character in LLM reasoning text (e.g. `→`) — see
the custom-dataset ablation's README for detail. Both fixes are in
`run_pipeline.py` and `source_retrieval_branches.py` as committed.

## Per-branch Recall@1 (9 plagiarised docs of 20 evaluated)

Recall@1 = does the branch's own top-ranked source document match one of the
ground-truth sources (from `datasets/processed/PAN2011_ground_truth/pan2011_plagiarism_spans.parquet`).

| Branch | Recall@1 |
|---|---|
| TF-IDF | **0.889** (8/9) |
| ESA | **0.889** (8/9) |
| LSA | 0.778 (7/9) |
| Embeddings | 0.778 (7/9) |

Both TF-IDF and ESA missed the same document (`suspicious-document00014`, a
multi-source case where all four branches failed to find either of its two
true sources — a genuinely hard document, not a TF-IDF-specific weakness).
TF-IDF's other miss on the earlier custom-dataset ablation does not recur
here; on this sample TF-IDF is tied for best, not worst.

**This does not support the thesis's current framing** ("TF-IDF receives
the lowest weight because... embeddings receives the highest weight because
dense neural representations are the most robust to obfuscation," Section
5.8.4) at the retrieval level. On both the custom dataset and this PAN-PC-11
sample, TF-IDF was never the clearly weakest branch. It may still be
reasonable to keep TF-IDF's fusion weight low and disable it by default —
but the justification should be its retrieval-time cost (which this run
strongly confirms: 15h/20 docs), not an unsupported recall gap.

## Full-pipeline results (retrieval + LLM confirmation), 20 docs

| Metric | Binary macro | Char micro | Char macro |
|---|---|---|---|
| Precision | 0.859 | 0.646 | 0.819 |
| Recall | 0.851 | 0.689 | 0.891 |
| F1 | 0.849 | 0.667 | 0.841 |

- 20 documents evaluated: 9 with GT plagiarism spans, 11 clean.
- 0/11 clean docs had false alarms.
- Retrieval Recall@20 (macro, fused ranking): 1.00 — the fused ranking found
  the correct source in its top 20 for every plagiarised doc that reached
  Gate 1 (per `retrieval_recall.parquet`, 9 rows — one row per doc that had
  candidates to score for recall).

These full-pipeline numbers are on a small (20-doc) sample and are **not**
directly comparable to the 308-doc `--skip-tfidf` baseline
(`old_results/308_docs_full_v3/`) — different sample, different size. They
are reported here for completeness of this run, not as a replacement for the
existing 308-doc baseline.

## Caveats

- n=9 plagiarised docs — one flipped doc moves each branch's recall by
  ~0.11. Not large enough to be definitive on its own; combined with the
  custom-dataset ablation (n=5, also showing TF-IDF non-dominated), it's
  real evidence against the "TF-IDF is empirically the weakest branch"
  claim, but not a large-scale ablation.
- Recall@1 only (each branch's single best guess), not Recall@K — the
  pipeline only persists each branch's top-1 pick, not a full per-branch
  top-K list.
- `branch_tfidf_score` (the numeric score, not the top-1 document identity)
  is `0.0` for every row in `analytics_summary.parquet`, same issue as
  observed in the custom-dataset ablation — `source_retrieval_branches.py`
  does not populate `_branch_tfidf_score` for the winning candidate the way
  it does for the other three branches. Top-1 identity is unaffected.

## Files

- `analytics_summary.parquet` — per-doc metrics, all 20 docs
- `retrieval_recall.parquet` — Recall@20 diagnostic, 9 docs that had
  candidates to score
- `run.log` — full stdout of the run (15h 8m)
