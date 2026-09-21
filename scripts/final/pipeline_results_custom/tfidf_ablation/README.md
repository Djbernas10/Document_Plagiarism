# Custom Dataset — TF-IDF Ablation Run

First (and so far only) run of the custom 10-doc dataset with the TF-IDF
branch enabled (all prior custom-dataset and PAN-PC-11 batch runs used
`--skip-tfidf`). Run with live per-document embeddings
(`--run-embeddings --embeddings-backend local`), not the cached embedding
parquet.

```
uv run python scripts/final/run_pipeline.py \
  --dataset custom --retrieval-recall-k 5 --relative-gap 0.85 --fresh \
  --run-embeddings --embeddings-backend local --ollama-model gemma4:26b
```

## Known issue: 3 of 10 docs did not reach LLM confirmation

Docs 00006, 00007, 00009 all passed Gate 1 (fusion scores 0.69-0.79) but
crashed before their LLM confirmation could run:

```
[WARN] Retrieval failed: 'charmap' codec can't encode character '→'
in position 49: character maps to <undefined>
```

Root cause: on Windows, when stdout is redirected to a file, Python opens it
with the console's legacy codepage (cp1252) instead of UTF-8. The LLM's
free-text `reasoning` field can legitimately contain non-ASCII characters
(e.g. an arrow), and printing a preview of it in `run_text_alignment`
([run_pipeline.py:334](../../run_pipeline.py#L334)) threw
`UnicodeEncodeError`, caught by the surrounding broad `except Exception`
and misreported as a retrieval failure — discarding an already-computed LLM
result. **Fixed** in this run's codebase by reconfiguring stdout/stderr to
UTF-8 at the top of `run_pipeline.py`, but this run itself predates the fix,
so `analytics_summary.parquet` here has only 7 of 10 docs (F1/plagdet
numbers from this file are therefore incomplete — do not cite them without
re-running docs 00006/00007/00009).

## Per-branch Recall@1 (retrieval only, unaffected by the crash above)

Branch top-1 picks are logged to stdout before the LLM stage, so this part
of the run is complete for all 10 docs. Recall@1 = does the branch's single
top-ranked source document match one of the ground-truth sources.

| Branch | Recall@1 (5 plagiarised docs) |
|---|---|
| LSA | 1.00 (5/5) |
| ESA | 1.00 (5/5) |
| TF-IDF | 0.80 (4/5) |
| Embeddings | 0.80 (4/5) |

Both misses are on doc 00010 (idea-level/concept-reuse paraphrase, the
hardest case documented in `datasets/custom_dataset/EXPERIMENTS_SUMMARY.md`):
TF-IDF picked `source-document00005.txt`, embeddings picked
`source-document00014.txt`; both wrong. LSA and ESA both picked the correct
`source-document00016.txt`.

On this dataset, TF-IDF is tied with the embedding branch, not clearly
weaker as the thesis's branch-weighting rationale assumes (Section 5.7 /
5.8.4: "TF-IDF receives the lowest weight because... embeddings receives the
highest weight because dense neural representations are the most robust to
obfuscation"). Caveat: n=5 plagiarised docs, so one flipped doc moves each
branch's recall by 0.20 — not enough to overturn the branch-weighting
design on its own, but enough that the "TF-IDF is empirically weaker" framing
should not be overstated from this dataset alone.

This does not speak to the actual stated reason TF-IDF is disabled in batch
runs (retrieval-time cost on PAN-PC-11's 11,093-document corpus) — see the
PAN-PC-11 TF-IDF ablation for that side of the argument.

Also note: `branch_tfidf_score` is `0.0` in every row of
`analytics_summary.parquet` even when `branch_tfidf_top1` is a real,
correctly-varying document — `source_retrieval_branches.py` does not
populate `_branch_tfidf_score` for the winning candidate the way it does for
the other three branches. Top-1 *identity* (used for the recall numbers
above) is unaffected; only the numeric score field is missing.

## Files

- `analytics_summary.parquet` — per-doc metrics as saved by this run (7/10
  docs; see crash note above)
- `run.log` — full stdout of the run
