# Custom dataset TF-IDF ablation run

The only run of the 10-document custom dataset with the TF-IDF branch turned on. Every earlier
custom-dataset and PAN-PC-11 batch run used `--skip-tfidf`. This run used live per-document
embeddings (`--run-embeddings --embeddings-backend local`) instead of the cached embedding parquet.

```
uv run python scripts/final/run_pipeline.py \
  --dataset custom --retrieval-recall-k 5 --relative-gap 0.85 --fresh \
  --run-embeddings --embeddings-backend local --ollama-model gemma4:26b
```

The run's output files (per-document metrics and the full log) stay local and are not committed.
This README records the results.

## Known issue: 3 of 10 docs did not reach LLM confirmation

Docs 00006, 00007 and 00009 passed Gate 1 (fusion scores 0.69-0.79) but crashed before LLM
confirmation:

```
[WARN] Retrieval failed: 'charmap' codec can't encode character '→'
in position 49: character maps to <undefined>
```

On Windows, when stdout is redirected to a file, Python opens it with the console's legacy codepage
(cp1252) instead of UTF-8. The LLM's free-text `reasoning` field can contain non-ASCII characters
such as an arrow. Printing a preview of it in `run_text_alignment` (`scripts/final/run_pipeline.py`)
raised `UnicodeEncodeError`. A broad `except Exception` caught it and reported it as a retrieval
failure, which threw away an LLM result that had already been computed.

`run_pipeline.py` now reconfigures stdout and stderr to UTF-8 at startup, but this run predates the
fix. Its saved metrics cover only 7 of the 10 docs, so its F1 and plagdet numbers are incomplete.
Do not cite them without re-running docs 00006, 00007 and 00009.

## Per-branch Recall@1 (retrieval only, not affected by the crash)

Each branch's top-1 pick is logged before the LLM stage, so this part of the run is complete for
all 10 docs. Recall@1 asks whether a branch's top-ranked source document is one of the ground-truth
sources.

| Branch | Recall@1 (5 plagiarised docs) |
|---|---|
| LSA | 1.00 (5/5) |
| ESA | 1.00 (5/5) |
| TF-IDF | 0.80 (4/5) |
| Embeddings | 0.80 (4/5) |

Both misses are on doc 00010, the idea-level and concept-reuse paraphrase case, which is the hardest
document in the set (see `datasets/custom_dataset/EXPERIMENTS_SUMMARY.md`). TF-IDF picked
`source-document00005.txt` and embeddings picked `source-document00014.txt`, both wrong. LSA and ESA
both picked the correct `source-document00016.txt`.

On this dataset TF-IDF ties with the embedding branch. It is not clearly weaker, as the project
report's branch-weighting rationale assumes (Sections 5.7 and 5.8.4: "TF-IDF receives the lowest weight
because... embeddings receives the highest weight because dense neural representations are the most
robust to obfuscation"). With only 5 plagiarised docs, one flipped doc moves a branch's recall by
0.20. That is not enough to overturn the branch weighting by itself, but it is enough that the
project report should not lean on this dataset to call TF-IDF empirically weaker.

This result does not address the stated reason TF-IDF is off in batch runs, which is retrieval cost
on PAN-PC-11's 11,093-document corpus. The PAN-PC-11 TF-IDF ablation
(`../../pipeline_results/tfidf_ablation/README.md`) covers that.

`branch_tfidf_score` is `0.0` for every doc in the saved metrics, even when `branch_tfidf_top1` holds
a real document that varies correctly. `source_retrieval_branches.py` does not fill
`_branch_tfidf_score` for the winning candidate as it does for the other three branches. The top-1
document identity used for the recall numbers above is unaffected. Only the numeric score is
missing.
