# PAN-PC-11 TF-IDF ablation run (first 20 docs)

The only PAN-PC-11 run with the TF-IDF branch turned on, using live per-document embeddings. Every
other batch run in this repo, including the 308-document final run in `old_results/`, used
`--skip-tfidf`.

```
uv run python scripts/final/run_pipeline.py \
  --dataset pan2011 --docs 20 --relative-gap 0.85 --min-top1-score 0.60 \
  --tfidf-batch-size 64 --fresh \
  --run-embeddings --embeddings-backend local --ollama-model gemma4:26b
```

The run's output files (per-document metrics, retrieval recall and the full log) stay local and are
not committed. This README records the results.

## Why only 20 docs

The PAN-PC-11 TF-IDF index has 2.8M source chunks in 29 on-disk shards. The 30-source custom dataset
has 1,590 rows in one shard. At PAN-PC-11 scale this run took 15h 8m for 20 documents, which fits
the project report statement (Section 5.7) that TF-IDF is by far the slowest branch at retrieval time on
this corpus. A TF-IDF run over all 308 documents (or the full 11,093) was not attempted.

## TF-IDF batch size bug fixed during this run

`search_tfidf_shards` in `04_source_retrieval/source_retrieval_branches.py` reloads all 29 shard
`.npz` files from disk for every query batch, not once per document. With the old hardcoded
`batch_size=8`, a suspicious document with 376 chunks re-read all 29 shards about 47 times.
Document 6 stalled for several hours in the first attempt at this run and was killed.

The batch size is now a CLI flag, `--tfidf-batch-size` (default still 8). `run_pipeline.py` passes
it through `lookup_pipeline()` to the `TFIDF_BATCH_SIZE` module global in
`source_retrieval_branches.py`, which the shard search uses. With `--tfidf-batch-size 64`, document
6 finished without stalling, and documents 1-5 got the same Gate 1 fusion scores as in the earlier
partial run, so the batch size changes speed and not results.

A second fix came from this session: on Windows, redirecting stdout to a file made Python use the
console's legacy codepage instead of UTF-8, and any non-ASCII character in the LLM reasoning (such
as `→`) crashed the run. The custom-dataset ablation README
(`../../pipeline_results_custom/tfidf_ablation/README.md`) has the details. Both fixes are
committed in `run_pipeline.py` and `source_retrieval_branches.py`.

## Per-branch Recall@1 (9 plagiarised docs out of 20)

Recall@1 asks whether a branch's top-ranked source document is one of the ground-truth sources in
the PAN 2011 annotations.

| Branch | Recall@1 |
|---|---|
| TF-IDF | 0.889 (8/9) |
| ESA | 0.889 (8/9) |
| LSA | 0.778 (7/9) |
| Embeddings | 0.778 (7/9) |

TF-IDF and ESA both missed `suspicious-document00014`, a multi-source document where none of the
four branches found either of its two true sources. That miss says more about the document than
about TF-IDF. The TF-IDF miss from the custom-dataset ablation does not happen here, and on this
sample TF-IDF ties for best.

This does not support the project report's current framing at the retrieval level (Section 5.8.4: "TF-IDF
receives the lowest weight because... embeddings receives the highest weight because dense neural
representations are the most robust to obfuscation"). On both the custom dataset and this
PAN-PC-11 sample, TF-IDF was never clearly the weakest branch. Keeping its fusion weight low and
disabling it by default may still be reasonable, but the reason should be retrieval cost (15h for
20 docs here), not a recall gap the data does not show.

## Full-pipeline results (retrieval + LLM confirmation), 20 docs

| Metric | Binary macro | Char micro | Char macro |
|---|---|---|---|
| Precision | 0.859 | 0.646 | 0.819 |
| Recall | 0.851 | 0.689 | 0.891 |
| F1 | 0.849 | 0.667 | 0.841 |

- 20 documents evaluated: 9 with ground-truth plagiarism spans, 11 clean.
- None of the 11 clean documents raised a false alarm.
- Retrieval Recall@20 (macro, fused ranking) was 1.00. The fused top 20 held the correct source for
  every plagiarised document that reached Gate 1 (9 documents had candidates to score).

These numbers come from a 20-document sample and cannot be compared directly with the 308-document
`--skip-tfidf` baseline in `old_results/308_docs_full_v3/`, which uses a different and larger
sample. They are reported for this run only and do not replace that baseline.

## Caveats

- Only 9 plagiarised documents, so one flipped document moves a branch's recall by about 0.11.
  Together with the custom-dataset ablation (5 plagiarised documents, where TF-IDF was also not the
  weakest), this is real evidence against the claim that TF-IDF is empirically the weakest branch,
  but it is not a large-scale ablation.
- Recall@1 only, not Recall@K. The pipeline saves each branch's top-1 pick, not a full per-branch
  top-K list.
- `branch_tfidf_score` (the numeric score, not the top-1 document) is `0.0` for every document in
  the saved metrics, the same issue seen in the custom-dataset ablation.
  `source_retrieval_branches.py` does not fill `_branch_tfidf_score` for the winning candidate as it
  does for the other three branches. The top-1 document identity is unaffected.
