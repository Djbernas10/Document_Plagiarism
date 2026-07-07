# Human Comparison Examples

Side-by-side samples of suspicious vs. source passages that the pipeline
**confirmed as plagiarism** in the final PAN 2011 run. Each pair is the
highest-similarity chunk pair the LLM was shown for that confirmed source, so a
human reviewer can quickly judge whether the detection is correct.

All examples come from the final configuration: `gemma4:26b` via Ollama, LLM
confirmation threshold 0.85, 25 chunk pairs per candidate. Text is shown as
stored after preprocessing (cleaning + Unicode normalisation), which is why some
spacing and punctuation differ from the raw corpus files. Source data is the
`pipeline_results/llm_debug/` dumps.

**Verified with real model reasoning.** The pipeline's `--debug-llm` dump
originally captured only the prompt sent to the LLM, not its response,
so earlier drafts of these examples showed similarity scores and a bare
verdict with no way to inspect *why* the model confirmed a source. This was
fixed in `run_pipeline.py` (`score_source_doc`), which now also persists the
parsed `score` / `is_likely_source` / `reasoning` fields alongside the prompt.
All five examples have been regenerated against this fix and now include the
model's actual quoted reasoning for the confirmed detection. For each, the
reasoning was checked against the 25 evidence pairs actually shown to the
model — four examples check out; one (Example 5) does not, and is kept
deliberately as a documented case where the reasoning is not grounded in the
shown evidence even though the score/verdict may still be correct.

| # | Suspicious → Source | Obfuscation | Top pair similarity | Reasoning quoted | Reasoning verified |
|---|---------------------|-------------|----------------------|-------------------|----------------------|
| 1 | 00007 → 06022 | Random / synonym substitution (low) | 0.95 | Yes | Yes |
| 2 | 00005 → 00178 | Random / synonym substitution (low) | 0.82 | Yes | Yes |
| 3 | 00012 → 07150 | Heavy paraphrase (high) | 0.81 | Yes | Yes |
| 4 | 00015 → 00853 | Heavy paraphrase (high) | 0.77 | Yes | Yes |
| 5 | 00015 → 07681 | Heavy paraphrase (high) | 0.76 | Yes | **No — see caveat in file** |

**How to read a pair.** The 🔴 suspicious text and 🟢 source text share a
sentence skeleton — clause order, named entities, and phrase fragments — even
when content words have been swapped (low obfuscation) or the surface form has
been aggressively scrambled (high obfuscation). That surviving structure is what
the embedding retrieval and the LLM rubric key on. Examples 3–5 are the harder,
high-obfuscation cases and are the more interesting ones for judging where the
pipeline's confirmations are trustworthy.
