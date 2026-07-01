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

| # | Suspicious → Source | Obfuscation | Top pair similarity |
|---|---------------------|-------------|---------------------|
| 1 | 00007 → 06022 | Random / synonym substitution (low) | 0.95 |
| 2 | 00005 → 00178 | Random / synonym substitution (low) | 0.82 |
| 3 | 00012 → 07150 | Heavy paraphrase (high) | 0.81 |
| 4 | 00015 → 00853 | Heavy paraphrase (high) | 0.77 |
| 5 | 00015 → 07681 | Heavy paraphrase (high) | 0.76 |

**How to read a pair.** The 🔴 suspicious text and 🟢 source text share a
sentence skeleton — clause order, named entities, and phrase fragments — even
when content words have been swapped (low obfuscation) or the surface form has
been aggressively scrambled (high obfuscation). That surviving structure is what
the embedding retrieval and the LLM rubric key on. Examples 3–5 are the harder,
high-obfuscation cases and are the more interesting ones for judging where the
pipeline's confirmations are trustworthy.
