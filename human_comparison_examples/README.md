# Human comparison examples

Five side-by-side samples of suspicious and source passages that the pipeline confirmed as
plagiarism in the final PAN 2011 run. Each file shows the highest-similarity chunk pair the LLM was
given for that confirmed source, so a reader can judge whether the detection is correct.

All examples come from the final configuration: `gemma4:26b` via Ollama, LLM confirmation threshold
0.85, 25 chunk pairs per candidate. The text is shown as stored after preprocessing (cleaning and
Unicode normalisation), so some spacing and punctuation differs from the raw corpus files. The pairs
were taken from the run's `--debug-llm` output, which is not committed.

Each example quotes the model's own reasoning for the confirmation. `score_source_doc` in
`scripts/final/run_pipeline.py` saves the parsed `score`, `is_likely_source` and `reasoning` fields
next to the prompt when `--debug-llm` is on. For each example, the reasoning was checked against the
25 evidence pairs the model actually saw. Four hold up. Example 5 does not: its reasoning is not
grounded in the evidence shown, even though the score and verdict may still be correct. It is kept
on purpose as a documented case of that failure.

| # | Suspicious → Source | Obfuscation | Top pair similarity | Reasoning quoted | Reasoning verified |
|---|---------------------|-------------|----------------------|-------------------|----------------------|
| 1 | [00007 → 06022](01_suspicious-document00007_vs_source-document06022.md) | Random / synonym substitution (low) | 0.95 | Yes | Yes |
| 2 | [00005 → 00178](02_suspicious-document00005_vs_source-document00178.md) | Random / synonym substitution (low) | 0.82 | Yes | Yes |
| 3 | [00012 → 07150](03_suspicious-document00012_vs_source-document07150.md) | Heavy paraphrase (high) | 0.81 | Yes | Yes |
| 4 | [00015 → 00853](04_suspicious-document00015_vs_source-document00853.md) | Heavy paraphrase (high) | 0.77 | Yes | Yes |
| 5 | [00015 → 07681](05_suspicious-document00015_vs_source-document07681.md) | Heavy paraphrase (high) | 0.76 | Yes | No, see the caveat in the file |

## How to read a pair

The 🔴 suspicious text and the 🟢 source text share a sentence skeleton: clause order, named
entities and phrase fragments. That skeleton survives when content words are swapped (low
obfuscation) and even when the surface form is heavily scrambled (high obfuscation), and it is what
the embedding retrieval and the LLM rubric pick up on. Examples 3 to 5 are the high-obfuscation
cases, and they say the most about when the pipeline's confirmations can be trusted.
