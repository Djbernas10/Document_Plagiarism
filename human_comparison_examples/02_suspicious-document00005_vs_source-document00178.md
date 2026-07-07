# Example 2 — suspicious-document00005 vs source-document00178

**Obfuscation type:** Random / synonym substitution (obfuscation: low)  
**Source document:** `part1__source-document00178.txt`  
**Top pair embedding similarity:** 0.8198  
**LLM verdict:** confirmed as plagiarism source (score ≥ 0.85, Gemma 4 26B)

> A single confirmed span. The shared narrative skeleton survives the word substitutions, which keeps the embedding similarity high enough to retrieve and confirm.

> **Bold** marks the phrases that match between the two texts.

---

## Flagged passage (highest-similarity chunk pair)

### 🔴 Suspicious text (the submitted document)

> or character **two miles below camp**, **its cluster of** Weir **adobe houses showing** now
> **through the cottonwoods that embowered the place**. He had come suddenly, thoughtfully, returning
> with Magney, **the engineer** in charge, when the the had been summoned east for a conference with
> the company'sympathy directors. Under Magney the work of construction had been inaugurated the
> previous summer, but progress had not been as rapid as unmarried; there had been delays, labor
> difficulties, likeable dust during the months since; and **Weir had** been chosen to defeat Magney.
> In his profession Weir had a toil, built on relentless reputation and sound ideas and daring

### 🟢 Source text (the original)

> whirling away to the nearest railway point, Bowenville, thirty-five miles distant. He thoughtfully
> watched the car, a black spot in a haze of dust, speeding towards the New Mexican town of San Mateo,
> on the Burntwood River **two miles below camp**, **its cluster of** brown **adobe houses showing**
> indistinctly **through the cottonwoods that embowered the place**. For Magney he felt a certain
> amount of sympathy, for **the engineer** was leaving with a recognition of defeat; he was a likeable
> man, as Steele **Weir had** discovered during their brief acquaintance, a good theoretical engineer,
> but lacking in the prime quality of a successful chief--fighting spirit and a

---

## Second flagged pair (embedding similarity 0.7963)

### 🔴 Suspicious text (the submitted document)

> If a horse passed a little quickly, everybody ran without knowing where. The apartments of
> Chamillart were crowded with lackeys, even into the street, sent by people desiring to be informed
> of the moment that a courier arrived; and this terror and uncertainty lasted nearly a month. The
> provinces were even more troubled than Paris. For brown **he felt a certain amount of** s, for
> latter **engineer was leaving with a recognition of** succeed; **he was a** local spot, **as
> Steele** Magney **had discovered during their brief acquaintance**, **a good theoretical** extreme,
> **but lacking in** a **quality of a successful** black-- **fighting spirit and** an indomitable
> will. He had re

### 🟢 Source text (the original)

> whirling away to the nearest railway point, Bowenville, thirty-five miles distant. He thoughtfully
> watched the car, a black spot in a haze of dust, speeding towards the New Mexican town of San Mateo,
> on the Burntwood River two miles below camp, its cluster of brown adobe houses showing indistinctly
> through the cottonwoods that embowered the place. For Magney **he felt a certain amount of**
> sympathy, for the **engineer was leaving with a recognition of** defeat; **he was a** likeable man,
> **as Steele** Weir **had discovered during their brief acquaintance**, **a good theoretical**
> engineer, **but lacking in** the prime **quality of a successful** chief--**fighting spirit and** a

---

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*

---

## LLM verdict and reasoning

**Score:** 0.85 · **is_likely_source:** True

> "The suspicious text exhibits clear synonym-swap obfuscation and structural mirroring of the candidate source. While some pairs contain unrelated 'noise' (like the Chamillart/Paris text), the core content in several pairs shows a pattern where specific phrases from the source are replaced with synonyms or slightly altered words while maintaining the exact narrative sequence and named entities (e.g., Magney, Weir, San Mateo). The presence of scrambled/corrupted text suggests an attempt to mask verbatim copying through character substitution or word replacement."
