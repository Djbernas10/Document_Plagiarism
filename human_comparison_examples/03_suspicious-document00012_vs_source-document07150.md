# Example 3 — suspicious-document00012 vs source-document07150

**Obfuscation type:** Heavy paraphrase (obfuscation: high)  
**Source document:** `part15__source-document07150.txt`  
**Top pair embedding similarity:** 0.8127  
**LLM verdict:** confirmed as plagiarism source (score ≥ 0.85, Gemma 4 26B)

> A harder case: the surface form is more aggressively rewritten, but enough named entities and phrase fragments survive for the retrieval + LLM stack to still confirm the source.

> **Bold** marks the phrases that match between the two texts.

---

## Flagged passage (highest-similarity chunk pair)

### 🔴 Suspicious text (the submitted document)

> of Cornelius, those **100**,**000**,**000** any, **and that of** S. vanderbilt, of sister, this
> **20**,**000**,**000**. [annotate: "They took who have if a State?"-- Pulp, Where s, 1889.] append
> to luck of records and same associate should keep of unit, by Vanderbilts have so own **about
> 300**,**000**,**000**. **Since** the universe where it became of Commonwealth have today incarnate;
> riches of these has will in centralised; great phenomenon have travel **beyond their** both proud of
> landmarks are amply; the ownership of **Vanderbilts have** boom and swell in argument was already,
> although they should alone stands for oligarchy has been infringe for control. Even be likely that
> it is some phe

### 🟢 Source text (the original)

> the first Cornelius, at $**100**,**000**,**000** each, **and that of** Frederick W. Vanderbilt, a
> brother of those two men, at $**20**,**000**,**000**. [Footnote: "Who Owns the United States?"--The
> Forum Magazine, November, 1889.] Adding the fortunes of the various other members of the Vanderbilt
> family, the Vanderbilts then possessed **about** $**300**,**000**,**000**. **Since** that time the
> population and resources of the United States have vastly increased; wealth in the hold of a few has
> become more intensely centralized; great fortunes have gone far **beyond their** already
> extraordinary boundaries of twenty years ago; the possessions of the **Vanderbilts have** expanded
> and swollen in value

---

## Second flagged pair (embedding similarity 0.7962)

### 🔴 Suspicious text (the submitted document)

> began his case. He was at sixty-nine years to vanderbilt, this scarce, unmolested, bold jargon. With
> this Court was gleaned; he write this words, and treasury was these American, for time, On court. It
> was invested of the he can get the so, and undoubtedly and however than the property. Like Astors,
> whilst he was able, bold, and real details. He kept his plans to himself. The contents of his
> pocket-book had been so well invested by Mr. Pickwick, we are told, that he had a handsome
> independence **for the** purpose of his last days. PICKWICK AND THE "GEORGE" Equip **what became of
> these** 1000000 **in loot**? Substance **went in profits to** a **owners**, **and** a Co

### 🟢 Source text (the original)

> returns from the "postages" far more than paid them **for the** transportation of mails. And **what
> became of these** millions **in loot**? Part **went in profits to** the **owners**, **and** another
> part was used as private capital by them to build more and newer ships constantly. Practically none
> of Vanderbilt's ships cost him a cent; the Government funds paid for their building. In fact, a
> careful tracing of the history of all of the subsidized steamship companies proves that this plunder
> from the Government was very considerably more than enough to build and equip their entire lines.
> One of the subsidized steamship lines was that of E. K. Collins & Co., a line r

---

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*

---

## LLM verdict and reasoning

**Score:** 0.95 · **is_likely_source:** True

> "The suspicious text contains highly distorted versions of the candidate source's exact phrasing, names, and numerical data. While many words have been replaced by nonsensical synonyms or garbled characters (e.g., '100,000,000 any' for '$100,000,000 each'), the sequence of information, specific numbers, and unique identifiers like dates and names are clearly derived from the source text."
