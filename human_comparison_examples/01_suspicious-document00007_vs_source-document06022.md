# Example 1 — suspicious-document00007 vs source-document06022

**Obfuscation type:** Random / synonym substitution (obfuscation: low)  
**Source document:** `part13__source-document06022.txt`  
**Top pair embedding similarity:** 0.9516  
**LLM verdict:** confirmed as plagiarism source (score ≥ 0.85, Gemma 4 26B)

> Sentence structure and clause order are preserved almost verbatim while content words are swapped for near-synonyms. This is the showcase document from the thesis (binary span F1 = 1.0, character F1 = 0.96).

> **Bold** marks the phrases that match between the two texts.

---

## Flagged passage (highest-similarity chunk pair)

### 🔴 Suspicious text (the submitted document)

> unable **people**, **if it remains at peace**. **Commerce is** now broad with great **art**; **but
> cannot produce it**. **Manufacture not** ever is due **to produce it**, **but** briefly **destroys
> whatever seeds of it exist**. **There is** this historic art careful to the **nation but that which
> is based on battle**. Knightly, **though i hope you love fighting for its own sake**, **you must**,
> **i imagine**, **be** special **at my assertion that there is any** the possible **fruit of
> fighting**. **You supposed**, so, **that your office was to defend the works of peace**, **but
> certainly not to found them**: nay, the thoughtful course of war, you may have thought, was
> persistently to destroy them. Hear why: i have given

### 🟢 Source text (the original)

> of soldiers. There is no art among a shepherd people, if it remains at peace. There is no art among
> an agricultural **people**, **if it remains at peace**. **Commerce is** barely consistent with fine
> **art**; **but cannot produce it**. **Manufacture not** only is unable **to produce it**, **but**
> invariably **destroys whatever seeds of it exist**. **There is** no great art possible to a **nation
> but that which is based on battle**. Now, **though I hope you love fighting for its own sake**,
> **you must**, **I imagine**, **be** surprised **at my assertion that there is any** such good
> **fruit of fighting**. **You supposed**, probably, **that your office was to defend the works of
> peace**, **but certainly not to found them**

---

## Second flagged pair (embedding similarity 0.9032)

### 🔴 Suspicious text (the submitted document)

> are twice affairs **every way** yet separate, only **noble**, **and** patiently great, that the
> considerable **teaching than their** then **example**, **and their few words of grave and tried
> counsel should be** the right in **you**, **or** indeed, **without assurance of** fine **modesty in
> the offerer**, **endured by you**. **But being asked**, **not** barely nor very, **i have not
> ventured** now **to refuse**; **and** it **will try**, **in** now this **words**, **to** consistent
> **before you** no **reason why you should accept my excuse**, **and hear me** wholly. **You may
> imagine that your work is** unfairly perpetual **to**, **and** noble from mine. Far so from that,
> all no good and pure arts of peace are founded on war; some different art

### 🟢 Source text (the original)

> to teach you yours. Nay, I knew that there ought to be no such need, for the great veteran soldiers
> of England are now men **every way** so thoughtful, so **noble**, **and** so good, that no other
> **teaching than their** knightly **example**, **and their few words of grave and tried counsel
> should be** either necessary for **you**, **or** even, **without assurance of** due **modesty in the
> offerer**, **endured by you**. **But being asked**, **not** once nor twice, **I have not ventured**
> persistently **to refuse**; **and** I **will try**, **in** very few **words**, **to** lay **before
> you** some **reason why you should accept my excuse**, **and hear me** patiently. **You may imagine
> that your work is** wholly foreign **to**, **and** separate fr

---

## Pair the LLM's reasoning actually cites (pair 3 of 25, embedding similarity 0.8886)

The reasoning below quotes "the supremacy of a Tintoret" and "laths of the roof" as
evidence — neither phrase is in the two highest-similarity pairs shown above. This is
the pair those quotes come from.

### 🔴 Suspicious text (the submitted document)

> as i believed, of all painters whatsoever. And nevertheless, i who tell you either of
> any use of war, should have been the last of men to tell you yet, had i trusted my the
> experience to only. I formed a faith, (whether great or surprised matters at present
> nothing,) **in the supremacy of a Tintoret**, under a roof covered with his pictures;
> and of no leaders, three of such noblest were once in the form of shreds of necessary
> canvas, mixed up with **the laths of the roof**, rent through by three Austrian shells.

### 🟢 Source text (the original)

> last of men to tell you so, had I trusted my own experience only. Hear why: I have
> given a considerable part of my life to the investigation of Venetian painting and
> the result of that enquiry was my fixing upon one man as the greatest of all
> Venetians, and therefore, as I believed, of all painters whatsoever. I formed this
> faith, (whether right or wrong matters at present nothing,) **in the supremacy of the
> painter Tintoret**, under a roof covered with his pictures; and of those pictures,
> three of the noblest were then in the form of shreds of ragged canvas, mixed up with
> **the laths of the roof**, rent through by three Austrian shells.

---

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*

---

## LLM verdict and reasoning

**Score:** 0.95 · **is_likely_source:** True

> "The suspicious text contains multiple instances of near-verbatim copying from the candidate source, characterized by significant word-for-word overlap and shared unique phrases (e.g., 'the supremacy of a Tintoret', 'laths of the roof', 'great art possible to a nation but that which is based on battle'). While some words are slightly altered or corrupted in the suspicious text, the sequence of ideas and specific phrasing remains almost identical across several pairs."
