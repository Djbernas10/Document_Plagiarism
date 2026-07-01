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

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*
