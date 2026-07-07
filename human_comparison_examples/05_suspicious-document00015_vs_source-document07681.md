# Example 5 — suspicious-document00015 vs source-document07681

**Obfuscation type:** Heavy paraphrase (obfuscation: high)  
**Source document:** `part16__source-document07681.txt`  
**Top pair embedding similarity:** 0.7574  
**LLM verdict:** confirmed as plagiarism source (score = 0.95, Gemma 4 26B)

> ⚠️ **Caveat example — reasoning does not match the shown evidence.** A second,
> independent source confirmed for the same suspicious document 00015. The score and
> verdict may well be correct, but the model's *reasoning* quotes phrases that do not
> appear in any of the 25 evidence pairs it was actually given (see below). This is
> kept in the set deliberately, as a documented instance of the reasoning-verification
> risk discussed in Chapter 7 (Future Work) — the model's justification is not always
> grounded in the evidence it was shown, even when the underlying verdict looks right.

> **Bold** marks the phrases that match between the two texts.

---

## Flagged passage (highest-similarity chunk pair)

### 🔴 Suspicious text (the submitted document)

> prestige, Englishmen here would long ago have adopted the Indian costume. I may mention incidentally
> **that I** do not go about Champaran bare headed. I do avoid shoes for sacred reasons. But I find
> too that it is more natural and healthier to avoid them whenever possible. He was not in families
> with he saw; he was not early experts is touched upon race. "to-morrow?" them said this Control. "we
> shall be ambitious **to have** point, dream," we state this Education. C., "to satisfied dream for
> we to-morrow." He has still not the courage, in spite of most admirable contact with me, to discard
> his semi-anglicised dress and whenever he goes to see officia

### 🟢 Source text (the original)

> the door his head was uncovered; but I had no sooner crossed the threshold than he made haste to don
> his flat turban,--reflecting, perhaps, **that I** had never seen him without it, and might resent
> his bare head as an indignity. Of course his feet were unshod. **To have** worn his sandals in my
> presence would have been a flagrant insult; but on the porch I espied those two queer clogs of wood,
> shaped to the sole of the foot, and having no other fastening than an impracticable-looking knob, to
> be held between the toes. This is the orthodox Hindoo dress; but the costume for public occasions of
> many Hindoos of rank has been for a quarter of a century i

---

## Second flagged pair (embedding similarity 0.7371)

### 🔴 Suspicious text (the submitted document)

> prestige, Englishmen here would long ago have adopted the Indian costume. I may mention incidentally
> that I do not go about Champaran bare headed. I do avoid shoes for sacred reasons. But I find too
> that it is more natural and healthier to avoid them whenever possible. He was not in families with
> he saw; he was not early experts is touched upon race. "to-morrow?" them said this Control. "we
> shall be ambitious **to have** point, dream," we state this Education. C., "to satisfied dream for
> we to-morrow." He has still not the courage, in spite of most admirable contact with me, to discard
> his semi-anglicised dress and whenever he goes to see officia

### 🟢 Source text (the original)

> that he should resume them on taking his leave. **To have** appeared in public with uncovered feet
> would have been a gross breach of propriety. Fine old Hindoo gentlemen, all of the olden time, find
> it difficult to express their mingled contempt, indignation, and regret for the innovation which
> substitutes the Cheapside shoe for the ceremonial slipper, or permits the wearing of the latter in a
> Sahib's office or drawing-room. It shows, they say, that the natives are losing their respect for
> the Sahibs. And yet the British authorities stupidly sanction it, even set the seal of fashion upon
> it, by allowing natives of rank, who visit Government House

---

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*

---

## LLM verdict and reasoning

**Score:** 0.95 · **is_likely_source:** True

> "The suspicious text contains verbatim fragments of the source candidate text embedded within a garbled/corrupted string. Specifically, phrases like 'I do not go about Champaran bare headed', 'I do avoid shoes for sacred reasons', 'it is more natural and healthier to avoid them whenever possible', 'I believe that our copying of the European dress is a sign of our degradation, humiliation and our weakness', and 'the dress I wear in Champaran is the dress I have always worn in India' are direct matches or near-verbatim extractions from the source material. The suspicious text appears to be an automated or corrupted reconstruction using segments of the original source."

**Not verified — reasoning cites text absent from the shown evidence.** None of the
25 suspicious/source chunk pairs actually sent to the model (searched exhaustively)
contain the quoted phrases above on the *source* side. Every source-side pair in this
candidate is third-person narration about someone else's turban and footwear customs
(e.g. "he made haste to don his flat turban... might resent his bare head as an
indignity"), not the first-person "I do not go about Champaran bare headed" passage
the suspicious text actually contains and the reasoning quotes. The suspicious-side
phrases are real (they are a well-known first-person passage), but the model's claim
that they are "direct matches... from the source material it was shown" does not
hold up against the actual pairs in this run's prompt. This suggests the model may be
drawing on knowledge or pattern-matching from outside the given context rather than
purely the evidence pairs, which is a genuine finding: a confirmation score/verdict
can look correct while the stated justification for it is not trustworthy on
inspection.
