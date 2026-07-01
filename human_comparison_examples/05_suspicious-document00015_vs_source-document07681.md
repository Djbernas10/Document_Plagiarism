# Example 5 — suspicious-document00015 vs source-document07681

**Obfuscation type:** Heavy paraphrase (obfuscation: high)  
**Source document:** `part16__source-document07681.txt`  
**Top pair embedding similarity:** 0.7574  
**LLM verdict:** confirmed as plagiarism source (score ≥ 0.85, Gemma 4 26B)

> A second, independent source confirmed for the same suspicious document 00015, showing the pipeline attributing different passages of one document to different origins.

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
