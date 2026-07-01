# Example 4 — suspicious-document00015 vs source-document00853

**Obfuscation type:** Heavy paraphrase (obfuscation: high)  
**Source document:** `part2__source-document00853.txt`  
**Top pair embedding similarity:** 0.7678  
**LLM verdict:** confirmed as plagiarism source (score ≥ 0.85, Gemma 4 26B)

> One of several sources for this multi-source suspicious document. Demonstrates confirmation under strong obfuscation where lexical overlap is partial.

> **Bold** marks the phrases that match between the two texts.

---

## Flagged passage (highest-similarity chunk pair)

### 🔴 Suspicious text (the submitted document)

> cognize you think a compartment could cry, simper. I project the Conveyance is this situation, you?
> "" Pontresina. Ives! "Thither we alter grave; her get headdress was the habits, she laughed." i've
> be jar my needlework, really, "she state," i care't forget. Cannot so the years. "A advancement; the
> ambulance and name and conductor cry the agnomen." specific Position! "she name." the acceleration.
> "That him was not the activity. There was the way. Each stories invite without force to cognize
> down; we were jostle to wear up. **St**. **Ives**." Ascent down those bar, "she wore. I did t, him
> tilt to gimmick. We establish ourselves today punish t wore on

### 🟢 Source text (the original)

> "Macclesfield," he said very decidedly. The elderly relative was fidgeting to say hers. I could have
> guessed it would be **St**. **Ives**. The conductress made her way from one end to the other. "All
> got towns?" she asked. "You, Sir? Pernambuco? I do wish you'd stick to English names. Are you all
> ready?" She rang the bell. "Now," she said, "the gentleman on the stool has to catch. The Post is
> going from Paris to Pontresina." I rose and looked wildly down the car. The flapper was beckoning
> slightly. Her contemptuous boredom had vanished, and she looked a merry child again. I rushed,
> stumbled, rocked into her place; she sank with a gasp into mine. "Yor

---

## Second flagged pair (embedding similarity 0.7440)

### 🔴 Suspicious text (the submitted document)

> cognize you think a compartment could cry, simper. I project the Conveyance is this situation, you?
> "" Pontresina. Ives! "Thither we alter grave; her get headdress was the habits, she laughed." i've
> be jar my needlework, really, "she state," i care't forget. Cannot so the years. "A advancement; the
> ambulance and name and conductor cry the agnomen." specific Position! "she name." the acceleration.
> "That him was not the activity. There was the way. Each stories invite without force to cognize
> down; we were jostle to wear up. **St**. **Ives**." Ascent down those bar, "she wore. I did t, him
> tilt to gimmick. We establish ourselves today punish t wore on

### 🟢 Source text (the original)

> "I'm here," he said. "Macclesfield for ever!" The flapper had scrambled up the front staircase
> against the rules. She cast herself down beside Macclesfield. "Here I am, old dear," she exclaimed.
> "I left York simply jammed in the wedge. Oh, isn't it fun? I never laughed so much. We never can be
> serious with each other after this, can we?" **St**. **Ives** nodded. "I'll never forget Pontresina
> climbing the rail," she said. "I used to think him so haughty; now--" "Albemarle Road--don't you
> want Albemarle Road?" the conductress was asking me. She spoke very loudly. "Pontresina--I'm
> Pontresina," I answered. "This is Albemarle Road. If you're going on it'l

---

*Extracted from the final-configuration run (`gemma4:26b`, LLM threshold 0.85, 25 chunk pairs per candidate) — `pipeline_results/llm_debug/`. Text is shown as stored after preprocessing (cleaning + normalisation); bold spans are computed by longest-common-subsequence token matching.*
