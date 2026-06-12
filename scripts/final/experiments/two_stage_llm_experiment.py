"""
EXPERIMENT: Two-stage LLM verification for synonym-swap obfuscation detection
==============================================================================
Status   : TESTED and REVERTED — did not improve F1
Tested on: part1__suspicious-document00005.txt (obfuscation=low, synonym-swap)
Result   : Recall fixed (0→1) but FP exploded (0→5) — net F1 worse

WHY IT FAILED
-------------
Doc 00005 is about historical military events (Marlborough, Flanders, Louis XIV).
The Stage 2 entity/fact prompt confirmed all 3 candidates because all source docs
from that era share the same named entities. The LLM cannot distinguish which
specific document was copied from when all candidates describe the same events.

Stage 1 only:  recall=0, FP=0,  F1=0.00
Stage 2 added: recall=1, FP=5,  F1=0.29  (6 detected spans, 1 correct, 5 wrong)

CONCLUSION
----------
Two-stage approach works in theory but fails when candidate source docs are
thematically homogeneous (same era, same characters, same events). The named
entity signal is not discriminative enough to identify the specific source.

FUTURE WORK (thesis Chapter 6)
-------------------------------
A better approach would combine:
  1. Character n-gram similarity (Jaccard on char trigrams) as a pre-filter
     to flag synonym-swap cases before the LLM sees them
  2. Sentence-structure similarity (dependency parse comparison) — swap
     preserves syntax even when lexis changes
  3. Back-translation round-trip — synonym-swapped text often converges
     back toward the original after translate→retranslate

These would need their own calibration runs and are out of scope for this thesis.

IMPLEMENTATION (saved here for reference)
------------------------------------------
"""

import time
import json
import re

# Constants used in this experiment
LLM_SCORE_THRESHOLD  = 0.95
LLM_STAGE2_THRESHOLD = 0.85
OLLAMA_MODEL         = "gemma4:e4b"


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM response: {raw[:300]}")
    clean = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match.group())
    return json.loads(clean)


def score_source_doc_stage2(source_doc_id: str, pairs: list[dict]) -> dict:
    """Stage 2 fallback: detects synonym-swap obfuscation by checking shared facts/entities/narrative."""
    from ollama import chat

    pairs_text = "\n\n".join([
        f"[Pair {i+1}]\n"
        f"SUSPICIOUS: {p['suspicious_text'][:600]}\n"
        f"SOURCE CANDIDATE: {p['source_text'][:600]}"
        for i, p in enumerate(pairs)
    ])

    prompt = (
        f"You are a plagiarism detection expert specialising in obfuscated plagiarism.\n"
        f"Below are {len(pairs)} text pair(s). The suspicious text may have had words replaced "
        f"with synonyms or near-synonyms, but the underlying content may still be stolen.\n\n"
        f"{pairs_text}\n\n"
        f"Your task: ignore word choice entirely. Focus ONLY on whether the two texts describe "
        f"the SAME specific facts, named entities (people, places, organisations), events, and "
        f"narrative sequence.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Score HIGH (>= 0.85) if the texts describe the same specific events, people, or places "
        f"in the same order, even if every content word has been replaced with a synonym.\n"
        f"- Score LOW (< 0.40) if the texts merely share a topic or general theme without describing "
        f"the same specific facts or named entities.\n"
        f"- Named entities (proper nouns: names, places, titles) that survive in both texts are "
        f"strong evidence of plagiarism — weight them heavily.\n"
        f"- If fewer than 3 pairs share specific facts or entities, score below 0.40.\n\n"
        f"Respond with ONLY a JSON object — no markdown, no explanation — with keys: "
        f"score (float 0-1), is_likely_source (bool), reasoning (string)."
    )

    t0 = time.time()
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "seed": 42},
        think=False,
    )
    elapsed = time.time() - t0

    data = _parse_json(response.message.content)
    return {
        "source_doc_id":        source_doc_id,
        "llm_score":            float(data.get("score", 0.0)),
        "llm_is_likely_source": bool(data.get("is_likely_source", False)),
        "llm_reasoning":        data.get("reasoning", ""),
        "elapsed_s":            round(elapsed, 1),
        "stage":                2,
    }


# --- How it was wired into run_text_alignment (after Stage 1 confirmation block) ---
#
# confirmed_ids = set(
#     llm_scores_df.loc[llm_scores_df["llm_score"] >= LLM_SCORE_THRESHOLD, "source_doc_id"]
# )
# print(f"  Confirmed sources  : {len(confirmed_ids)} / {len(llm_scores_df)} (Stage 1)")
#
# if not confirmed_ids:
#     rejected_ids = set(llm_scores_df["source_doc_id"]) - confirmed_ids
#     stage2_rows = []
#     for source_doc_id in rejected_ids:
#         group = top_pairs[top_pairs["source_doc_id"] == source_doc_id]
#         pairs = group[["suspicious_text", "source_text", "embedding_score"]].to_dict("records")
#         try:
#             stage2_rows.append(score_source_doc_stage2(source_doc_id, pairs))
#         except Exception as e:
#             stage2_rows.append({
#                 "source_doc_id": source_doc_id, "llm_score": 0.0,
#                 "llm_is_likely_source": False, "llm_reasoning": f"Error: {e}",
#                 "elapsed_s": 0.0, "stage": 2,
#             })
#
#     if stage2_rows:
#         stage2_df = pd.DataFrame(stage2_rows).sort_values("llm_score", ascending=False).reset_index(drop=True)
#         confirmed_ids = set(
#             stage2_df.loc[stage2_df["llm_score"] >= LLM_STAGE2_THRESHOLD, "source_doc_id"]
#         )
#         llm_scores_df = pd.concat([llm_scores_df, stage2_df], ignore_index=True)
