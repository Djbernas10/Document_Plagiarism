"""Isolated experiment: re-send the already-dumped LLM debug pairs for
suspicious-document00007.txt to Ollama with a tweaked prompt, to see whether a
prompt change could have fixed the false rejection of source-document00018.txt
(a genuine verbatim match per ground truth) without touching run_pipeline.py
or source_retrieval_branches.py (which would also affect the already-finalized
PAN2011 308-doc results).

Does NOT call retrieval or modify any shared pipeline code. Reuses the pairs
already dumped to pipeline_results_custom/llm_debug/suspicious-document00007.txt/.

Run from project root:
    python scripts/final/experiments/rescore_doc00007_prompt_variant.py
"""

import json
import re
import time
from pathlib import Path

from ollama import chat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEBUG_DIR = PROJECT_ROOT / "scripts" / "final" / "pipeline_results_custom" / "llm_debug" / "suspicious-document00007.txt"

OLLAMA_MODEL = "gemma4:26b"

GT_SOURCES = {"source-document00018.txt", "source-document00016.txt"}


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM response: {raw[:300]}")
    clean = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", match.group())
    return json.loads(clean)


def build_prompt_variant(pairs: list[dict]) -> str:
    """
    Differences from the production prompt (run_pipeline.score_source_doc):
      1. Pairs deduplicated by (suspicious_text, source_text) — the original 25-pair
         batch had many near-identical repeats (same suspicious chunk vs different
         low-relevance source chunks), which may dilute attention on the one strong pair.
      2. Pairs sorted by embedding_score and explicitly labeled with their rank, so the
         model is told which pair is most likely to matter.
      3. An explicit instruction: "Read EVERY pair before deciding — a single pair with
         clear overlap is sufficient to justify a high score even if other pairs show no
         overlap," since the production failure looked like the model anchored on the
         majority of weak pairs and ignored the one strong pair.
    """
    seen = set()
    deduped = []
    for p in sorted(pairs, key=lambda p: -p["embedding_score"]):
        key = (p["suspicious_text"], p["source_text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    pairs_text = "\n\n".join(
        f"[Pair {i+1} | embedding_similarity={p['embedding_score']:.3f}]\n"
        f"SUSPICIOUS: {p['suspicious_text'][:600]}\n"
        f"SOURCE CANDIDATE: {p['source_text'][:600]}"
        for i, p in enumerate(deduped)
    )

    return (
        f"You are a strict plagiarism detection expert.\n"
        f"Below are {len(deduped)} DISTINCT text pair(s) (duplicates removed), sorted by "
        f"embedding similarity (highest first). Each pair shows a chunk from a SUSPICIOUS "
        f"document alongside a chunk from a CANDIDATE SOURCE document.\n\n"
        f"{pairs_text}\n\n"
        f"Your task: score the likelihood that the suspicious text was copied from this source.\n\n"
        f"IMPORTANT: Read EVERY pair before deciding. A single pair with clear verbatim or "
        f"near-verbatim overlap is sufficient to justify a high score, even if most other "
        f"pairs show no overlap at all — low-ranked pairs are noise from imperfect retrieval, "
        f"not evidence against plagiarism.\n\n"
        f"SCORING RUBRIC — you MUST use one of these exact values:\n"
        f"  0.00 — No overlap in ANY pair. Texts share only a topic or general theme.\n"
        f"  0.25 — Weak overlap. Similar vocabulary, different sentence structure. "
        f"         OR same-author reuse (different volumes of the same work).\n"
        f"  0.50 — Moderate overlap. Some shared phrases but unclear if copied.\n"
        f"  0.85 — Synonym-swap obfuscation. Same sentence structure and narrative sequence "
        f"         with content words replaced by synonyms. Named entities still match.\n"
        f"  0.95 — Near-verbatim. At least one pair shows verbatim or near-verbatim copying "
        f"         with unique shared phrases, names, or sequences.\n"
        f"  1.00 — Exact verbatim copy in at least one pair.\n\n"
        f"RULES:\n"
        f"- Pick 0.85 if sentence structure is preserved with synonym substitution.\n"
        f"- Pick 0.00-0.25 if texts share a topic/era but sentence structures differ.\n"
        f"- Same-author reuse (different volumes/editions of same work) = 0.25, NOT plagiarism.\n"
        f"- You MUST return exactly one of: 0.00, 0.25, 0.50, 0.85, 0.95, 1.00.\n\n"
        f"Respond with ONLY a JSON object — no markdown, no explanation — with keys: "
        f"score (float 0-1), is_likely_source (bool), reasoning (string)."
    )


def score_with_variant(source_doc_id: str, pairs: list[dict]) -> dict:
    prompt = build_prompt_variant(pairs)
    t0 = time.time()
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "top_p": 0.95, "top_k": 64, "seed": 42},
        think=False,
    )
    elapsed = time.time() - t0
    data = _parse_json(response.message.content)
    return {
        "source_doc_id": source_doc_id,
        "llm_score": float(data.get("score", 0.0)),
        "llm_is_likely_source": bool(data.get("is_likely_source", False)),
        "llm_reasoning": data.get("reasoning", ""),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    debug_files = sorted(DEBUG_DIR.glob("*.json"))
    if not debug_files:
        raise FileNotFoundError(f"No debug dumps found in {DEBUG_DIR}")

    print(f"Re-scoring {len(debug_files)} candidate(s) for suspicious-document00007.txt with prompt variant:\n")
    for path in debug_files:
        with open(path, encoding="utf-8") as f:
            dump = json.load(f)
        source_doc_id = dump["source_doc_id"]
        pairs = dump["pairs"]

        result = score_with_variant(source_doc_id, pairs)
        is_gt = source_doc_id in GT_SOURCES
        tag = "[GT SOURCE]" if is_gt else "[distractor]"
        print(f"{tag} {source_doc_id}")
        print(f"  score={result['llm_score']:.3f}  likely={result['llm_is_likely_source']}  t={result['elapsed_s']}s")
        print(f"  reasoning: {result['llm_reasoning'][:200]}")
        print()


if __name__ == "__main__":
    main()
