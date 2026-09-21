"""
Compute, per scope (1-80, 81-308, 1-308 for PAN 2011; plus the 10-doc custom
dataset), from the existing regenerated plagdet_summary.parquet files only
(no pipeline rerun):
  (a) plagiarized docs detected (>=1 TP), out of plagiarized total
  (b) clean docs with >=1 false positive detection, out of clean total

Also sanity-checks "firing documents" (docs with det_spans > 0) against the
sum of (a)'s count + (b)'s count, and against known values from the thesis.

Source: scripts/final/pipeline_results/plagdet_summary.parquet (PAN 2011) and
        scripts/final/pipeline_results_custom/plagdet_summary.parquet (custom)
  - gt_spans > 0  => plagiarized document (ground truth)
  - gt_spans == 0 => clean document (ground truth)
  - tp_chars > 0  => this document has >=1 true positive detection
  - det_spans > 0 => this document produced >=1 detection (regardless of TP/FP)
  For a clean doc, gt_spans==0 means tp_chars is always 0 by construction
  (no ground truth to overlap), so "clean doc has a detection" == det_spans>0
  == "clean doc has a false positive" (any detection on a clean doc is a FP).

Does not touch or rerun the LLM pipeline.
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
SUMMARY_PATH = SCRIPT_DIR / "pipeline_results" / "plagdet_summary.parquet"
CUSTOM_SUMMARY_PATH = SCRIPT_DIR / "pipeline_results_custom" / "plagdet_summary.parquet"
OUT_CSV = SCRIPT_DIR / "pipeline_results" / "detection_fp_rates_by_scope.csv"


def doc_num(doc_id: str) -> int:
    # e.g. "part1__suspicious-document00081.txt" -> 81
    return int("".join(ch for ch in doc_id.split("document")[-1] if ch.isdigit()))


def compute_scope(df: pd.DataFrame, lo: int, hi: int, label: str) -> dict:
    sub = df[(df["n"] >= lo) & (df["n"] <= hi)]

    plag = sub[sub["gt_spans"] > 0]
    clean = sub[sub["gt_spans"] == 0]

    plag_detected = plag[plag["tp_chars"] > 0]
    clean_fp = clean[clean["det_spans"] > 0]

    firing = sub[sub["det_spans"] > 0]

    n_plag = len(plag)
    n_clean = len(clean)
    n_plag_detected = len(plag_detected)
    n_clean_fp = len(clean_fp)
    n_firing = len(firing)

    detection_rate = n_plag_detected / n_plag if n_plag else 0.0
    fp_rate = n_clean_fp / n_clean if n_clean else 0.0

    # Sanity: firing docs == (plag docs that fired) + (clean docs with FP)
    # "Plag docs that fired" is det_spans>0 among plag docs, which need not be
    # identical to "plag docs detected" (tp_chars>0) if a plag doc fired but
    # produced only FPs against its own ground truth (det_spans>0, tp_chars==0).
    plag_fired = plag[plag["det_spans"] > 0]
    n_plag_fired = len(plag_fired)
    sum_check = n_plag_fired + n_clean_fp

    return {
        "scope": label,
        "lo": lo,
        "hi": hi,
        "plagiarized_docs": n_plag,
        "plagiarized_detected": n_plag_detected,
        "detection_rate": round(detection_rate, 4),
        "clean_docs": n_clean,
        "clean_docs_with_fp": n_clean_fp,
        "fp_rate": round(fp_rate, 4),
        "firing_docs": n_firing,
        "plag_docs_fired": n_plag_fired,
        "sum_plag_fired_plus_clean_fp": sum_check,
        "sum_matches_firing": sum_check == n_firing,
    }


def compute_custom_dataset() -> dict:
    df = pd.read_parquet(CUSTOM_SUMMARY_PATH)
    print(f"Loaded {CUSTOM_SUMMARY_PATH} ({len(df)} rows)")
    df["n"] = 1  # single scope, no doc-number split needed
    return compute_scope(df, df["n"].min(), df["n"].max(), "custom-10doc")


def main():
    df = pd.read_parquet(SUMMARY_PATH)
    print(f"Loaded {SUMMARY_PATH} ({len(df)} rows)")
    df["n"] = df["suspicious_doc_id"].map(doc_num)

    scopes = [
        (81, 308, "81-308"),
        (1, 80, "1-80"),
        (1, 308, "1-308"),
    ]

    results = [compute_scope(df, lo, hi, label) for lo, hi, label in scopes]
    results.append(compute_custom_dataset())

    print()
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    for r in results:
        print(f"[{r['scope']}] plag={r['plagiarized_docs']} clean={r['clean_docs']} "
              f"firing={r['firing_docs']}  plag_fired={r['plag_docs_fired']} + clean_fp={r['clean_docs_with_fp']} "
              f"= {r['sum_plag_fired_plus_clean_fp']}  matches_firing={r['sum_matches_firing']}")

    overall = next(r for r in results if r["scope"] == "1-308")
    checks_passed = True

    # NOTE: the thesis originally cited 87/156, computed from a stale copy of
    # analytics_summary.parquet that still had pre-regeneration detection counts
    # for suspicious-document00010/00012. The corrected, current per-document
    # data (character-overlap TP, tp_chars>0) gives 85/156 -- see the
    # "Update (2026-09-21)" section of 308_docs_full_v3/README.md. The check
    # below validates against the corrected value, not the stale thesis figure.
    if not (overall["plagiarized_detected"] == 85 and overall["plagiarized_docs"] == 156):
        print(f"\nFAIL: 1-308 plagiarized detected expected 85/156 (corrected value), got "
              f"{overall['plagiarized_detected']}/{overall['plagiarized_docs']}")
        checks_passed = False

    if not (overall["clean_docs_with_fp"] == 18 and overall["clean_docs"] == 152):
        print(f"\nFAIL: 1-308 clean FP expected 18/152, got "
              f"{overall['clean_docs_with_fp']}/{overall['clean_docs']}")
        checks_passed = False

    expected_firing = {"1-80": 33, "81-308": 79, "1-308": 112}
    for r in results:
        if r["scope"] not in expected_firing:
            continue  # custom-10doc has no pre-existing expected value to check against
        exp = expected_firing[r["scope"]]
        if r["firing_docs"] != exp:
            print(f"\nFAIL: {r['scope']} firing docs expected {exp}, got {r['firing_docs']}")
            checks_passed = False
        if not r["sum_matches_firing"]:
            print(f"\nFAIL: {r['scope']} plag_fired + clean_fp "
                  f"({r['sum_plag_fired_plus_clean_fp']}) != firing_docs ({r['firing_docs']})")
            checks_passed = False

    print()
    if checks_passed:
        print("ALL SANITY CHECKS PASSED")
    else:
        print("ONE OR MORE SANITY CHECKS FAILED -- see FAIL lines above. Not adjusting anything.")

    print()
    print("=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)

    out_rows = []
    for r in results:
        out_rows.append({
            "scope": r["scope"],
            "plagiarized_docs": r["plagiarized_docs"],
            "plagiarized_detected": r["plagiarized_detected"],
            "detection_rate": r["detection_rate"],
            "clean_docs": r["clean_docs"],
            "clean_docs_with_fp": r["clean_docs_with_fp"],
            "fp_rate": r["fp_rate"],
        })
    out_df = pd.DataFrame(out_rows)

    def fmt_pct(x):
        return f"{x*100:.1f}%"

    display_df = out_df.copy()
    display_df["detection_rate"] = display_df["detection_rate"].map(fmt_pct)
    display_df["fp_rate"] = display_df["fp_rate"].map(fmt_pct)
    print(display_df.to_string(index=False))

    out_df.to_csv(OUT_CSV, index=False)
    print()
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
