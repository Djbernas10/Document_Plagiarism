"""
Compute the OFFICIAL PAN 2011 (Potthast et al.) plagdet score, pooled across
the ENTIRE corpus in a single pass -- not per-document averaging (macro) and
not raw pooled character-count ratios (micro), which is what compute_plagdet.py
currently reports.

Definition (Potthast et al. 2011, "Overview of the 3rd International
Competition on Plagiarism Detection", CLEF2011wn-PAN-PotthastEt2011a.pdf,
Section 1.2, eq. 1-2 -- verified directly against the PDF in this repo):

  S = the set of ALL ground-truth plagiarism cases in the corpus
  R = the set of ALL detections reported by the detector

  s (case reference)     = set of character positions of (splg, dplg) i.e. the
                            suspicious-side span of GT case s (source side is
                            used only to test whether r "detects" s)
  r (detection reference) = set of character positions of the suspicious-side
                            span of detection r

  s ⊓ r = s ∩ r if r detects s, else ∅   (a detection "detects" a case iff
          they overlap on the suspicious side AND on the source side AND
          reference the same source document -- see compute_plagdet.py's
          `intervals_overlap` checks, reused here unchanged)

  prec(S,R) = (1/|R|) * sum_{r in R} | union_{s in S} (s ⊓ r) | / |r|
  rec(S,R)  = (1/|S|) * sum_{s in S} | union_{r in R} (s ⊓ r) | / |s|

  IMPORTANT: S = ground truth, R = detections. Precision sums over
  DETECTIONS r (each one weighted by its own length |r|, using how much of
  it is covered by the union of GT cases it overlaps). Recall sums over
  GT CASES s (each one weighted by its own length |s|, using how much of it
  is covered by the union of detections that overlap it). This is the
  opposite S/R convention from the formula the user's secondary source
  (Franco-Salvador reproduction) used, which had precision "summed over
  ground-truth case r in R" -- that reproduction swapped/mislabeled the S/R
  roles relative to the original PAN paper. This script follows the ORIGINAL
  PAN paper's S=GT, R=detections convention, confirmed directly from the PDF.

  gran(S,R) = (1/|S_R|) * sum_{s in S_R} |R_s|
    where S_R = GT cases detected by >=1 detection, R_s = detections of s
    (pooled over the whole corpus, not averaged per document)

  plagdet(S,R) = F1 / log2(1 + gran(S,R))

Each case/detection contributes ONE equally-weighted term to its average
regardless of character length (length only scales the *within-term*
overlap fraction, not the weight of the term in the sum) -- this is what
distinguishes it from "micro" pooled-character-count ratios, which weight
every case/detection by its raw character length instead of counting it once.

Data source: reuses per_doc/*.parquet (detections) and the same GT-XML
loader as compute_plagdet.py (load_gt_spans), read-only. Does not touch or
rerun the LLM pipeline.

Usage:
  python compute_plagdet_official.py --run 308            # PAN 308-doc run
  python compute_plagdet_official.py --run custom          # 10-doc custom set
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from compute_plagdet import (
    GT_XML_DIR,
    CUSTOM_GT_XML_DIR,
    PER_DOC_DIR,
    load_gt_spans,
    intervals_overlap,
    overlap_chars,
)

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "pipeline_results"
RESULTS_DIR_CUSTOM = SCRIPT_DIR / "pipeline_results_custom"


def classify_case_category(gt: dict) -> str:
    """
    Same classification rule as compute_plagdet.py's per-obfuscation
    breakdown (kept in sync deliberately -- duplicated rather than imported
    since compute_plagdet.py applies it inline inside its main() loop rather
    than as a standalone function).
    """
    obf = gt.get("obfuscation", "none")
    typ = gt.get("type", "")
    manual_obf = gt.get("manual_obfuscation", "false")
    if "translation" in typ:
        return "translation-manual" if manual_obf == "true" else "translation-auto"
    elif typ == "":
        return "none"
    elif obf == "none":
        return "none"
    elif obf == "low" and "artificial" in typ:
        return "paraphrase-auto-low"
    elif obf == "high" and "artificial" in typ:
        return "paraphrase-auto-high"
    else:
        return f"{obf}"


def build_case_and_detection_records(doc_ids, per_doc_dir, gt_xml_dir, flat_gt_names=False):
    """
    For every document, load its GT cases and detections, and emit two flat
    lists of records pooled across the WHOLE corpus:
      cases:      one dict per GT case s, with its suspicious span + which
                  detections (globally unique ids) overlap it and by how much
      detections: one dict per detection r, with its suspicious span + which
                  cases (globally unique ids) overlap it and by how much

    A GT case and a detection are linked ("r detects s") using the same
    3-part test as compute_plagdet_doc: suspicious-side overlap, source-side
    overlap, same source document -- but here every case/detection keeps a
    globally unique id so unions can be computed across the ENTIRE corpus
    instead of within one document.

    Each case record also carries a "category" field (PAN 2011 Table 3
    obfuscation category, same classifier as compute_plagdet.py's
    per-obfuscation breakdown) so per-category pooling can filter on it.
    """
    cases = []
    detections = []
    excluded_docs = []

    case_uid = 0
    det_uid = 0

    for doc_id in doc_ids:
        gt_spans = load_gt_spans(doc_id, gt_xml_dir=gt_xml_dir)

        per_doc_path = per_doc_dir / f"{doc_id}.parquet"
        det_df = pd.read_parquet(per_doc_path) if per_doc_path.exists() else pd.DataFrame()

        doc_detections = det_df.to_dict("records") if not det_df.empty else []
        for d in doc_detections:
            parts = d["source_doc_id"].split("__")
            d["src_basename"] = parts[1] if len(parts) > 1 else d["source_doc_id"]

        # Register detections for this doc with globally unique ids
        local_det_ids = []
        for d in doc_detections:
            detections.append({
                "uid": det_uid,
                "doc_id": doc_id,
                "susp_start": d["suspicious_start_char"],
                "susp_end": d["suspicious_end_char"],
                "src_basename": d["src_basename"],
                "src_start": d["source_start_char"],
                "src_end": d["source_end_char"],
                "overlapping_case_uids": [],  # filled below
            })
            local_det_ids.append(det_uid)
            det_uid += 1

        # Register GT cases for this doc with globally unique ids
        local_case_records = []
        for gt in gt_spans:
            gt_susp_start = gt["susp_offset"]
            gt_susp_end = gt["susp_offset"] + gt["susp_length"]
            gt_src_start = gt["src_offset"]
            gt_src_end = gt["src_offset"] + gt["src_length"]
            case_rec = {
                "uid": case_uid,
                "doc_id": doc_id,
                "susp_start": gt_susp_start,
                "susp_end": gt_susp_end,
                "src_reference": gt["src_reference"],
                "src_start": gt_src_start,
                "src_end": gt_src_end,
                "category": classify_case_category(gt),
                "overlapping_det_uids": [],  # filled below
            }
            cases.append(case_rec)
            local_case_records.append(case_rec)
            case_uid += 1

        # Link detections <-> cases within this doc (detection can only ever
        # overlap cases from the same doc, since offsets are doc-local)
        for case_rec in local_case_records:
            for di, d in zip(local_det_ids, doc_detections):
                if d["src_basename"] != case_rec["src_reference"]:
                    continue
                if not intervals_overlap(d["suspicious_start_char"], d["suspicious_end_char"],
                                          case_rec["susp_start"], case_rec["susp_end"]):
                    continue
                if not intervals_overlap(d["source_start_char"], d["source_end_char"],
                                          case_rec["src_start"], case_rec["src_end"]):
                    continue
                case_rec["overlapping_det_uids"].append(di)
                detections[di]["overlapping_case_uids"].append(case_rec["uid"])

    return cases, detections


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def compute_official_plagdet(cases, detections):
    """
    prec(S,R) = (1/|R|) * sum_r  |union_{s detects r}(s ∩ r)| / |r|
    rec(S,R)  = (1/|S|) * sum_s  |union_{r detects s}(s ∩ r)| / |s|
    gran(S,R) = (1/|S_R|) * sum_{s in S_R} |R_s|
    """
    cases_by_uid = {c["uid"]: c for c in cases}

    # --- Precision: one term per detection r ---
    prec_terms = []
    for d in detections:
        r_len = d["susp_end"] - d["susp_start"]
        if r_len <= 0:
            continue
        overlapping_case_intervals = [
            (cases_by_uid[cu]["susp_start"], cases_by_uid[cu]["susp_end"])
            for cu in d["overlapping_case_uids"]
        ]
        merged = merge_intervals(overlapping_case_intervals)
        covered = sum(overlap_chars(s, e, d["susp_start"], d["susp_end"]) for s, e in merged)
        prec_terms.append(covered / r_len)
    precision = sum(prec_terms) / len(detections) if len(detections) > 0 else 0.0

    # --- Recall: one term per GT case s ---
    rec_terms = []
    detections_by_uid = {d["uid"]: d for d in detections}
    for c in cases:
        s_len = c["susp_end"] - c["susp_start"]
        if s_len <= 0:
            continue
        overlapping_det_intervals = [
            (detections_by_uid[du]["susp_start"], detections_by_uid[du]["susp_end"])
            for du in c["overlapping_det_uids"]
        ]
        merged = merge_intervals(overlapping_det_intervals)
        covered = sum(overlap_chars(s, e, c["susp_start"], c["susp_end"]) for s, e in merged)
        rec_terms.append(covered / s_len)
    recall = sum(rec_terms) / len(cases) if len(cases) > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- Granularity: pooled over the whole corpus, not per-document averaged ---
    detected_cases = [c for c in cases if len(c["overlapping_det_uids"]) > 0]
    if detected_cases:
        granularity = sum(len(c["overlapping_det_uids"]) for c in detected_cases) / len(detected_cases)
    else:
        granularity = 1.0

    plagdet = f1 / math.log2(1 + granularity) if f1 > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "granularity": granularity,
        "plagdet": plagdet,
        "n_cases": len(cases),
        "n_detections": len(detections),
        "n_cases_detected": len(detected_cases),
    }


def compute_official_precision_only(cases, detections):
    """
    Same precision term definition as compute_official_plagdet, exposed
    standalone for the "conditional precision" subset computation (restricted
    to a filtered set of cases/detections rather than the whole corpus).
    """
    cases_by_uid = {c["uid"]: c for c in cases}
    prec_terms = []
    for d in detections:
        r_len = d["susp_end"] - d["susp_start"]
        if r_len <= 0:
            continue
        overlapping_case_intervals = [
            (cases_by_uid[cu]["susp_start"], cases_by_uid[cu]["susp_end"])
            for cu in d["overlapping_case_uids"]
        ]
        merged = merge_intervals(overlapping_case_intervals)
        covered = sum(overlap_chars(s, e, d["susp_start"], d["susp_end"]) for s, e in merged)
        prec_terms.append(covered / r_len)
    precision = sum(prec_terms) / len(detections) if len(detections) > 0 else 0.0
    return precision, len(detections)


def find_mismatched_docs(doc_ids, plagdet_summary_df, per_doc_dir):
    """
    Compare each doc's det_spans/det_chars in the archived plagdet_summary
    against what's actually in the live per_doc/ cache today. Returns the
    list of doc_ids whose cached detections no longer match the summary that
    produced the previously-reported macro/micro figures.
    """
    summary = plagdet_summary_df.set_index("suspicious_doc_id")
    mismatched = []
    for doc_id in doc_ids:
        row = summary.loc[doc_id]
        p = per_doc_dir / f"{doc_id}.parquet"
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        det_spans_cache = len(df)
        det_chars_cache = (df["suspicious_end_char"] - df["suspicious_start_char"]).sum() if len(df) > 0 else 0
        if det_spans_cache != row["det_spans"] or det_chars_cache != row["det_chars"]:
            mismatched.append(doc_id)
    return mismatched


def run(run_name, summary_path, per_doc_dir, gt_xml_dir, out_dir, plag_only=True, doc_id_filter=None):
    print("=" * 70)
    print(f"OFFICIAL PAN 2011 PLAGDET -- {run_name}")
    print("=" * 70)

    summary = pd.read_parquet(summary_path)
    all_doc_ids = summary["suspicious_doc_id"].tolist()
    if doc_id_filter is not None:
        all_doc_ids = [d for d in all_doc_ids if doc_id_filter(d)]
        summary = summary[summary["suspicious_doc_id"].isin(all_doc_ids)]

    mismatched = find_mismatched_docs(all_doc_ids, summary, per_doc_dir)
    clean_doc_ids = [d for d in all_doc_ids if d not in mismatched]

    print(f"Total docs in archived run: {len(all_doc_ids)}")
    print(f"Docs whose live per_doc/ cache matches the archived summary: {len(clean_doc_ids)}")
    if mismatched:
        print(f"EXCLUDED (cache no longer matches archived summary, likely overwritten by a later run):")
        for d in mismatched:
            print(f"  - {d}")

    # Official formula pools GT cases and detections across the corpus.
    # We include ALL clean docs (both plagiarised and clean-with-FPs) in R
    # and S, matching the PAN definition where S/R are corpus-wide sets, not
    # restricted to plagiarised documents only -- unlike this codebase's
    # "macro plagdet (plag only)"/"micro plagdet" which restrict to plag_df
    # for some totals. We report both variants for direct comparability.
    doc_ids_for_pooling = clean_doc_ids

    cases, detections = build_case_and_detection_records(
        doc_ids_for_pooling, per_doc_dir, gt_xml_dir
    )

    result_all = compute_official_plagdet(cases, detections)

    print()
    print(f"[All {len(clean_doc_ids)} matching docs, incl. clean docs in R/S pooling]")
    print(f"  |S| (GT cases)      : {result_all['n_cases']}")
    print(f"  |R| (detections)    : {result_all['n_detections']}")
    print(f"  Official precision  : {result_all['precision']:.4f}")
    print(f"  Official recall     : {result_all['recall']:.4f}")
    print(f"  Official F1         : {result_all['f1']:.4f}")
    print(f"  Official granularity: {result_all['granularity']:.4f}")
    print(f"  Official plagdet    : {result_all['plagdet']:.4f}")

    # Plagiarised-only pooling (S restricted to docs with gt_spans > 0, R
    # restricted to detections from those same docs) for direct comparability
    # against "Macro plagdet (plag only)" / "Micro plagdet" which both
    # restrict to plagiarised documents for at least part of their totals.
    plag_doc_ids = summary[summary["suspicious_doc_id"].isin(clean_doc_ids) &
                            (summary["gt_spans"] > 0)]["suspicious_doc_id"].tolist()
    cases_plag, detections_plag = build_case_and_detection_records(
        plag_doc_ids, per_doc_dir, gt_xml_dir
    )
    result_plag = compute_official_plagdet(cases_plag, detections_plag)

    print()
    print(f"[Plagiarised docs only, {len(plag_doc_ids)} docs -- R/S pooled from these docs' cases/detections only]")
    print(f"  |S| (GT cases)      : {result_plag['n_cases']}")
    print(f"  |R| (detections)    : {result_plag['n_detections']}")
    print(f"  Official precision  : {result_plag['precision']:.4f}")
    print(f"  Official recall     : {result_plag['recall']:.4f}")
    print(f"  Official F1         : {result_plag['f1']:.4f}")
    print(f"  Official granularity: {result_plag['granularity']:.4f}")
    print(f"  Official plagdet    : {result_plag['plagdet']:.4f}")

    # --- Conditional precision: restrict to docs with >=1 detection ---
    # "Documents where it fires at all" = det_spans > 0 in the summary.
    # Uses the corpus-wide (all_docs) pooling scope, matching how the
    # existing "0.310 overall" figure was scoped (all 308 docs, not
    # plag-only) before this conditional subset is carved out of it.
    firing_doc_ids = summary[summary["suspicious_doc_id"].isin(clean_doc_ids) &
                              (summary["det_spans"] > 0)]["suspicious_doc_id"].tolist()
    cases_firing, detections_firing = build_case_and_detection_records(
        firing_doc_ids, per_doc_dir, gt_xml_dir
    )
    cond_precision, cond_n_det = compute_official_precision_only(cases_firing, detections_firing)

    print()
    print(f"[Conditional precision: {len(firing_doc_ids)} docs with >=1 detection out of {len(clean_doc_ids)}]")
    print(f"  |R| (detections, firing docs only): {cond_n_det}")
    print(f"  Official conditional precision    : {cond_precision:.4f}")

    # --- Per-category official plagdet ---
    # S_category = GT cases classified into this category (pooled across ALL
    # matching docs, not just docs "of" that category, since one document can
    # contain cases from multiple categories). R_category = only the
    # detections that overlap >=1 case in S_category (a detection has no
    # category of its own; it inherits relevance from what it detects).
    cases_full, detections_full = build_case_and_detection_records(
        clean_doc_ids, per_doc_dir, gt_xml_dir
    )
    detections_full_by_uid = {d["uid"]: d for d in detections_full}

    categories = sorted(set(c["category"] for c in cases_full))
    category_results = {}
    print()
    print(f"[Per-category official plagdet, pooled over {len(clean_doc_ids)} docs]")
    for cat in categories:
        cat_cases = [c for c in cases_full if c["category"] == cat]
        cat_det_uids = sorted(set(du for c in cat_cases for du in c["overlapping_det_uids"]))
        cat_detections = [detections_full_by_uid[du] for du in cat_det_uids]

        # Restrict each case's/detection's cross-links to this category's own
        # subset so unions inside compute_official_plagdet only ever consider
        # in-category counterparts (a case linked to a detection that also
        # covers an out-of-category case must not credit that out-of-category
        # overlap here -- but since a detection's overlapping_case_uids may
        # include cases outside this category, and a case's overlapping_det_uids
        # may include detections that also matched other categories' cases,
        # we rebuild filtered copies rather than reuse the shared dicts as-is).
        cat_case_uids = set(c["uid"] for c in cat_cases)
        filtered_cases = []
        for c in cat_cases:
            fc = dict(c)
            fc["overlapping_det_uids"] = [du for du in c["overlapping_det_uids"] if du in set(cat_det_uids)]
            filtered_cases.append(fc)
        filtered_detections = []
        for d in cat_detections:
            fd = dict(d)
            fd["overlapping_case_uids"] = [cu for cu in d["overlapping_case_uids"] if cu in cat_case_uids]
            filtered_detections.append(fd)

        result_cat = compute_official_plagdet(filtered_cases, filtered_detections)
        category_results[cat] = result_cat
        print(f"  {cat:22s} |S|={result_cat['n_cases']:4d} |R|={result_cat['n_detections']:4d} "
              f"P={result_cat['precision']:.4f} R={result_cat['recall']:.4f} "
              f"F1={result_cat['f1']:.4f} gran={result_cat['granularity']:.4f} "
              f"plagdet={result_cat['plagdet']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = [
        {"variant": "all_docs_pooled", **result_all, "n_docs": len(clean_doc_ids),
         "n_docs_excluded": len(mismatched)},
        {"variant": "plag_docs_only_pooled", **result_plag, "n_docs": len(plag_doc_ids),
         "n_docs_excluded": len(mismatched)},
        {"variant": "conditional_precision_firing_docs_only",
         "precision": cond_precision, "recall": None, "f1": None,
         "granularity": None, "plagdet": None,
         "n_cases": None, "n_detections": cond_n_det, "n_cases_detected": None,
         "n_docs": len(firing_doc_ids), "n_docs_excluded": len(mismatched)},
    ]
    for cat, result_cat in category_results.items():
        out_rows.append({"variant": f"category_{cat}", **result_cat,
                          "n_docs": None, "n_docs_excluded": len(mismatched)})
    out_df = pd.DataFrame(out_rows)
    out_path = out_dir / "plagdet_official_summary.parquet"
    out_df.to_parquet(out_path, index=False)
    print()
    print(f"Saved {out_path}")

    excluded_path = out_dir / "plagdet_official_excluded_docs.txt"
    excluded_path.write_text(
        f"Docs excluded from official pooling because the live per_doc/ cache\n"
        f"no longer matches the archived {summary_path.name} totals\n"
        f"(cache likely overwritten by a later pipeline run touching these doc ids):\n\n"
        + "\n".join(mismatched) + "\n"
    )
    print(f"Saved {excluded_path}")

    return result_all, result_plag, mismatched, cond_precision, category_results


def _extract_doc_number(doc_id: str) -> int:
    # e.g. "part1__suspicious-document00081.txt" -> 81
    digits = "".join(ch for ch in doc_id.split("document")[-1] if ch.isdigit())
    return int(digits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["308", "custom"], required=True)
    parser.add_argument("--doc-range", type=str, default=None,
                         help="Restrict to doc numbers in this inclusive range, e.g. '1-80' or '81-308' "
                              "(308 run only). Used to separate the tuning subset (1-80, used during "
                              "development) from the held-out tail (81-308).")
    args = parser.parse_args()

    doc_id_filter = None
    range_label = ""
    if args.doc_range is not None:
        lo, hi = (int(x) for x in args.doc_range.split("-"))
        doc_id_filter = lambda d, lo=lo, hi=hi: lo <= _extract_doc_number(d) <= hi
        range_label = f", docs {lo}-{hi}"

    if args.run == "308":
        out_subdir = "official_plagdet"
        if args.doc_range is not None:
            out_subdir = f"official_plagdet_docs_{args.doc_range.replace('-', '_')}"
        run(
            run_name=f"PAN 2011, 308-doc subset (regenerated after filling in docs 00010/00012){range_label}",
            summary_path=RESULTS_DIR / "plagdet_summary.parquet",
            per_doc_dir=PER_DOC_DIR,
            gt_xml_dir=GT_XML_DIR,
            out_dir=RESULTS_DIR / out_subdir,
            doc_id_filter=doc_id_filter,
        )
    else:
        run(
            run_name="Custom dataset, 10-doc subset (regenerated after filling in docs 00001/00007/00008/00009)",
            summary_path=RESULTS_DIR_CUSTOM / "plagdet_summary.parquet",
            per_doc_dir=RESULTS_DIR_CUSTOM / "per_doc",
            gt_xml_dir=CUSTOM_GT_XML_DIR,
            out_dir=RESULTS_DIR_CUSTOM / "official_plagdet",
            doc_id_filter=doc_id_filter,
        )


if __name__ == "__main__":
    main()
