"""
Compute PAN-style plagdet score + granularity for all evaluated docs.

Formulas (Potthast et al. 2011):
  A detection r "detects" GT case s iff:
    - suspicious offsets overlap: splg ∩ rplg ≠ ∅
    - source offsets overlap:     ssrc ∩ rsrc ≠ ∅
    - same source document:       dsrc == d'src

  precision(S,R)    = |{r∈R : r detects some s∈S}| / |R|      (char-level in our impl)
  recall(S,R)       = |{s∈S : s detected by some r∈R}| / |S|  (char-level in our impl)
  granularity(S,R)  = (1/|S_R|) * Σ_{s∈S_R} |R_s|
    where S_R = GT cases detected by ≥1 detection
          R_s = detections that detect s
    (GT cases with zero detections are excluded — not counted as granularity=0)
  plagdet(S,R)      = F1 / log2(1 + granularity)

We use character-level precision/recall (same as analytics_summary char_f1) for
comparability with the PAN 2011 published results table.

Outputs:
  scripts/final/pipeline_results/plagdet_summary.parquet
  scripts/final/pipeline_results/obfuscation_breakdown.parquet
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import math

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "pipeline_results"
PER_DOC_DIR  = RESULTS_DIR / "per_doc"
GT_XML_DIR   = SCRIPT_DIR.parents[1] / "datasets" / "PAN2011" / "usable" / "suspicious-document" / "part1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def overlap_chars(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def load_gt_spans(doc_id: str) -> list[dict]:
    """Load GT plagiarism spans from PAN 2011 XML for one suspicious doc."""
    xml_name = doc_id.replace("part1__", "").replace(".txt", ".xml")
    xml_path = GT_XML_DIR / xml_name
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    spans = []
    for f in tree.findall('.//feature[@name="plagiarism"]'):
        src_ref = f.get("source_reference", "")
        # Normalise source_reference to the same format as source_doc_id in per_doc
        # GT has e.g. "source-document00178.txt", pipeline uses "part1__source-document00178.txt"
        # We need to find which part the source belongs to — but GT XML doesn't tell us the part.
        # Use bare filename for matching (strip part prefix from pipeline side when comparing).
        spans.append({
            "susp_offset": int(f.get("this_offset", 0)),
            "susp_length": int(f.get("this_length", 0)),
            "src_reference": src_ref,          # bare: "source-document00178.txt"
            "src_offset": int(f.get("source_offset", 0)),
            "src_length": int(f.get("source_length", 0)),
            "obfuscation": f.get("obfuscation", "none"),
            "type": f.get("type", ""),
        })
    return spans


def compute_plagdet_doc(gt_spans: list[dict], det_df: pd.DataFrame) -> dict:
    """
    Compute char-level precision, recall, granularity, plagdet for one doc.
    det_df: merged detected spans (per_doc parquet), may be empty.
    gt_spans: list of GT span dicts.
    """
    if not gt_spans:
        # Clean doc
        if det_df.empty or len(det_df) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "granularity": 1.0, "plagdet": 1.0,
                    "gt_spans": 0, "det_spans": 0,
                    "tp_chars": 0, "gt_chars": 0, "det_chars": 0}
        else:
            det_chars = sum(r["suspicious_end_char"] - r["suspicious_start_char"]
                            for _, r in det_df.iterrows())
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                    "granularity": 1.0, "plagdet": 0.0,
                    "gt_spans": 0, "det_spans": len(det_df),
                    "tp_chars": 0, "gt_chars": 0, "det_chars": det_chars}

    gt_chars = sum(s["susp_length"] for s in gt_spans)

    if det_df.empty or len(det_df) == 0:
        # Plagiarised doc with zero detections: source was missed entirely.
        # Precision=0.0 (not 1.0) so macro-averaging is not inflated by missed sources.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "granularity": 1.0, "plagdet": 0.0,
                "gt_spans": len(gt_spans), "det_spans": 0,
                "tp_chars": 0, "gt_chars": gt_chars, "det_chars": 0}

    detections = det_df.to_dict("records")

    # Strip part prefix from source_doc_id for matching with GT source_reference
    for d in detections:
        parts = d["source_doc_id"].split("__")
        d["src_basename"] = parts[1] if len(parts) > 1 else d["source_doc_id"]

    # For each GT span, find which detections detect it
    R_s = {i: [] for i in range(len(gt_spans))}   # gt_idx -> list of det indices
    det_detects_any = [False] * len(detections)

    for gi, gt in enumerate(gt_spans):
        gt_susp_start = gt["susp_offset"]
        gt_susp_end   = gt["susp_offset"] + gt["susp_length"]
        gt_src_start  = gt["src_offset"]
        gt_src_end    = gt["src_offset"] + gt["src_length"]

        for di, det in enumerate(detections):
            if det["src_basename"] != gt["src_reference"]:
                continue
            if not intervals_overlap(det["suspicious_start_char"], det["suspicious_end_char"],
                                     gt_susp_start, gt_susp_end):
                continue
            if not intervals_overlap(det["source_start_char"], det["source_end_char"],
                                     gt_src_start, gt_src_end):
                continue
            R_s[gi].append(di)
            det_detects_any[di] = True

    # Char-level recall: overlap chars between detected and GT on suspicious side
    tp_chars = 0
    for gi, gt in enumerate(gt_spans):
        gt_susp_start = gt["susp_offset"]
        gt_susp_end   = gt["susp_offset"] + gt["susp_length"]
        # Union of all detecting detections' suspicious spans overlapping this GT span
        detected_intervals = []
        for di in R_s[gi]:
            det = detections[di]
            detected_intervals.append((det["suspicious_start_char"], det["suspicious_end_char"]))
        if not detected_intervals:
            continue
        # Merge intervals and count overlap with GT span
        detected_intervals.sort()
        merged = [detected_intervals[0]]
        for start, end in detected_intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for (s, e) in merged:
            tp_chars += overlap_chars(s, e, gt_susp_start, gt_susp_end)

    det_chars = sum(d["suspicious_end_char"] - d["suspicious_start_char"] for d in detections)

    # Precision convention: if no detections were made, precision is undefined.
    # We use 0.0 (missed source = false negative, not perfect precision) so that
    # macro-averaging over plagiarised docs is not inflated by zero-detection docs.
    char_precision = tp_chars / det_chars if det_chars > 0 else 0.0
    char_recall    = tp_chars / gt_chars  if gt_chars  > 0 else 1.0
    char_f1 = (2 * char_precision * char_recall / (char_precision + char_recall)
               if (char_precision + char_recall) > 0 else 0.0)

    # Granularity: only over GT spans that were detected by at least one detection
    S_R = [gi for gi in range(len(gt_spans)) if len(R_s[gi]) > 0]
    if S_R:
        granularity = sum(len(R_s[gi]) for gi in S_R) / len(S_R)
    else:
        granularity = 1.0  # no detections at all → granularity=1 by convention

    plagdet = char_f1 / math.log2(1 + granularity) if char_f1 > 0 else 0.0

    return {
        "precision": round(char_precision, 4),
        "recall":    round(char_recall, 4),
        "f1":        round(char_f1, 4),
        "granularity": round(granularity, 4),
        "plagdet":   round(plagdet, 4),
        "gt_spans":  len(gt_spans),
        "det_spans": len(detections),
        "tp_chars":  tp_chars,
        "gt_chars":  gt_chars,
        "det_chars": det_chars,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytics", type=Path,
                        default=RESULTS_DIR / "analytics_summary.parquet",
                        help="Path to analytics_summary.parquet to compute plagdet for")
    parser.add_argument("--per-doc-dir", type=Path, default=PER_DOC_DIR,
                        help="Directory containing per_doc/*.parquet span files")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                        help="Output directory for plagdet_summary and obfuscation_breakdown")
    parser.add_argument("--extended", action="store_true",
                        help="Use _extended.parquet files (char n-gram aligner output) where available")
    args = parser.parse_args()

    analytics = pd.read_parquet(args.analytics)
    per_doc_dir = args.per_doc_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = analytics["suspicious_doc_id"].tolist()

    label = "extended (char n-gram aligner)" if args.extended else "base"
    print(f"Computing plagdet for {len(doc_ids)} documents [{label}]...")

    plagdet_rows = []
    obf_rows = []

    for doc_id in doc_ids:
        gt_spans = load_gt_spans(doc_id)

        # Prefer _extended.parquet if --extended flag set and file exists
        extended_path = per_doc_dir / f"{doc_id}_extended.parquet"
        base_path     = per_doc_dir / f"{doc_id}.parquet"
        if args.extended and extended_path.exists():
            per_doc_path = extended_path
        else:
            per_doc_path = base_path

        if per_doc_path.exists():
            det_df = pd.read_parquet(per_doc_path)
        else:
            det_df = pd.DataFrame()

        result = compute_plagdet_doc(gt_spans, det_df)
        result["suspicious_doc_id"] = doc_id
        plagdet_rows.append(result)

        # Per-obfuscation-type breakdown
        for gt in gt_spans:
            obf = gt.get("obfuscation", "none")
            typ = gt.get("type", "")
            # Classify into PAN 2011 Table 3 categories
            if typ == "":
                category = "none"
            elif obf == "none":
                category = "none"
            elif obf == "low" and "artificial" in typ:
                category = "paraphrase-auto-low"
            elif obf == "high" and "artificial" in typ:
                category = "paraphrase-auto-high"
            elif "translation" in typ and "automatic" in typ:
                category = "translation-auto"
            elif "translation" in typ:
                category = "translation-manual"
            else:
                category = f"{obf}"

            # Check if this GT span was detected
            src_basename = gt["src_reference"]
            gt_susp_start = gt["susp_offset"]
            gt_susp_end   = gt["susp_offset"] + gt["susp_length"]
            gt_src_start  = gt["src_offset"]
            gt_src_end    = gt["src_offset"] + gt["src_length"]

            detected = False
            tp_c = 0
            if not det_df.empty and len(det_df) > 0:
                for _, d in det_df.iterrows():
                    parts = d["source_doc_id"].split("__")
                    src_base = parts[1] if len(parts) > 1 else d["source_doc_id"]
                    if src_base != src_basename:
                        continue
                    if (intervals_overlap(d["suspicious_start_char"], d["suspicious_end_char"],
                                         gt_susp_start, gt_susp_end) and
                        intervals_overlap(d["source_start_char"], d["source_end_char"],
                                         gt_src_start, gt_src_end)):
                        detected = True
                        tp_c += overlap_chars(d["suspicious_start_char"], d["suspicious_end_char"],
                                              gt_susp_start, gt_susp_end)

            obf_rows.append({
                "suspicious_doc_id": doc_id,
                "category": category,
                "obfuscation": obf,
                "type": typ,
                "gt_susp_length": gt["susp_length"],
                "detected": detected,
                "tp_chars": tp_c,
            })

    # ---------------------------------------------------------------------------
    # plagdet_summary.parquet
    # ---------------------------------------------------------------------------
    plagdet_df = pd.DataFrame(plagdet_rows)
    col_order = ["suspicious_doc_id", "precision", "recall", "f1", "granularity",
                 "plagdet", "gt_spans", "det_spans", "tp_chars", "gt_chars", "det_chars"]
    plagdet_df = plagdet_df[col_order]
    plagdet_df.to_parquet(out_dir / "plagdet_summary.parquet", index=False)
    print(f"Saved plagdet_summary.parquet ({len(plagdet_df)} rows)")

    # Aggregate — macro over plagiarised docs only (clean docs trivially score 1.0
    # and would inflate the macro average)
    plag_df  = plagdet_df[plagdet_df["gt_spans"] > 0]

    macro_plagdet = plag_df["plagdet"].mean()   if len(plag_df) > 0 else 0.0
    macro_f1      = plag_df["f1"].mean()        if len(plag_df) > 0 else 0.0
    macro_gran    = plag_df["granularity"].mean() if len(plag_df) > 0 else 1.0
    macro_prec    = plag_df["precision"].mean() if len(plag_df) > 0 else 0.0
    macro_rec     = plag_df["recall"].mean()    if len(plag_df) > 0 else 0.0

    # Clean FP penalty: clean docs with detections score plagdet=0 (precision=0)
    # Include them in a full-corpus macro for completeness
    macro_plagdet_all = plagdet_df["plagdet"].mean()
    macro_f1_all      = plagdet_df["f1"].mean()

    # Micro plagdet from corpus totals (plagiarised docs only)
    total_tp  = plag_df["tp_chars"].sum()
    total_gt  = plag_df["gt_chars"].sum()
    total_det = plagdet_df["det_chars"].sum()  # includes FP chars from clean docs
    micro_p  = total_tp / total_det if total_det > 0 else 0.0
    micro_r  = total_tp / total_gt  if total_gt  > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_gran = macro_gran
    micro_plagdet = micro_f1 / math.log2(1 + micro_gran) if micro_f1 > 0 else 0.0

    print()
    print("=" * 50)
    print("AGGREGATE PLAGDET RESULTS")
    print("=" * 50)
    zero_det_plag = (plag_df["det_spans"] == 0).sum()
    print(f"  [plag docs only — {len(plag_df)} docs, {zero_det_plag} with zero detections]")
    print(f"  Macro plagdet  : {macro_plagdet:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Macro precision: {macro_prec:.4f}  (0.0 for zero-detection docs)")
    print(f"  Macro recall   : {macro_rec:.4f}")
    print(f"  Macro gran.    : {macro_gran:.4f}")
    print(f"  Micro plagdet  : {micro_plagdet:.4f}")
    print(f"  Micro F1       : {micro_f1:.4f}")
    print(f"  Micro precision: {micro_p:.4f}")
    print(f"  Micro recall   : {micro_r:.4f}")
    print()
    print(f"  [all docs — {len(plagdet_df)} docs, incl. clean]")
    print(f"  Macro plagdet  : {macro_plagdet_all:.4f}")
    print(f"  Macro F1       : {macro_f1_all:.4f}")

    # ---------------------------------------------------------------------------
    # obfuscation_breakdown.parquet (Task 2)
    # ---------------------------------------------------------------------------
    if obf_rows:
        obf_df = pd.DataFrame(obf_rows)
        breakdown_rows = []
        for cat, grp in obf_df.groupby("category"):
            n_gt   = len(grp)
            n_det  = grp["detected"].sum()
            tp_c   = grp["tp_chars"].sum()
            gt_c   = grp["gt_susp_length"].sum()
            recall = tp_c / gt_c if gt_c > 0 else 0.0

            # Precision: tp_chars for this category / det_chars only for docs
            # that have GT spans in this category (avoids double-counting docs
            # that appear in multiple categories)
            doc_ids_cat = grp["suspicious_doc_id"].unique()
            det_c = plagdet_df[plagdet_df["suspicious_doc_id"].isin(doc_ids_cat)]["det_chars"].sum()
            # Scale det_chars proportionally by how much of each doc's GT belongs to this category
            # Simple approximation: use tp_c / recall as the effective det denominator
            precision = tp_c / det_c if det_c > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            # Granularity: mean detections per detected GT span for this category
            # Count how many detections overlap each detected GT span in this category
            detected_gt_rows = grp[grp["detected"]]
            if len(detected_gt_rows) > 0:
                gran = plag_df[plag_df["suspicious_doc_id"].isin(doc_ids_cat)]["granularity"].mean()
                gran = gran if not np.isnan(gran) else 1.0
            else:
                gran = 1.0
            plagdet_cat = f1 / math.log2(1 + gran) if f1 > 0 else 0.0
            breakdown_rows.append({
                "category": cat,
                "gt_spans": n_gt,
                "detected_spans": int(n_det),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "granularity": round(gran, 4),
                "plagdet": round(plagdet_cat, 4),
            })
        breakdown_df = pd.DataFrame(breakdown_rows).sort_values("category")
        breakdown_df.to_parquet(out_dir / "obfuscation_breakdown.parquet", index=False)
        print()
        print("=" * 50)
        print("PER-OBFUSCATION BREAKDOWN")
        print("=" * 50)
        print(breakdown_df.to_string(index=False))
        print()
        print(f"Saved obfuscation_breakdown.parquet")


if __name__ == "__main__":
    main()