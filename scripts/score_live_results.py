from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


@dataclass(frozen=True)
class FrameRow:
    frame_index: int
    time_sec: float
    manual_label: str
    pred_label: str
    prob: Optional[List[float]]


def _read_per_frame_csv(path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        prob_cols = [f"prob_{name}" for name in CANONICAL_7]
        for d in r:
            try:
                fi = int(float(d.get("frame_index") or 0))
            except Exception:
                continue
            try:
                ts = float(d.get("time_sec") or 0.0)
            except Exception:
                ts = 0.0

            prob: Optional[List[float]]
            if all((d.get(c) not in (None, "") for c in prob_cols)):
                try:
                    prob = [float(d.get(c) or 0.0) for c in prob_cols]
                except Exception:
                    prob = None
            else:
                prob = None
            rows.append(
                FrameRow(
                    frame_index=fi,
                    time_sec=ts,
                    manual_label=(d.get("manual_label") or "").strip(),
                    pred_label=(d.get("pred_label") or "").strip(),
                    prob=prob,
                )
            )
    return rows


def _is_valid_label(s: str) -> bool:
    return bool(s) and s != "(unstable)"


def _score_transition_fair(
    frames: List[FrameRow],
    *,
    min_hold_ms: float,
    exclusion_ms: float,
) -> Tuple[List[bool], Dict[str, float]]:
    """Protocol-lite scoring:

    - Only score frames where manual_label exists.
    - Ignore frames within +/- exclusion_ms of a manual label change.
    - Ignore segments shorter than min_hold_ms.
    """

    if not frames:
        return [], {"scored_frames": 0.0, "accuracy": 0.0}

    # Identify manual segments.
    segments: List[Tuple[int, int, str]] = []
    start = 0
    cur = frames[0].manual_label
    for i in range(1, len(frames)):
        if frames[i].manual_label != cur:
            segments.append((start, i - 1, cur))
            start = i
            cur = frames[i].manual_label
    segments.append((start, len(frames) - 1, cur))

    # Build a mask of frames to score.
    score_mask = [False] * len(frames)

    for seg_start, seg_end, label in segments:
        if not label:
            continue

        # Segment duration
        t0 = frames[seg_start].time_sec
        t1 = frames[seg_end].time_sec
        dur_ms = max(0.0, (t1 - t0) * 1000.0)
        if dur_ms < min_hold_ms:
            continue

        # Exclude transition zones
        excl = exclusion_ms / 1000.0
        left_t = t0 + excl
        right_t = t1 - excl
        for i in range(seg_start, seg_end + 1):
            t = frames[i].time_sec
            if left_t <= t <= right_t:
                score_mask[i] = True

    scored = 0
    correct = 0

    for i, fr in enumerate(frames):
        if not score_mask[i]:
            continue
        if not _is_valid_label(fr.pred_label):
            # considered incorrect
            scored += 1
            continue
        scored += 1
        if fr.pred_label == fr.manual_label:
            correct += 1

    acc = (correct / scored) if scored else 0.0
    return score_mask, {"scored_frames": float(scored), "accuracy": float(acc)}


def _confusion_and_f1(
    *,
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[str, object]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    per_class_f1: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}
    f1_vals: List[float] = []
    f1_present: List[float] = []
    correct = 0
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[t][i] for t in range(num_classes) if t != i)
        fn = sum(cm[i][p] for p in range(num_classes) if p != i)
        support = sum(cm[i][p] for p in range(num_classes))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0
        per_class_f1[CANONICAL_7[i]] = float(f1)
        per_class_support[CANONICAL_7[i]] = int(support)
        f1_vals.append(float(f1))
        if support > 0:
            f1_present.append(float(f1))
        correct += tp

    total = len(y_true)
    acc = (correct / total) if total else 0.0
    macro_f1 = (sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0

    macro_f1_present = (sum(f1_present) / len(f1_present)) if f1_present else 0.0

    lowest3 = sorted(f1_vals)[:3]
    minority_f1_lowest3 = (sum(lowest3) / len(lowest3)) if lowest3 else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_f1_present": float(macro_f1_present),
        "minority_f1_lowest3": float(minority_f1_lowest3),
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "confusion_matrix": cm,
    }


def _jitter_rate(frames: List[FrameRow]) -> float:
    """Pred label flips per minute (ignoring '(unstable)')."""
    preds = [fr.pred_label for fr in frames if _is_valid_label(fr.pred_label)]
    if len(preds) < 2:
        return 0.0
    flips = 0
    prev = preds[0]
    for p in preds[1:]:
        if p != prev:
            flips += 1
        prev = p
    duration_sec = max(1e-6, frames[-1].time_sec - frames[0].time_sec)
    return float(flips / (duration_sec / 60.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score demo per-frame logs against manual labels (protocol-lite).")
    ap.add_argument("--per-frame", type=Path, required=True, help="per_frame.csv from demo/realtime_demo.py")
    ap.add_argument("--out", type=Path, required=True, help="Where to write score_results.json")
    ap.add_argument("--min-hold-ms", type=float, default=600.0)
    ap.add_argument("--exclusion-ms", type=float, default=250.0)
    ap.add_argument(
        "--pred-source",
        type=str,
        default="both",
        choices=["smoothed", "raw", "both"],
        help=(
            "Which prediction to score: 'smoothed' uses pred_label (EMA+hysteresis+vote), "
            "'raw' uses argmax(prob_*), 'both' computes both."
        ),
    )
    args = ap.parse_args()

    frames = _read_per_frame_csv(args.per_frame)
    if not frames:
        raise SystemExit(f"No rows read from: {args.per_frame}")

    # Basic counts
    manual_frames = sum(1 for fr in frames if fr.manual_label)
    unstable_frames = sum(1 for fr in frames if fr.pred_label == "(unstable)")

    score_mask, scored = _score_transition_fair(frames, min_hold_ms=float(args.min_hold_ms), exclusion_ms=float(args.exclusion_ms))
    jitter = _jitter_rate(frames)

    # Build scored label lists for macro-F1 etc.
    label_to_idx = {name: i for i, name in enumerate(CANONICAL_7)}
    y_true: List[int] = []
    y_pred_smoothed: List[int] = []
    y_pred_raw: List[int] = []

    missing_prob = 0
    for i, fr in enumerate(frames):
        if not score_mask[i]:
            continue
        if not _is_valid_label(fr.manual_label):
            continue
        if fr.manual_label not in label_to_idx:
            continue
        t = int(label_to_idx[fr.manual_label])
        y_true.append(t)

        # Smoothed prediction from pred_label
        if fr.pred_label in label_to_idx:
            y_pred_smoothed.append(int(label_to_idx[fr.pred_label]))
        else:
            # Treat unstable/unknown as a wrong prediction by mapping to -1 and skipping from F1.
            # (Accuracy in 'scored' already counts it as incorrect.)
            y_pred_smoothed.append(-1)

        # Raw prediction from argmax(prob_*)
        if fr.prob is None:
            missing_prob += 1
            y_pred_raw.append(-1)
        else:
            best = max(range(len(fr.prob)), key=lambda j: fr.prob[j])
            y_pred_raw.append(int(best))

    # Filter out invalid preds for F1 calculations
    def _filter_valid(y_p: List[int]) -> Tuple[List[int], List[int]]:
        yt2: List[int] = []
        yp2: List[int] = []
        for t, p in zip(y_true, y_p):
            if 0 <= p < len(CANONICAL_7):
                yt2.append(t)
                yp2.append(p)
        return yt2, yp2

    f1_metrics: Dict[str, object] = {
        "scored_frames_for_f1": int(len(y_true)),
        "missing_prob_rows": int(missing_prob),
    }
    if args.pred_source in ("smoothed", "both"):
        yt_s, yp_s = _filter_valid(y_pred_smoothed)
        f1_metrics["smoothed"] = _confusion_and_f1(y_true=yt_s, y_pred=yp_s, num_classes=len(CANONICAL_7))
    if args.pred_source in ("raw", "both"):
        yt_r, yp_r = _filter_valid(y_pred_raw)
        f1_metrics["raw"] = _confusion_and_f1(y_true=yt_r, y_pred=yp_r, num_classes=len(CANONICAL_7))

    payload = {
        "per_frame": str(args.per_frame),
        "frames_total": int(len(frames)),
        "frames_with_manual": int(manual_frames),
        "unstable_frames": int(unstable_frames),
        "protocol": {
            "min_hold_ms": float(args.min_hold_ms),
            "exclusion_ms": float(args.exclusion_ms),
        },
        "scored": scored,
        "metrics": f1_metrics,
        "jitter_flips_per_min": float(jitter),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
