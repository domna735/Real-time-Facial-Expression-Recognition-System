from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FrameRow:
    frame_index: int
    time_sec: float
    manual_label: str
    pred_label: str


def _read_per_frame_csv(path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                fi = int(float(d.get("frame_index") or 0))
            except Exception:
                continue
            try:
                ts = float(d.get("time_sec") or 0.0)
            except Exception:
                ts = 0.0
            rows.append(
                FrameRow(
                    frame_index=fi,
                    time_sec=ts,
                    manual_label=(d.get("manual_label") or "").strip(),
                    pred_label=(d.get("pred_label") or "").strip(),
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
) -> Dict[str, float]:
    """Protocol-lite scoring:

    - Only score frames where manual_label exists.
    - Ignore frames within +/- exclusion_ms of a manual label change.
    - Ignore segments shorter than min_hold_ms.
    """

    if not frames:
        return {"scored_frames": 0.0, "accuracy": 0.0}

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
    return {"scored_frames": float(scored), "accuracy": float(acc)}


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
    args = ap.parse_args()

    frames = _read_per_frame_csv(args.per_frame)
    if not frames:
        raise SystemExit(f"No rows read from: {args.per_frame}")

    # Basic counts
    manual_frames = sum(1 for fr in frames if fr.manual_label)
    unstable_frames = sum(1 for fr in frames if fr.pred_label == "(unstable)")

    scored = _score_transition_fair(frames, min_hold_ms=float(args.min_hold_ms), exclusion_ms=float(args.exclusion_ms))
    jitter = _jitter_rate(frames)

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
        "jitter_flips_per_min": float(jitter),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
