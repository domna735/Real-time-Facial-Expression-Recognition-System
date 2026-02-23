from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


@dataclass(frozen=True)
class TemplateRow:
    frame_index: int
    time_sec: float
    manual_label: str


def _read_template_rows(path: Path) -> List[TemplateRow]:
    rows: List[TemplateRow] = []
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
                TemplateRow(
                    frame_index=fi,
                    time_sec=ts,
                    manual_label=(d.get("manual_label") or "").strip(),
                )
            )
    return rows


def _apply_hysteresis(probs: List[float], current_idx: Optional[int], delta: float) -> Optional[int]:
    if not probs:
        return current_idx

    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    if current_idx is None:
        return top_idx

    if top_idx == current_idx:
        return current_idx

    if probs[top_idx] >= probs[current_idx] + float(delta):
        return top_idx
    return current_idx


def _vote_smooth(labels: Deque[int], *, window: int, min_count: int) -> Optional[int]:
    if window <= 1:
        return labels[-1] if labels else None
    if not labels:
        return None
    # Avoid importing Counter in hot loop.
    counts: Dict[int, int] = {}
    for lab in labels:
        counts[int(lab)] = counts.get(int(lab), 0) + 1
    lab, cnt = max(counts.items(), key=lambda kv: kv[1])
    return int(lab) if cnt >= int(min_count) else None


def _largest_face(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def _crop_with_margin(frame_bgr, box: Tuple[int, int, int, int], margin: float = 0.15):
    x, y, w, h = box
    H, W = frame_bgr.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    return frame_bgr[y1:y2, x1:x2]


def _load_thresholds(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Re-run inference on a recorded demo session video while preserving manual labels/time from an existing per_frame.csv. "
            "This enables fair A/B scoring with scripts/score_live_results.py."
        )
    )
    ap.add_argument("--video", type=Path, required=True, help="Path to session_annotated.mp4 (or raw)")
    ap.add_argument(
        "--template-per-frame",
        type=Path,
        required=True,
        help="Existing per_frame.csv containing manual_label + time_sec to reuse.",
    )
    ap.add_argument("--model-ckpt", type=Path, required=True, help="Student checkpoint (.pt) to evaluate")
    ap.add_argument(
        "--detector",
        type=str,
        default="yunet",
        choices=["yunet", "dnn", "haar"],
        help="Face detector to use (should match the original run).",
    )
    ap.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Optional thresholds.json from the original run (EMA/vote/hysteresis params).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "dml"],
        help="Device preference for model inference.",
    )
    ap.add_argument("--out-per-frame", type=Path, required=True, help="Where to write the new per_frame.csv")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    if not args.template_per_frame.exists():
        raise SystemExit(f"Template per_frame.csv not found: {args.template_per_frame}")
    if not args.model_ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.model_ckpt}")

    template_rows = _read_template_rows(args.template_per_frame)
    if not template_rows:
        raise SystemExit(f"No rows read from template: {args.template_per_frame}")

    # Reuse demo code for model loading & detector implementations.
    from demo.realtime_demo import FaceDetector, _load_student_from_checkpoint  # type: ignore

    infer, meta = _load_student_from_checkpoint(args.model_ckpt, prefer_device=str(args.device))

    model_dir = REPO_ROOT / "demo" / "models"
    detector = FaceDetector(str(args.detector), model_dir=model_dir)

    thr = _load_thresholds(args.thresholds_json)
    ema_alpha = float(thr.get("ema_alpha") or 0.70)
    hysteresis_delta = float(thr.get("hysteresis_delta") or 0.08)
    vote_window = int(thr.get("vote_window") or 15)
    vote_min_count = int(thr.get("vote_min_count") or 8)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    ema_probs: Optional[List[float]] = None
    hyster_idx: Optional[int] = None
    votes: Deque[int] = deque(maxlen=vote_window)

    args.out_per_frame.parent.mkdir(parents=True, exist_ok=True)
    with args.out_per_frame.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(
            f_out,
            fieldnames=[
                "frame_index",
                "time_sec",
                "manual_label",
                "pred_label",
                *[f"prob_{name}" for name in CANONICAL_7],
                "detector",
                "model",
                "ckpt",
            ],
        )
        w.writeheader()

        for i, tr in enumerate(template_rows):
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] Video ended early at template row {i}/{len(template_rows)}")
                break

            faces = detector.detect(frame)
            face_box = _largest_face(faces)

            probs = [0.0] * len(CANONICAL_7)
            pred_idx: Optional[int] = None

            if face_box is not None:
                crop = _crop_with_margin(frame, face_box)
                _logits, probs = infer(crop)

                if ema_probs is None:
                    ema_probs = list(probs)
                else:
                    a = float(ema_alpha)
                    ema_probs = [a * e + (1.0 - a) * p for e, p in zip(ema_probs, probs)]

                hyster_idx = _apply_hysteresis(ema_probs, hyster_idx, hysteresis_delta)
                votes = deque(votes, maxlen=vote_window)
                if hyster_idx is not None:
                    votes.append(int(hyster_idx))
                pred_idx = _vote_smooth(votes, window=vote_window, min_count=vote_min_count)

            pred_label = CANONICAL_7[pred_idx] if pred_idx is not None else "(unstable)"

            row = {
                "frame_index": int(tr.frame_index),
                "time_sec": float(tr.time_sec),
                "manual_label": tr.manual_label,
                "pred_label": pred_label,
                **{f"prob_{name}": f"{float(p):.6f}" for name, p in zip(CANONICAL_7, probs)},
                "detector": str(args.detector),
                "model": str(meta.get("model") or ""),
                "ckpt": str(args.model_ckpt),
            }
            w.writerow(row)

    print(f"Wrote: {args.out_per_frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
