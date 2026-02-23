from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


@dataclass(frozen=True)
class FrameRow:
    frame_index: int
    time_sec: float
    pred_label: str
    manual_label: str
    probs: Optional[List[float]]


@dataclass(frozen=True)
class Selected:
    frame_index: int
    time_sec: float
    label: str
    confidence: float
    ce_weight: float
    neg_label: Optional[str]


def _read_per_frame_csv(path: Path) -> List[FrameRow]:
    rows: List[FrameRow] = []
    prob_cols = [f"prob_{name}" for name in CANONICAL_7]
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

            probs: Optional[List[float]]
            if all((d.get(c) not in (None, "") for c in prob_cols)):
                try:
                    probs = [float(d.get(c) or 0.0) for c in prob_cols]
                except Exception:
                    probs = None
            else:
                probs = None

            rows.append(
                FrameRow(
                    frame_index=fi,
                    time_sec=ts,
                    pred_label=(d.get("pred_label") or "").strip(),
                    manual_label=(d.get("manual_label") or "").strip(),
                    probs=probs,
                )
            )
    return rows


def _argmax_prob(probs: List[float]) -> Tuple[int, float]:
    best_i = max(range(len(probs)), key=lambda j: probs[j])
    return int(best_i), float(probs[best_i])


def _detect_face_haar(frame_bgr, *, min_size: int = 60) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    return int(x), int(y), int(w), int(h)


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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_frames(
    rows: Sequence[FrameRow],
    *,
    label_source: str,
    tau_high: float,
    tau_mid: float,
    min_frame_gap: int,
    max_per_class: int,
    seed: int,
    require_not_unstable: bool,
    require_probs: bool,
    stable_rule: str,
) -> List[Selected]:
    by_label: Dict[str, List[Selected]] = {name: [] for name in CANONICAL_7}
    rng = random.Random(int(seed))

    for r in rows:
        if require_not_unstable and r.pred_label == "(unstable)":
            continue
        if require_probs and r.probs is None:
            continue

        if r.probs is None:
            continue
        best_i, pmax = _argmax_prob(r.probs)
        raw_label = CANONICAL_7[best_i]
        smoothed_label = r.pred_label

        if stable_rule == "raw_eq_smoothed":
            if not smoothed_label or smoothed_label == "(unstable)":
                continue
            if smoothed_label != raw_label:
                continue
        elif stable_rule != "none":
            raise ValueError(f"Unknown stable_rule: {stable_rule}")

        if label_source == "raw":
            lab = raw_label
        elif label_source == "smoothed":
            lab = smoothed_label
        else:
            raise ValueError(f"Unknown label_source: {label_source}")

        if lab not in by_label:
            continue

        if not (0.0 <= pmax <= 1.0):
            # Not expected, but keep selection sane.
            continue

        if pmax >= tau_high:
            ce_w = 1.0
            neg_lab = None
        elif pmax >= tau_mid:
            # Medium confidence: do NOT trust pseudo-label as a positive target;
            # instead apply NegL using neg_label==predicted label.
            ce_w = 0.0
            neg_lab = lab
        else:
            continue

        by_label[lab].append(
            Selected(
                frame_index=int(r.frame_index),
                time_sec=float(r.time_sec),
                label=lab,
                confidence=float(pmax),
                ce_weight=float(ce_w),
                neg_label=neg_lab,
            )
        )

    # Enforce min frame gap then subsample.
    selected: List[Selected] = []
    for lab, items in by_label.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: x.frame_index)
        filtered: List[Selected] = []
        last = -10**12
        for it in items:
            if int(it.frame_index) - int(last) >= int(min_frame_gap):
                filtered.append(it)
                last = it.frame_index

        if max_per_class > 0 and len(filtered) > int(max_per_class):
            filtered = filtered[:]
            rng.shuffle(filtered)
            filtered = sorted(filtered[: int(max_per_class)], key=lambda x: x.frame_index)

        selected.extend(filtered)

    selected = sorted(selected, key=lambda x: x.frame_index)
    return selected


def _write_manifest(
    *,
    manifest_path: Path,
    image_rel_paths: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    ce_weights: Sequence[float],
    neg_labels: Sequence[Optional[str]],
    source: str,
    split: str,
) -> None:
    _ensure_dir(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(
            fp,
            fieldnames=[
                "image_path",
                "label",
                "split",
                "source",
                "confidence",
                "weight",
                "neg_label",
            ],
        )
        w.writeheader()
        for p, lab, conf, wgt, neg in zip(image_rel_paths, labels, confidences, ce_weights, neg_labels):
            w.writerow(
                {
                    "image_path": p.replace("\\", "/"),
                    "label": lab,
                    "split": split,
                    "source": source,
                    "confidence": f"{float(conf):.6f}",
                    "weight": f"{float(wgt):.6f}",
                    "neg_label": "" if not neg else str(neg),
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a self-learning buffer from a realtime run: per_frame.csv + session_*.mp4 -> images + manifest.csv. "
            "High-confidence frames become pseudo-labeled positives; medium-confidence frames become NegL-only samples."
        )
    )
    ap.add_argument("--per-frame", type=Path, required=True, help="Path to per_frame.csv")
    ap.add_argument("--video", type=Path, required=True, help="Path to session_annotated.mp4 (or raw)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: sibling folder 'buffer_selflearn' next to per_frame.csv.",
    )

    ap.add_argument(
        "--label-source",
        type=str,
        default="raw",
        choices=["raw", "smoothed"],
        help="Which label to use for pseudo label / neg_label: raw argmax(prob_*) or smoothed pred_label.",
    )
    ap.add_argument("--tau-high", type=float, default=0.90, help="High-confidence threshold for pseudo-label positives.")
    ap.add_argument("--tau-mid", type=float, default=0.50, help="Lower bound for medium-confidence NegL-only band.")

    ap.add_argument(
        "--stable-rule",
        type=str,
        default="raw_eq_smoothed",
        choices=["none", "raw_eq_smoothed"],
        help="Optional stability filter. raw_eq_smoothed keeps only frames where raw==smoothed and smoothed != '(unstable)'.",
    )

    ap.add_argument("--split", type=str, default="train", help="Split to write into manifest.csv (train|val|test).")
    ap.add_argument("--source", type=str, default="webcam_selflearn", help="Source string written into manifest.csv")

    ap.add_argument("--min-frame-gap", type=int, default=10, help="Minimum gap in frames between saved samples for the same class.")
    ap.add_argument("--max-per-class", type=int, default=300, help="Maximum saved samples per class (after gap filtering). 0 means no cap.")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--require-not-unstable",
        action="store_true",
        help="If set: drop frames where pred_label == '(unstable)'.",
    )
    ap.add_argument(
        "--require-probs",
        action="store_true",
        help="If set: require prob_* columns to exist for the row (recommended).",
    )

    ap.add_argument(
        "--face-crop",
        action="store_true",
        help="If set: attempt Haar face detection and save face crops (fallback to full frame if none found).",
    )
    ap.add_argument("--crop-margin", type=float, default=0.15)

    args = ap.parse_args()

    per_frame: Path = Path(args.per_frame)
    video: Path = Path(args.video)
    if not per_frame.is_absolute():
        per_frame = (REPO_ROOT / per_frame).resolve()
    if not video.is_absolute():
        video = (REPO_ROOT / video).resolve()

    if not per_frame.exists():
        raise SystemExit(f"per_frame.csv not found: {per_frame}")
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    if args.out_dir is None:
        out_dir = per_frame.parent / "buffer_selflearn"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()

    tau_high = float(args.tau_high)
    tau_mid = float(args.tau_mid)
    if not (0.0 <= tau_mid <= tau_high <= 1.0):
        raise SystemExit("Require 0 <= tau_mid <= tau_high <= 1")

    rows = _read_per_frame_csv(per_frame)
    if not rows:
        raise SystemExit(f"No rows read from: {per_frame}")

    selected = _select_frames(
        rows,
        label_source=str(args.label_source),
        tau_high=tau_high,
        tau_mid=tau_mid,
        min_frame_gap=int(args.min_frame_gap),
        max_per_class=int(args.max_per_class),
        seed=int(args.seed),
        require_not_unstable=bool(args.require_not_unstable),
        require_probs=bool(args.require_probs),
        stable_rule=str(args.stable_rule),
    )

    if not selected:
        raise SystemExit(
            "No frames selected. Try lowering --tau-high/--tau-mid, or relax --stable-rule, "
            "or verify per_frame.csv has prob_* columns."
        )

    _ensure_dir(out_dir)
    img_dir = out_dir / "images"
    _ensure_dir(img_dir)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video}")

    rel_paths: List[str] = []
    labels: List[str] = []
    confs: List[float] = []
    ce_w: List[float] = []
    negs: List[Optional[str]] = []

    saved = 0
    for it in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(it.frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        if bool(args.face_crop):
            box = _detect_face_haar(frame)
            if box is not None:
                frame = _crop_with_margin(frame, box, margin=float(args.crop_margin))

        name = f"frame_{int(it.frame_index):07d}_{it.label}.jpg"
        out_path = img_dir / name
        ok2 = cv2.imwrite(str(out_path), frame)
        if not ok2:
            continue

        rel_paths.append(str(out_path.relative_to(out_dir)))
        labels.append(it.label)
        confs.append(float(it.confidence))
        ce_w.append(float(it.ce_weight))
        negs.append(it.neg_label)
        saved += 1

    cap.release()

    if saved == 0:
        raise SystemExit("No frames were saved from video (read/write failure).")

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(
        manifest_path=manifest_path,
        image_rel_paths=rel_paths,
        labels=labels,
        confidences=confs,
        ce_weights=ce_w,
        neg_labels=negs,
        source=str(args.source),
        split=str(args.split).lower(),
    )

    # Summary
    by_label: Dict[str, int] = {k: 0 for k in CANONICAL_7}
    pos = 0
    neg = 0
    for lab, wgt, nlab in zip(labels, ce_w, negs):
        by_label[lab] += 1
        if wgt > 0:
            pos += 1
        if nlab:
            neg += 1

    summary = {
        "per_frame": str(per_frame),
        "video": str(video),
        "out_dir": str(out_dir),
        "selected_frames": int(len(selected)),
        "saved_frames": int(saved),
        "tau_high": float(tau_high),
        "tau_mid": float(tau_mid),
        "label_source": str(args.label_source),
        "stable_rule": str(args.stable_rule),
        "positive_samples": int(pos),
        "negl_only_samples": int(neg),
        "per_label_counts": dict(by_label),
    }
    (out_dir / "buffer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {out_dir / 'buffer_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
