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

# Canonical labels (keep consistent with training)
from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


@dataclass(frozen=True)
class SelectedFrame:
    frame_index: int
    label: str
    time_sec: float
    confidence: Optional[float]


def _read_per_frame_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _max_prob(row: dict) -> Optional[float]:
    probs: List[float] = []
    for name in CANONICAL_7:
        v = (row.get(f"prob_{name}") or "").strip()
        if not v:
            continue
        try:
            probs.append(float(v))
        except Exception:
            continue
    if not probs:
        return None
    return float(max(probs))


def _prob_of_label(row: dict, label: str) -> Optional[float]:
    v = (row.get(f"prob_{label}") or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _parse_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _parse_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_face_haar(frame_bgr, *, min_size: int = 60) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    # pick largest
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


def _select_frames_manual(
    rows: Sequence[dict],
    *,
    min_frame_gap: int,
    max_per_class: int,
    seed: int,
    require_not_unstable: bool,
) -> List[SelectedFrame]:
    # Collect candidate frames per class.
    by_label: Dict[str, List[SelectedFrame]] = {name: [] for name in CANONICAL_7}

    for r in rows:
        manual = (r.get("manual_label") or "").strip()
        if not manual or manual not in by_label:
            continue
        if require_not_unstable:
            pred = (r.get("pred_label") or "").strip()
            if pred == "(unstable)":
                continue

        fi = _parse_int(r.get("frame_index"), default=-1)
        if fi < 0:
            continue
        tsec = _parse_float(r.get("time_sec"), default=math.nan)

        conf = _prob_of_label(r, manual)
        if conf is None:
            conf = _max_prob(r)

        by_label[manual].append(
            SelectedFrame(
                frame_index=fi,
                label=manual,
                time_sec=tsec,
                confidence=conf,
            )
        )

    # Enforce min frame gap (per class) then subsample to max_per_class.
    out: List[SelectedFrame] = []
    rng = random.Random(int(seed))

    for label, items in by_label.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: x.frame_index)
        filtered: List[SelectedFrame] = []
        last = -10**12
        for it in items:
            if it.frame_index - last >= int(min_frame_gap):
                filtered.append(it)
                last = it.frame_index

        if max_per_class > 0 and len(filtered) > max_per_class:
            # Sample deterministically.
            filtered = filtered[:]
            rng.shuffle(filtered)
            filtered = sorted(filtered[: int(max_per_class)], key=lambda x: x.frame_index)

        out.extend(filtered)

    out = sorted(out, key=lambda x: x.frame_index)
    return out


def _write_manifest(
    *,
    manifest_path: Path,
    image_rel_paths: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[Optional[float]],
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
            ],
        )
        w.writeheader()
        for p, lab, conf in zip(image_rel_paths, labels, confidences):
            w.writerow(
                {
                    "image_path": p.replace("\\", "/"),
                    "label": lab,
                    "split": split,
                    "source": source,
                    "confidence": "" if conf is None else f"{float(conf):.6f}",
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a small training buffer from a labeled realtime run: per_frame.csv + session_*.mp4 -> images + manifest.csv"
        )
    )
    ap.add_argument("--per-frame", type=Path, required=True, help="Path to per_frame.csv")
    ap.add_argument("--video", type=Path, required=True, help="Path to session_annotated.mp4 (or raw)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: sibling folder 'buffer_manual' next to per_frame.csv.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["manual"],
        default="manual",
        help="Frame selection mode. Currently only 'manual' is supported.",
    )
    ap.add_argument("--split", type=str, default="train", help="Split to write into manifest.csv (train|val|test).")
    ap.add_argument("--source", type=str, default="webcam_manual", help="Source string written into manifest.csv")
    ap.add_argument(
        "--min-frame-gap",
        type=int,
        default=10,
        help="Minimum gap in frames between saved samples for the same class.",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="Maximum saved samples per class (after gap filtering). 0 means no cap.",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed used when max-per-class truncates.")
    ap.add_argument(
        "--require-not-unstable",
        action="store_true",
        help="If set: only keep rows where pred_label != '(unstable)'.",
    )
    ap.add_argument(
        "--face-crop",
        action="store_true",
        help="If set: attempt Haar face detection and save face crops (fallback to full frame if none found).",
    )
    ap.add_argument(
        "--crop-margin",
        type=float,
        default=0.15,
        help="Margin used for face crops when --face-crop is enabled.",
    )

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
        out_dir = per_frame.parent / "buffer_manual"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()

    images_dir = out_dir / "images"
    _ensure_dir(images_dir)

    rows = _read_per_frame_csv(per_frame)

    selected = _select_frames_manual(
        rows,
        min_frame_gap=int(args.min_frame_gap),
        max_per_class=int(args.max_per_class),
        seed=int(args.seed),
        require_not_unstable=bool(args.require_not_unstable),
    )

    if not selected:
        raise SystemExit(
            "No frames selected. Check that per_frame.csv contains manual_label entries and that selection flags are not too strict."
        )

    # Prepare fast lookup by frame index.
    wanted: Dict[int, SelectedFrame] = {s.frame_index: s for s in selected}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video}")

    saved_rel_paths: List[str] = []
    saved_labels: List[str] = []
    saved_confs: List[Optional[float]] = []

    frame_index = 0
    saved = 0
    missed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        info = wanted.get(frame_index)
        if info is not None:
            img = frame
            if bool(args.face_crop):
                box = _detect_face_haar(frame)
                if box is not None:
                    img = _crop_with_margin(frame, box, margin=float(args.crop_margin))

            fname = f"frame_{frame_index:06d}_{info.label}.jpg"
            out_path = images_dir / fname
            cv2.imwrite(str(out_path), img)

            rel = out_path.resolve().relative_to(REPO_ROOT)
            saved_rel_paths.append(str(rel).replace("\\", "/"))
            saved_labels.append(info.label)
            saved_confs.append(info.confidence)
            saved += 1

        frame_index += 1

    cap.release()

    # Count requested frames that exceeded video length.
    max_seen = frame_index - 1
    for k in wanted.keys():
        if k > max_seen:
            missed += 1

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(
        manifest_path=manifest_path,
        image_rel_paths=saved_rel_paths,
        labels=saved_labels,
        confidences=saved_confs,
        source=str(args.source),
        split=str(args.split).lower(),
    )

    summary = {
        "per_frame": str(per_frame),
        "video": str(video),
        "out_dir": str(out_dir),
        "mode": str(args.mode),
        "selected_frames": int(len(selected)),
        "saved_images": int(saved),
        "missed_due_to_video_length": int(missed),
        "min_frame_gap": int(args.min_frame_gap),
        "max_per_class": int(args.max_per_class),
        "require_not_unstable": bool(args.require_not_unstable),
        "face_crop": bool(args.face_crop),
        "classes": list(CANONICAL_7),
    }
    (out_dir / "buffer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {out_dir / 'buffer_summary.json'}")
    print(f"Saved images: {saved} (selected={len(selected)}, missed={missed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
