from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


LABEL_ID_TO_CANONICAL: Dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


@dataclass(frozen=True)
class ExpwRow:
    image_name: str
    face_id: int
    top: int
    left: int
    right: int
    bottom: int
    confidence: float
    label_id: int


def parse_label_lst(path: Path) -> List[ExpwRow]:
    rows: List[ExpwRow] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        try:
            rows.append(
                ExpwRow(
                    image_name=parts[0],
                    face_id=int(parts[1]),
                    top=int(float(parts[2])),
                    left=int(float(parts[3])),
                    right=int(float(parts[4])),
                    bottom=int(float(parts[5])),
                    confidence=float(parts[6]),
                    label_id=int(parts[7]),
                )
            )
        except Exception:
            continue
    return rows


def clamp_bbox(top: int, left: int, right: int, bottom: int, w: int, h: int) -> Tuple[int, int, int, int]:
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w))
    bottom = max(0, min(bottom, h))
    if right <= left:
        right = min(w, left + 1)
    if bottom <= top:
        bottom = min(h, top + 1)
    return top, left, right, bottom


def pad_bbox(
    top: int,
    left: int,
    right: int,
    bottom: int,
    *,
    w: int,
    h: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    if pad_ratio <= 0:
        return top, left, right, bottom
    bw = max(1, right - left)
    bh = max(1, bottom - top)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    return clamp_bbox(top - pad_y, left - pad_x, right + pad_x, bottom + pad_y, w=w, h=h)


def stratified_split(items: List[ExpwRow], train_frac: float, val_frac: float, seed: int) -> Dict[int, str]:
    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(items):
        by_label[r.label_id].append(i)

    split_for_index: Dict[int, str] = {}
    rng = random.Random(seed)
    for label_id, indices in by_label.items():
        indices = indices[:]
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        for idx in train_idx:
            split_for_index[idx] = "train"
        for idx in val_idx:
            split_for_index[idx] = "val"
        for idx in test_idx:
            split_for_index[idx] = "test"

    return split_for_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a separate high-quality ExpW manifest + optional face crops")
    ap.add_argument(
        "--expw-root",
        type=Path,
        default=Path("Training_data") / "Expression in-the-Wild (ExpW) Dataset",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Where ExpW images are extracted (default: <expw-root>/origin)",
    )
    ap.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="Path to label.lst (default: <expw-root>/label.lst)",
    )
    ap.add_argument("--min-confidence", type=float, default=60.0, help="Filter low-confidence face boxes")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
    )
    ap.add_argument(
        "--out-dataset",
        type=str,
        default="expw_hq",
        help="Folder name under Training_data_cleaned/ for cropped ExpW outputs",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "expw_hq_manifest.csv",
        help="Where to write the ExpW-specific manifest",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("outputs") / "expw_hq_import_report.json",
        help="Where to write a JSON summary",
    )
    ap.add_argument(
        "--write-crops",
        action="store_true",
        help="Actually write cropped face images under Training_data_cleaned/<out-dataset>/...",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for written crops (higher = larger files; ignored if not writing crops)",
    )
    ap.add_argument(
        "--jpeg-subsampling",
        type=int,
        default=0,
        help="JPEG subsampling (0=best quality, 2=smaller). Ignored if not writing crops.",
    )
    ap.add_argument(
        "--bbox-pad-ratio",
        type=float,
        default=0.0,
        help="Pad bbox by this fraction of bbox size before cropping (e.g., 0.10 adds 10% context).",
    )
    ap.add_argument(
        "--absolute-paths",
        action="store_true",
        help="When not writing crops, store absolute image paths in the manifest (recommended).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For quick dry runs: only process the first N kept rows (0 = no limit)",
    )

    args = ap.parse_args()

    expw_root = args.expw_root
    label_file = args.label_file or (expw_root / "label.lst")
    images_dir = args.images_dir or (expw_root / "origin")

    if not label_file.exists():
        print("ERROR: label file not found:", label_file)
        return 2

    if not images_dir.exists():
        print("ERROR: ExpW images folder not found:", images_dir)
        print("Hint: extract the split archive (origin.7z.001 + origin.7z.002..008) into an 'origin' folder.")
        return 2

    # Common extraction layout is <expw-root>/origin/origin/*.jpg because the archive contains an 'origin/' folder.
    # If that's the case, transparently fix images_dir.
    nested_origin = images_dir / "origin"
    if nested_origin.exists() and nested_origin.is_dir():
        images_dir = nested_origin

    rows = parse_label_lst(label_file)
    kept: List[ExpwRow] = []
    missing_images = 0
    bad_label = 0

    for r in rows:
        if r.label_id not in LABEL_ID_TO_CANONICAL:
            bad_label += 1
            continue
        if r.confidence < args.min_confidence:
            continue
        if not (images_dir / r.image_name).exists():
            missing_images += 1
            continue
        kept.append(r)

    if args.limit and len(kept) > args.limit:
        kept = kept[: args.limit]

    if not kept:
        print("ERROR: no usable ExpW rows after filtering.")
        print("missing_images:", missing_images)
        print("bad_label:", bad_label)
        return 2

    split_for_index = stratified_split(kept, args.train_frac, args.val_frac, args.seed)

    out_dataset_dir = args.out_root / args.out_dataset
    counts = defaultdict(int)

    manifest_rows: List[dict] = []
    name_counters: Dict[str, int] = defaultdict(int)

    processed = 0
    written = 0

    for i, r in enumerate(kept):
        split = split_for_index.get(i, "train")
        label = LABEL_ID_TO_CANONICAL[r.label_id]
        src = images_dir / r.image_name

        if args.write_crops:
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                top, left, right, bottom = clamp_bbox(r.top, r.left, r.right, r.bottom, w=w, h=h)
                top, left, right, bottom = pad_bbox(
                    top,
                    left,
                    right,
                    bottom,
                    w=w,
                    h=h,
                    pad_ratio=args.bbox_pad_ratio,
                )
                face = im.crop((left, top, right, bottom))

            stem = Path(r.image_name).stem
            key = f"{stem}_f{r.face_id}"
            name_counters[key] += 1
            suffix = name_counters[key]
            out_name = f"{key}_{suffix:02d}.jpg"
            dst = out_dataset_dir / split / label / out_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            face.save(
                dst,
                format="JPEG",
                quality=int(args.jpeg_quality),
                subsampling=int(args.jpeg_subsampling),
                optimize=True,
            )
            written += 1
            rel_image_path = dst.relative_to(args.out_root).as_posix()
        else:
            # No crops written: point manifest at original image file.
            # Use absolute paths by default to avoid needing a different out_root.
            rel_image_path = src.resolve().as_posix() if args.absolute_paths else src.as_posix()

        manifest_rows.append(
            {
                "image_path": rel_image_path,
                "label": label,
                "split": split,
                "source": args.out_dataset,
                "confidence": f"{r.confidence:.4f}",
                "orig_image": r.image_name,
                "face_id": str(r.face_id),
                "bbox_top": str(r.top),
                "bbox_left": str(r.left),
                "bbox_right": str(r.right),
                "bbox_bottom": str(r.bottom),
            }
        )
        counts[f"{split}|{label}"] += 1

        processed += 1
        if args.write_crops and (processed % 5000 == 0):
            print(f"processed {processed}/{len(kept)} (written={written})")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image_path",
                "label",
                "split",
                "source",
                "confidence",
                "orig_image",
                "face_id",
                "bbox_top",
                "bbox_left",
                "bbox_right",
                "bbox_bottom",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "expw_root": str(expw_root),
        "images_dir": str(images_dir),
        "label_file": str(label_file),
        "min_confidence": args.min_confidence,
        "write_crops": bool(args.write_crops),
        "out_dataset": str(out_dataset_dir) if args.write_crops else None,
        "manifest": str(args.manifest),
        "rows_total": len(rows),
        "rows_kept": len(kept),
        "missing_images": missing_images,
        "bad_label": bad_label,
        "split_fracs": {"train": args.train_frac, "val": args.val_frac, "test": max(0.0, 1.0 - args.train_frac - args.val_frac)},
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("kept:", report["rows_kept"], "/", report["rows_total"])
    print("manifest:", args.manifest)
    print("report:", args.report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
