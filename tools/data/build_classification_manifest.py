from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


CANONICAL = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "test": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


FIELDNAMES = [
    "image_path",
    "label",
    "split",
    "source",
    # Optional ExpW-style metadata (blank for non-ExpW rows)
    "confidence",
    "orig_image",
    "face_id",
    "bbox_top",
    "bbox_left",
    "bbox_right",
    "bbox_bottom",
]


def iter_image_files(folder: Path) -> Iterator[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def detect_labeled_folder_dataset(dataset_dir: Path) -> bool:
    # Must contain at least one recognized split directory.
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.lower() in SPLIT_MAP:
            return True
    return False


def write_folder_dataset_rows(out_root: Path, dataset_dir: Path, writer: csv.DictWriter) -> int:
    written = 0
    dataset_name = dataset_dir.name

    for split_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        split_key = split_dir.name.lower()
        if split_key not in SPLIT_MAP:
            continue
        split = SPLIT_MAP[split_key]

        for label_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label = label_dir.name
            if label not in CANONICAL:
                continue
            for img_path in iter_image_files(label_dir):
                rel = img_path.relative_to(out_root).as_posix()
                writer.writerow(
                    {
                        "image_path": rel,
                        "label": label,
                        "split": split,
                        "source": dataset_name,
                        "confidence": "",
                        "orig_image": "",
                        "face_id": "",
                        "bbox_top": "",
                        "bbox_left": "",
                        "bbox_right": "",
                        "bbox_bottom": "",
                    }
                )
                written += 1

    return written


def append_expw_manifest(expw_manifest: Path, writer: csv.DictWriter) -> int:
    written = 0
    with expw_manifest.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            image_path = (r.get("image_path") or "").strip()
            label = (r.get("label") or "").strip()
            split = (r.get("split") or "").strip()
            source = (r.get("source") or "").strip() or "expw_full"
            confidence = (r.get("confidence") or "").strip()
            orig_image = (r.get("orig_image") or "").strip()
            face_id = (r.get("face_id") or "").strip()
            bbox_top = (r.get("bbox_top") or "").strip()
            bbox_left = (r.get("bbox_left") or "").strip()
            bbox_right = (r.get("bbox_right") or "").strip()
            bbox_bottom = (r.get("bbox_bottom") or "").strip()
            if not image_path or not label:
                continue
            # Normalize split
            split = split.lower()
            split = split if split in {"train", "val", "test"} else (SPLIT_MAP.get(split, "train"))
            writer.writerow(
                {
                    "image_path": image_path,
                    "label": label,
                    "split": split,
                    "source": source,
                    "confidence": confidence,
                    "orig_image": orig_image,
                    "face_id": face_id,
                    "bbox_top": bbox_top,
                    "bbox_left": bbox_left,
                    "bbox_right": bbox_right,
                    "bbox_bottom": bbox_bottom,
                }
            )
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild Training_data_cleaned/classification_manifest.csv")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
        help="Root containing cleaned datasets",
    )
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
        help="Where to write the unified manifest",
    )
    ap.add_argument(
        "--include-expw-manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "expw_full_manifest.csv",
        help="Append ExpW-full rows from this manifest (set empty to skip)",
    )
    ap.add_argument(
        "--skip-expw",
        action="store_true",
        help="Do not append ExpW-full rows",
    )

    args = ap.parse_args()

    out_root = args.out_root
    if not out_root.exists():
        print("ERROR: out_root not found:", out_root)
        return 2

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Write manifest
    base_written = 0
    expw_written = 0

    with args.out_manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()

        for dataset_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
            name = dataset_dir.name
            if name.startswith("."):
                continue
            # Keep ExpW HQ separate; it has its own manifest.
            if name in {"expw_hq"}:
                continue
            # Skip non-folder-labeled datasets (e.g., YOLO)
            if not detect_labeled_folder_dataset(dataset_dir):
                continue

            base_written += write_folder_dataset_rows(out_root, dataset_dir, writer)

        if not args.skip_expw:
            expw_manifest = args.include_expw_manifest
            if expw_manifest and expw_manifest.exists():
                expw_written = append_expw_manifest(expw_manifest, writer)
            else:
                print("WARN: ExpW manifest not found, skipping:", expw_manifest)

    print("base_rows:", base_written)
    print("expw_rows:", expw_written)
    print("total_rows:", base_written + expw_written)
    print("out:", args.out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
