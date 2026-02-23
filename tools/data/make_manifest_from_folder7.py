from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANONICAL_7: Tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)


def _norm_label(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch.isalnum())


# Accept common folder variants
FOLDER_TO_CANONICAL: Dict[str, Optional[str]] = {
    "angry": "Angry",
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "happiness": "Happy",
    "sad": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "neutral": "Neutral",
    # drop/ignore extras if present
    "contempt": None,
    "unknown": None,
    "other": None,
}


def _map_folder_to_label(folder_name: str) -> Optional[str]:
    key = _norm_label(folder_name)
    if key in FOLDER_TO_CANONICAL:
        return FOLDER_TO_CANONICAL[key]
    return None


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@dataclass(frozen=True)
class Row:
    image_path: str  # posix-like, relative to data_root
    label: str
    split: str
    source: str


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a canonical manifest CSV from a folder-based 7-emotion dataset. "
            "Expected layout like <root>/{train,test}/{angry,disgust,...}/*.jpg"
        )
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder containing split subfolders (e.g., Training_data/FER2013)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output manifest path (e.g., Training_data/fer2013_kaggle_msambare_manifest.csv)",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="fer2013_kaggle_msambare",
        help="Value to put in the manifest 'source' column",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to include (default: train,test)",
    )

    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    split_names = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not split_names:
        raise SystemExit("No splits specified")

    rows: List[Row] = []
    dropped_unknown_label = 0

    for split in split_names:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise SystemExit(f"split dir not found: {split_dir}")

        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label = _map_folder_to_label(class_dir.name)
            if label is None:
                # unknown folder or excluded label
                dropped_unknown_label += 1
                continue

            for img_path in _iter_images(class_dir):
                rel = img_path.relative_to(dataset_root)
                # Our loader resolves relative paths against eval-data-root.
                # So store path relative to the dataset root's parent is NOT safe.
                # We store it relative to repo root by default usage (eval-data-root '.')
                # therefore: prefix with dataset_root (posix) relative to repo root.
                # If dataset_root is already relative (e.g. Training_data/FER2013), this is perfect.
                rel_repo = (dataset_root / rel).as_posix()

                rows.append(
                    Row(
                        image_path=rel_repo,
                        label=label,
                        split=split,
                        source=str(args.source),
                    )
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_path", "label", "split", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_path": r.image_path, "label": r.label, "split": r.split, "source": r.source})

    print(
        {
            "dataset_root": dataset_root.as_posix(),
            "out": args.out.as_posix(),
            "splits": split_names,
            "rows": len(rows),
            "dropped_unknown_label_dirs": dropped_unknown_label,
            "labels": list(CANONICAL_7),
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
