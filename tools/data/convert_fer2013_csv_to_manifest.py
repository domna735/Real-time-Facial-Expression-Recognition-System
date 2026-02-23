from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


# FER2013 Kaggle/ICML label mapping is typically:
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
IDX_TO_LABEL: Tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)


@dataclass(frozen=True)
class Fer2013Row:
    emotion: int
    pixels: str
    usage: str


def _iter_fer2013_csv_rows(path: Path) -> Iterable[Fer2013Row]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            emo_s = (r.get("emotion") or "").strip()
            pix = (r.get("pixels") or "").strip()
            usage = (r.get("Usage") or r.get("usage") or "").strip()
            if not emo_s or not pix or not usage:
                continue
            try:
                emo = int(emo_s)
            except Exception:
                continue
            yield Fer2013Row(emotion=emo, pixels=pix, usage=usage)


def _save_48x48_grayscale_png(*, pixels: str, out_path: Path) -> None:
    parts = pixels.split()
    if len(parts) != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {len(parts)}")

    try:
        vals = [int(p) for p in parts]
    except Exception as e:
        raise ValueError("Pixels are not all ints") from e

    if any((v < 0 or v > 255) for v in vals):
        raise ValueError("Pixel values out of [0,255]")

    img = Image.new("L", (48, 48))
    img.putdata(vals)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _usage_to_split(usage: str, *, mode: str) -> str:
    # Mode controls how we tag split in the manifest.
    # - eval: selected Usage rows become split=test (so eval_student_checkpoint finds them)
    # - full: preserve semantics (Training->train, PublicTest->val, PrivateTest->test)
    u = usage.strip().lower()
    if mode == "full":
        if u == "training":
            return "train"
        if u == "publictest":
            return "val"
        if u == "privatetest":
            return "test"
        return "train"

    # eval mode
    return "test"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Convert Kaggle/ICML FER2013 fer2013.csv into PNG images + a manifest CSV compatible with this repo. "
            "This does not download data; it only converts a local fer2013.csv you provide."
        )
    )
    ap.add_argument("--fer2013-csv", type=Path, required=True, help="Path to fer2013.csv (Kaggle/ICML format)")
    ap.add_argument(
        "--usage",
        type=str,
        default="PublicTest",
        choices=["Training", "PublicTest", "PrivateTest"],
        help="Which Usage split to export",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data") / "FER2013_official_from_csv",
        help="Where to write images/ and the manifest CSV",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "full"],
        help=(
            "How to set the manifest split column. 'eval' writes split=test for the exported rows. "
            "'full' preserves Training/PublicTest/PrivateTest as train/val/test."
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests (0 = no cap)",
    )

    args = ap.parse_args()

    src_csv: Path = args.fer2013_csv
    if not src_csv.exists():
        raise SystemExit(f"fer2013.csv not found: {src_csv}")

    usage_wanted = str(args.usage)
    out_root: Path = args.out_root

    img_dir = out_root / "images" / usage_wanted
    manifest_path = out_root / f"manifest__{usage_wanted.lower()}.csv"

    rows_out: List[Dict[str, str]] = []

    kept = 0
    dropped_bad_label = 0
    dropped_bad_pixels = 0

    for i, r in enumerate(_iter_fer2013_csv_rows(src_csv)):
        if r.usage != usage_wanted:
            continue

        if not (0 <= int(r.emotion) < len(IDX_TO_LABEL)):
            dropped_bad_label += 1
            continue

        label = IDX_TO_LABEL[int(r.emotion)]
        rel_img = (img_dir / f"fer2013_{usage_wanted.lower()}_{kept:06d}.png").as_posix()
        abs_img = Path(rel_img)

        try:
            _save_48x48_grayscale_png(pixels=r.pixels, out_path=abs_img)
        except Exception:
            dropped_bad_pixels += 1
            continue

        rows_out.append(
            {
                "image_path": rel_img.replace("\\", "/"),
                "label": label,
                "split": _usage_to_split(r.usage, mode=str(args.mode)),
                "source": f"fer2013_{usage_wanted.lower()}",
            }
        )
        kept += 1

        if args.max_rows and kept >= int(args.max_rows):
            break

    out_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_path", "label", "split", "source"])
        w.writeheader()
        for rr in rows_out:
            w.writerow(rr)

    print(
        {
            "fer2013_csv": src_csv.as_posix(),
            "usage": usage_wanted,
            "out_root": out_root.as_posix(),
            "images_dir": img_dir.as_posix(),
            "manifest": manifest_path.as_posix(),
            "kept": kept,
            "dropped_bad_label": dropped_bad_label,
            "dropped_bad_pixels": dropped_bad_pixels,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
