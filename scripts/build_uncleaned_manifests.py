from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_7 = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _resolve_aligned_dir(base_aligned_dir: Path) -> Tuple[Path, str]:
        """Return (actual_aligned_dir, rel_suffix).

        Some uncleaned RAF-style datasets end up nested like:
            .../Image/aligned/aligned/*.jpg

        while others are:
            .../Image/aligned/*.jpg

        rel_suffix is the relative folder segment(s) to append to the manifest path.
        """
        nested = base_aligned_dir / "aligned"
        if nested.exists() and nested.is_dir():
                return nested, "aligned/aligned"
        return base_aligned_dir, "aligned"


def _write_csv(out_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    fieldnames = [
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
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_list:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield ln


def build_rafdb_basic(*, data_root: Path, out_csv: Path) -> None:
    # RAFDB-basic label mapping (standard RAF):
    # 1 Surprise, 2 Fear, 3 Disgust, 4 Happy, 5 Sad, 6 Angry, 7 Neutral
    idx_to_label = {
        1: "Surprise",
        2: "Fear",
        3: "Disgust",
        4: "Happy",
        5: "Sad",
        6: "Angry",
        7: "Neutral",
    }

    label_file = data_root / "RAFDB-basic" / "basic" / "EmoLabel" / "list_patition_label.txt"
    base_img_dir = data_root / "RAFDB-basic" / "basic" / "Image" / "aligned"
    if not label_file.exists():
        raise SystemExit(f"Not found: {label_file}")
    if not base_img_dir.exists():
        raise SystemExit(f"Not found: {base_img_dir}")

    img_dir, rel_aligned_suffix = _resolve_aligned_dir(base_img_dir)

    rows: List[Dict[str, str]] = []
    missing = 0

    for ln in _iter_lines(label_file):
        parts = ln.split()
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        try:
            lab_i = int(parts[1])
        except Exception:
            continue
        label = idx_to_label.get(lab_i)
        if label is None:
            continue

        split = "train"
        if name.startswith("test_"):
            split = "test"
        elif name.startswith("train_"):
            split = "train"

        aligned_name = name
        if aligned_name.lower().endswith(".jpg"):
            aligned_name = aligned_name[:-4] + "_aligned.jpg"
        else:
            aligned_name = aligned_name + "_aligned.jpg"

        rel_parts = rel_aligned_suffix.split("/")
        rel_path = Path("RAFDB-basic") / "basic" / "Image" / Path(*rel_parts) / aligned_name
        full_path = data_root / rel_path
        if not full_path.exists():
            missing += 1
            # keep row anyway for now; exporter/build_splits will drop missing

        rows.append(
            {
                "image_path": rel_path.as_posix(),
                "label": label,
                "split": split,
                "source": "rafdb_basic_uncleaned",
            }
        )

    _write_csv(out_csv, rows)
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(rows)} | Missing aligned images: {missing}")


def build_rafml_argmax(*, data_root: Path, out_csv: Path) -> None:
    # RAF-ML distribution columns: Surprise, Fear, Disgust, Happiness, Sadness, Anger
    cols = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry"]

    dist_file = data_root / "RAF-ML" / "RAF-ML" / "EmoLabel" / "distribution.txt"
    part_file = data_root / "RAF-ML" / "RAF-ML" / "EmoLabel" / "partition_label.txt"
    base_img_dir = data_root / "RAF-ML" / "RAF-ML" / "Image" / "aligned"
    if not dist_file.exists():
        raise SystemExit(f"Not found: {dist_file}")
    if not part_file.exists():
        raise SystemExit(f"Not found: {part_file}")
    if not base_img_dir.exists():
        raise SystemExit(f"Not found: {base_img_dir}")

    img_dir, rel_aligned_suffix = _resolve_aligned_dir(base_img_dir)

    # partition_label: <name> <0/1>
    split_by_name: Dict[str, str] = {}
    for ln in _iter_lines(part_file):
        parts = ln.split()
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        try:
            flag = int(parts[1])
        except Exception:
            continue
        # By convention in RAF-ML experiments: 0=train, 1=test
        split_by_name[name] = "test" if flag == 1 else "train"

    rows: List[Dict[str, str]] = []
    missing = 0
    unknown_split = 0

    for ln in _iter_lines(dist_file):
        parts = ln.split()
        if len(parts) < 7:
            continue
        name = parts[0].strip()
        try:
            probs = [float(x) for x in parts[1:7]]
        except Exception:
            continue
        k = max(range(6), key=lambda i: probs[i])
        label = cols[k]

        split = split_by_name.get(name)
        if split is None:
            unknown_split += 1
            split = "train"

        aligned_name = name
        if aligned_name.lower().endswith(".jpg"):
            aligned_name = aligned_name[:-4] + "_aligned.jpg"
        else:
            aligned_name = aligned_name + "_aligned.jpg"

        rel_parts = rel_aligned_suffix.split("/")
        rel_path = Path("RAF-ML") / "RAF-ML" / "Image" / Path(*rel_parts) / aligned_name
        full_path = data_root / rel_path
        if not full_path.exists():
            missing += 1

        rows.append(
            {
                "image_path": rel_path.as_posix(),
                "label": label,
                "split": split,
                "source": "rafml_argmax_uncleaned",
            }
        )

    _write_csv(out_csv, rows)
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(rows)} | Missing aligned images: {missing} | Unknown split: {unknown_split}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build minimal classification manifests from uncleaned Training_data.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "Training_data",
        help="Root folder that contains the uncleaned datasets.",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_basic = sub.add_parser("rafdb-basic", help="Build manifest from RAFDB-basic")
    ap_basic.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "Training_data" / "uncleaned_manifests" / "rafdb_basic_manifest.csv",
    )

    ap_ml = sub.add_parser("rafml-argmax", help="Build manifest from RAF-ML distribution (argmax)")
    ap_ml.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "Training_data" / "uncleaned_manifests" / "rafml_argmax_manifest.csv",
    )

    args = ap.parse_args()
    data_root = Path(args.data_root)

    if args.cmd == "rafdb-basic":
        build_rafdb_basic(data_root=data_root, out_csv=Path(args.out))
    elif args.cmd == "rafml-argmax":
        build_rafml_argmax(data_root=data_root, out_csv=Path(args.out))
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
