from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


CANONICAL = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _norm_label(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


# Map common variants -> canonical
_LABEL_MAP: Dict[str, Optional[str]] = {
    # canonical
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
    # variants
    "anger": "Angry",
    "happiness": "Happy",
    "sadness": "Sad",
    "suprise": "Surprise",
    "surprised": "Surprise",
    # common extra classes to exclude
    "contempt": None,
    "none": None,
    "other": None,
    "unknown": None,
}


def map_to_canonical(folder_name: str) -> Optional[str]:
    key = _norm_label(folder_name)
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    return None


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def unique_name_for(src: Path, rel: Path) -> str:
    # Avoid collisions when different datasets share filenames
    h = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"{src.stem}__{h}{src.suffix}"


@dataclass
class FolderDatasetSpec:
    name: str
    root: Path


@dataclass
class YoloDatasetSpec:
    name: str
    root: Path
    yaml_path: Path


@dataclass
class RafDbBasicSpec:
    name: str
    root: Path
    label_file: Path


@dataclass
class RafCompoundSpec:
    name: str
    root: Path
    label_file: Path


@dataclass
class RafMlSpec:
    name: str
    root: Path
    distribution_file: Path
    partition_file: Path


def discover_folder_splits(root: Path) -> List[Path]:
    candidates = [
        root / "train",
        root / "test",
        root / "val",
        root / "valid",
        root / "validation",
    ]
    return [p for p in candidates if p.exists() and p.is_dir()]


def clean_folder_dataset(
    spec: FolderDatasetSpec,
    out_root: Path,
    mode: str,
    apply: bool,
) -> dict:
    report: dict = {
        "type": "folder",
        "dataset": spec.name,
        "source": str(spec.root),
        "output": str(out_root / spec.name),
        "splits": {},
        "excluded": {},
        "notes": [],
    }

    splits = discover_folder_splits(spec.root)
    if not splits:
        report["notes"].append("No standard split folders found; skipped.")
        return report

    for split_path in splits:
        split_name = split_path.name
        split_counts = {k: 0 for k in CANONICAL}
        excluded_counts: Dict[str, int] = {}

        for class_dir in sorted([p for p in split_path.iterdir() if p.is_dir()]):
            canonical = map_to_canonical(class_dir.name)
            files = [p for p in class_dir.rglob("*") if p.is_file()]
            if canonical is None:
                excluded_counts[class_dir.name] = excluded_counts.get(class_dir.name, 0) + len(files)
                continue

            for f in files:
                rel = f.relative_to(spec.root)
                # Re-home to: out_root/<dataset>/<split>/<Canonical>/<file>
                dst_dir = out_root / spec.name / split_name / canonical
                # Keep filename but avoid collisions
                dst = dst_dir / f.name
                if dst.exists():
                    dst = dst_dir / unique_name_for(f, rel)
                split_counts[canonical] += 1
                if apply:
                    safe_link_or_copy(f, dst, mode)

        report["splits"][split_name] = split_counts
        if excluded_counts:
            report["excluded"][split_name] = excluded_counts

    return report


def _parse_yolo_names(data: dict) -> List[str]:
    names = data.get("names")
    if isinstance(names, dict):
        # ultralytics sometimes uses {0: name, 1: name}
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return names
    raise ValueError("data.yaml missing 'names' list/dict")


def clean_yolo_dataset(spec: YoloDatasetSpec, out_root: Path, mode: str, apply: bool) -> dict:
    report: dict = {
        "type": "yolo",
        "dataset": spec.name,
        "source": str(spec.root),
        "output": str(out_root / spec.name),
        "yaml": str(spec.yaml_path),
        "old_names": None,
        "new_names": CANONICAL,
        "mapping": {},
        "label_lines_kept": 0,
        "label_lines_dropped": 0,
        "notes": [],
    }

    if not spec.yaml_path.exists():
        report["notes"].append("data.yaml not found; skipped.")
        return report

    data = yaml.safe_load(spec.yaml_path.read_text(encoding="utf-8"))
    old_names = _parse_yolo_names(data)
    report["old_names"] = old_names

    # Build mapping old_idx -> new_idx (or None to drop)
    new_index = {name: i for i, name in enumerate(CANONICAL)}
    mapping: Dict[int, Optional[int]] = {}
    for i, n in enumerate(old_names):
        canonical = map_to_canonical(n)
        mapping[i] = new_index[canonical] if canonical in new_index and canonical is not None else None

    report["mapping"] = {str(k): (None if v is None else int(v)) for k, v in mapping.items()}

    # Write updated yaml
    new_yaml = dict(data)
    new_yaml["names"] = CANONICAL
    new_yaml["nc"] = len(CANONICAL)

    # Copy/link tree + rewrite labels
    split_dirs = [p for p in [spec.root / "train", spec.root / "valid", spec.root / "test"] if p.exists()]
    if not split_dirs:
        report["notes"].append("No train/valid/test under YOLO root; skipped.")
        return report

    if apply:
        (out_root / spec.name).mkdir(parents=True, exist_ok=True)
        (out_root / spec.name / "data.yaml").write_text(yaml.safe_dump(new_yaml, sort_keys=False), encoding="utf-8")

    for split in split_dirs:
        for sub in ["images", "labels"]:
            src_sub = split / sub
            if not src_sub.exists():
                continue
            for f in src_sub.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(spec.root)
                out_path = out_root / spec.name / rel

                if sub == "images":
                    if apply:
                        safe_link_or_copy(f, out_path, mode)
                    continue

                # labels: rewrite
                text = f.read_text(encoding="utf-8").strip().splitlines()
                new_lines: List[str] = []
                for line in text:
                    if not line.strip():
                        continue
                    parts = line.split()
                    try:
                        old_cls = int(float(parts[0]))
                    except Exception:
                        report["label_lines_dropped"] += 1
                        continue
                    new_cls = mapping.get(old_cls)
                    if new_cls is None:
                        report["label_lines_dropped"] += 1
                        continue
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts))

                report["label_lines_kept"] += len(new_lines)

                if apply:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return report


_RAFDB_BASIC_ID_TO_CANONICAL: Dict[int, str] = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Angry",
    7: "Neutral",
}


def _find_raf_image(root: Path, filename: str) -> Optional[Path]:
    # Try aligned first: Image/aligned/aligned/<stem>_aligned.jpg
    stem = Path(filename).stem
    aligned = root / "Image" / "aligned" / "aligned" / f"{stem}_aligned.jpg"
    if aligned.exists():
        return aligned
    # Some datasets might not have suffix
    aligned2 = root / "Image" / "aligned" / "aligned" / filename
    if aligned2.exists():
        return aligned2
    original = root / "Image" / "original" / "original" / filename
    if original.exists():
        return original
    return None


def clean_rafdb_basic(spec: RafDbBasicSpec, out_root: Path, mode: str, apply: bool) -> dict:
    report: dict = {
        "type": "rafdb_basic",
        "dataset": spec.name,
        "source": str(spec.root),
        "output": str(out_root / spec.name),
        "splits": {"train": {k: 0 for k in CANONICAL}, "test": {k: 0 for k in CANONICAL}},
        "missing_images": 0,
        "notes": [],
    }

    if not spec.label_file.exists():
        report["notes"].append("list_patition_label.txt not found; skipped.")
        return report

    for line in spec.label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        fname, label_s = parts[0], parts[1]
        try:
            label_id = int(label_s)
        except ValueError:
            continue
        canonical = _RAFDB_BASIC_ID_TO_CANONICAL.get(label_id)
        if canonical is None:
            continue
        split = "train" if fname.lower().startswith("train_") else "test" if fname.lower().startswith("test_") else "train"
        src_img = _find_raf_image(spec.root, fname)
        if src_img is None:
            report["missing_images"] += 1
            continue

        dst_dir = out_root / spec.name / split / canonical
        dst = dst_dir / src_img.name
        report["splits"][split][canonical] += 1
        if apply:
            safe_link_or_copy(src_img, dst, mode)

    return report


_RAF_COMPOUND_ID_TO_PRIMARY: Dict[int, str] = {
    1: "Happy",  # Happily Surprised
    2: "Happy",  # Happily Disgusted
    3: "Sad",    # Sadly Fearful
    4: "Sad",    # Sadly Angry
    5: "Sad",    # Sadly Surprised
    6: "Sad",    # Sadly Disgusted
    7: "Fear",   # Fearfully Angry
    8: "Fear",   # Fearfully Surprised
    9: "Angry",  # Angrily Surprised
    10: "Angry", # Angrily Disgusted
    11: "Disgust",# Disgustedly Surprised
}


def clean_raf_compound(spec: RafCompoundSpec, out_root: Path, mode: str, apply: bool) -> dict:
    report: dict = {
        "type": "raf_compound",
        "dataset": spec.name,
        "source": str(spec.root),
        "output": str(out_root / spec.name),
        "splits": {"train": {k: 0 for k in CANONICAL}, "test": {k: 0 for k in CANONICAL}},
        "missing_images": 0,
        "notes": [
            "RAF-DB-compound labels are 11 compound emotions. This cleaner maps each label to its primary/base emotion (Happy/Sad/Fear/Angry/Disgust). Neutral is not present.",
        ],
    }

    if not spec.label_file.exists():
        report["notes"].append("list_patition_label.txt not found; skipped.")
        return report

    for line in spec.label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        fname, label_s = parts[0], parts[1]
        try:
            label_id = int(label_s)
        except ValueError:
            continue
        canonical = _RAF_COMPOUND_ID_TO_PRIMARY.get(label_id)
        if canonical is None:
            continue
        split = "train" if fname.lower().startswith("train_") else "test" if fname.lower().startswith("test_") else "train"
        src_img = _find_raf_image(spec.root, fname)
        if src_img is None:
            report["missing_images"] += 1
            continue
        dst_dir = out_root / spec.name / split / canonical
        dst = dst_dir / src_img.name
        report["splits"][split][canonical] += 1
        if apply:
            safe_link_or_copy(src_img, dst, mode)

    return report


_RAFML_COLS: List[str] = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry"]


def clean_rafml(spec: RafMlSpec, out_root: Path, mode: str, apply: bool) -> dict:
    report: dict = {
        "type": "rafml",
        "dataset": spec.name,
        "source": str(spec.root),
        "output": str(out_root / spec.name),
        "splits": {"train": {k: 0 for k in CANONICAL}, "test": {k: 0 for k in CANONICAL}},
        "missing_images": 0,
        "notes": [
            "RAF-ML provides 6-dimensional distributions (Surprise/Fear/Disgust/Happiness/Sadness/Anger). This cleaner uses argmax to assign a single label. Neutral is not present.",
        ],
    }

    if not (spec.distribution_file.exists() and spec.partition_file.exists()):
        report["notes"].append("distribution.txt or partition_label.txt missing; skipped.")
        return report

    partition: Dict[str, int] = {}
    for line in spec.partition_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        partition[parts[0]] = int(parts[1])

    for line in spec.distribution_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        fname = parts[0]
        probs = []
        for p in parts[1:]:
            try:
                probs.append(float(p))
            except ValueError:
                probs.append(0.0)
        if len(probs) != 6:
            continue
        best = max(range(6), key=lambda i: probs[i])
        canonical = _RAFML_COLS[best]
        split = "test" if partition.get(fname, 0) == 1 else "train"

        src_img = _find_raf_image(spec.root, fname)
        if src_img is None:
            report["missing_images"] += 1
            continue
        dst_dir = out_root / spec.name / split / canonical
        dst = dst_dir / src_img.name
        report["splits"][split][canonical] += 1
        if apply:
            safe_link_or_copy(src_img, dst, mode)

    return report


def write_classification_manifest(out_root: Path, manifest_path: Path) -> dict:
    # Build CSV from folder datasets under out_root (skip YOLO-style).
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    split_map = {"valid": "val", "validation": "val", "val": "val"}

    rows: List[dict] = []
    counts = {k: 0 for k in CANONICAL}

    for dataset_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        if dataset_dir.name == "affectnet_yolo_format":
            continue
        for split_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            split_name = split_map.get(split_dir.name.lower(), split_dir.name.lower())
            if split_name not in {"train", "val", "test"}:
                # Treat everything else as train
                split_name = "train"
            for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                if class_dir.name not in CANONICAL:
                    continue
                for f in class_dir.rglob("*"):
                    if not f.is_file() or f.suffix.lower() not in exts:
                        continue
                    rel = f.relative_to(out_root)
                    rows.append(
                        {
                            "image_path": str(rel).replace("\\", "/"),
                            "label": class_dir.name,
                            "split": split_name,
                            "source": dataset_dir.name,
                        }
                    )
                    counts[class_dir.name] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_path", "label", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": len(rows), "counts": counts, "manifest": str(manifest_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean FER datasets to 7 emotions without modifying originals.")
    ap.add_argument(
        "--source-root",
        type=Path,
        default=Path("Training_data"),
        help="Root folder containing raw datasets.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
        help="Output folder for cleaned dataset views.",
    )
    ap.add_argument("--mode", choices=["link", "copy"], default="link", help="Hardlink or copy files.")
    ap.add_argument("--apply", action="store_true", help="Actually write output. Otherwise dry-run.")
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("Training_data_cleaned") / "clean_report.json",
        help="Where to write the JSON report (only when --apply).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
        help="Where to write the classification CSV manifest (only when --apply).",
    )
    args = ap.parse_args()

    src = args.source_root
    out = args.out_root

    folder_specs = [
        FolderDatasetSpec(
            name="affectnet_full_balanced",
            root=src / "affectnet full balanced dataset" / "affectnet_full_balanced",
        ),
        FolderDatasetSpec(
            name="fer2013_uniform_7",
            root=src
            / "FER-2013 7-emotions Uniform Dataset"
            / "FER2013_7emotions_Uniform_Augmented_Dataset",
        ),
        FolderDatasetSpec(name="ferplus", root=src / "FERPlus"),
    ]

    yolo_spec = YoloDatasetSpec(
        name="affectnet_yolo_format",
        root=src / "Facial Expression Image Data AFFECTNET YOLO Format" / "YOLO_format",
        yaml_path=src / "Facial Expression Image Data AFFECTNET YOLO Format" / "YOLO_format" / "data.yaml",
    )

    rafdb_basic = RafDbBasicSpec(
        name="rafdb_basic",
        root=src / "RAFDB-basic" / "basic",
        label_file=src / "RAFDB-basic" / "basic" / "EmoLabel" / "list_patition_label.txt",
    )
    raf_compound = RafCompoundSpec(
        name="rafdb_compound_mapped",
        root=src / "RAF-DB-compound" / "compound",
        label_file=src / "RAF-DB-compound" / "compound" / "EmoLabel" / "list_patition_label.txt",
    )
    rafml = RafMlSpec(
        name="rafml_argmax",
        root=src / "RAF-ML" / "RAF-ML",
        distribution_file=src / "RAF-ML" / "RAF-ML" / "EmoLabel" / "distribution.txt",
        partition_file=src / "RAF-ML" / "RAF-ML" / "EmoLabel" / "partition_label.txt",
    )

    full_report = {
        "canonical": CANONICAL,
        "mode": args.mode,
        "apply": bool(args.apply),
        "source_root": str(src),
        "out_root": str(out),
        "reports": [],
        "warnings": [
            "RAF-AU provides Action Units (AUs), not 7-class emotion labels; it is not converted here.",
        ],
    }

    for spec in folder_specs:
        full_report["reports"].append(clean_folder_dataset(spec, out, args.mode, args.apply))

    full_report["reports"].append(clean_yolo_dataset(yolo_spec, out, args.mode, args.apply))

    full_report["reports"].append(clean_rafdb_basic(rafdb_basic, out, args.mode, args.apply))
    full_report["reports"].append(clean_raf_compound(raf_compound, out, args.mode, args.apply))
    full_report["reports"].append(clean_rafml(rafml, out, args.mode, args.apply))

    if args.apply:
        out.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        manifest_summary = write_classification_manifest(out, args.manifest)
        full_report["manifest"] = manifest_summary
        args.report.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    # Print quick summary
    print("apply:", args.apply)
    print("mode:", args.mode)
    print("out_root:", out)
    for r in full_report["reports"]:
        print(f"- {r['dataset']}: {r['type']}")
    if args.apply:
        print("report:", args.report)
        print("manifest:", args.manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
