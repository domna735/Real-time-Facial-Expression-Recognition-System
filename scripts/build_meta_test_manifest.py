from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


DEFAULT_FIELDS = [
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


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
        return rows


def iter_input_csvs(root: Path, include_classification_manifest_test: bool) -> List[Path]:
    csvs = sorted(root.glob("test_*.csv"))
    if include_classification_manifest_test:
        p = root / "classification_manifest.csv"
        if p.exists():
            csvs.append(p)
    return csvs


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    # Ensure required keys exist.
    for k in DEFAULT_FIELDS:
        row.setdefault(k, "")

    # Normalize split
    row["split"] = (row.get("split") or "").strip().lower()

    # Normalize label/source whitespace
    row["label"] = (row.get("label") or "").strip()
    row["source"] = (row.get("source") or "").strip()

    return row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bigger evaluation manifest by merging multiple existing test CSVs. "
            "This is intended for more stable ensemble-teacher evaluation (no train leakage)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Training_data_cleaned"),
        help="Root folder that contains test_*.csv manifests (default: Training_data_cleaned)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Training_data_cleaned") / "test_all_sources.csv",
        help="Output CSV path (default: Training_data_cleaned/test_all_sources.csv)",
    )
    parser.add_argument(
        "--include-classification-manifest-test",
        action="store_true",
        help=(
            "Also include rows from Training_data_cleaned/classification_manifest.csv with split==test. "
            "This makes the file bigger but may introduce duplicates if your test_*.csv already cover those rows."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root
    out: Path = args.out

    inputs = iter_input_csvs(root, args.include_classification_manifest_test)
    if not inputs:
        raise SystemExit(f"No input CSVs found under: {root}")

    kept: List[Dict[str, str]] = []
    seen_image_paths: Set[str] = set()
    sources = Counter()

    for p in inputs:
        rows = read_rows(p)
        if not rows:
            continue

        # Special-case: classification_manifest.csv could contain train/val/test.
        for row in rows:
            row = normalize_row(row)
            if row["split"] != "test":
                continue

            image_path = row.get("image_path", "")
            if not image_path:
                continue

            if image_path in seen_image_paths:
                continue

            seen_image_paths.add(image_path)
            kept.append(row)
            sources[row.get("source", "") or "(missing)"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row.get(k, "") for k in DEFAULT_FIELDS})

    print(f"WROTE\t{out}")
    print(f"ROWS\t{len(kept)}")
    print("SOURCES")
    for k, v in sources.most_common():
        print(f"  {k}\t{v}")
    print("INPUTS")
    for p in inputs:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
