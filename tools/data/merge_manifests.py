from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable


BASE_FIELDS = ["image_path", "label", "split", "source"]


def iter_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            yield r


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge multiple manifest CSVs into a single 4-column manifest")
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
        help="Existing unified manifest (4 columns)",
    )
    ap.add_argument(
        "--add",
        type=Path,
        action="append",
        default=[],
        help="Additional manifest(s) to append (can have extra columns)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
        help="Output manifest path",
    )

    args = ap.parse_args()

    if not args.base.exists():
        print("ERROR: base manifest not found:", args.base)
        return 2

    if not args.add:
        print("ERROR: no --add manifests provided")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # If writing to same path as base, write to a temp file then replace.
    out_path = args.out
    tmp_path = out_path
    if out_path.resolve() == args.base.resolve():
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    written = 0
    sources_seen = set()

    with tmp_path.open("w", newline="", encoding="utf-8") as out_fp:
        writer = csv.DictWriter(out_fp, fieldnames=BASE_FIELDS)
        writer.writeheader()

        def write_from(path: Path) -> None:
            nonlocal written
            for r in iter_rows(path):
                row = {
                    "image_path": (r.get("image_path") or "").strip(),
                    "label": (r.get("label") or "").strip(),
                    "split": (r.get("split") or "").strip(),
                    "source": (r.get("source") or "").strip(),
                }
                if not row["image_path"] or not row["label"]:
                    continue
                if row["source"]:
                    sources_seen.add(row["source"])
                writer.writerow(row)
                written += 1

        write_from(args.base)
        for p in args.add:
            if not p.exists():
                print("ERROR: add manifest not found:", p)
                return 2
            write_from(p)

    if tmp_path != out_path:
        os.replace(str(tmp_path), str(out_path))

    print("wrote_rows:", written)
    print("sources:", ", ".join(sorted(sources_seen)))
    print("out:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
