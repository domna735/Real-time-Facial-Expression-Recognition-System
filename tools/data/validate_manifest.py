from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import (  # noqa: E402
    CANONICAL_7,
    ManifestRow,
    read_manifest,
    resolve_image_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate classification_manifest.csv against Training_data_cleaned")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
    )
    ap.add_argument(
        "--check-bbox",
        action="store_true",
        help="If bbox columns exist, crop sampled images to validate bbox bounds",
    )
    ap.add_argument("--decode-samples", type=int, default=200, help="How many images to attempt to decode")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--write-json",
        type=Path,
        default=Path("outputs") / "manifest_validation.json",
        help="Write a JSON summary here",
    )
    args = ap.parse_args()

    rows = read_manifest(args.manifest)

    label_set = set(CANONICAL_7)
    bad_labels = 0
    missing_paths = 0
    by_source_split_label: Dict[Tuple[str, str, str], int] = defaultdict(int)

    valid_rows: List[ManifestRow] = []
    for r in rows:
        if r.label not in label_set:
            bad_labels += 1
            continue
        p = resolve_image_path(args.out_root, r.image_path)
        if not p.exists():
            missing_paths += 1
            continue
        valid_rows.append(r)
        by_source_split_label[(r.source, r.split, r.label)] += 1

    # Decode sample
    decode_ok = 0
    decode_fail = 0
    if args.decode_samples > 0 and valid_rows:
        # Deterministic pseudo-shuffle without importing random
        step = max(1, len(valid_rows) // max(1, args.decode_samples))
        sampled = valid_rows[::step][: args.decode_samples]
        for r in sampled:
            p = resolve_image_path(args.out_root, r.image_path)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if args.check_bbox and all(
                        v is not None
                        for v in (r.bbox_top, r.bbox_left, r.bbox_right, r.bbox_bottom)
                    ):
                        w, h = im.size
                        top = max(0, min(int(r.bbox_top), h - 1))
                        left = max(0, min(int(r.bbox_left), w - 1))
                        right = max(0, min(int(r.bbox_right), w))
                        bottom = max(0, min(int(r.bbox_bottom), h))
                        if right <= left:
                            right = min(w, left + 1)
                        if bottom <= top:
                            bottom = min(h, top + 1)
                        _ = im.crop((left, top, right, bottom))
                decode_ok += 1
            except Exception:
                decode_fail += 1

    # Summaries
    per_source = defaultdict(int)
    for r in valid_rows:
        per_source[r.source] += 1

    summary = {
        "manifest": str(args.manifest),
        "out_root": str(args.out_root),
        "rows_total": len(rows),
        "rows_valid": len(valid_rows),
        "bad_labels": bad_labels,
        "missing_paths": missing_paths,
        "decode": {
            "attempted": min(args.decode_samples, len(valid_rows)),
            "ok": decode_ok,
            "fail": decode_fail,
        },
        "counts": {
            "by_source": dict(sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_source_split_label": {
                f"{s}|{sp}|{l}": c for (s, sp, l), c in sorted(by_source_split_label.items())
            },
        },
        "canonical": list(CANONICAL_7),
    }

    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("rows_total:", summary["rows_total"])
    print("rows_valid:", summary["rows_valid"])
    print("bad_labels:", bad_labels)
    print("missing_paths:", missing_paths)
    print("decode_ok/fail:", decode_ok, decode_fail)
    print("json:", args.write_json)

    # Non-zero exit if clearly broken
    return 0 if (bad_labels == 0 and missing_paths == 0 and decode_fail == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
