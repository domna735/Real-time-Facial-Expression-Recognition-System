from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


COLUMNS = [
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


def _norm_split(v: str) -> str:
    v = (v or "").strip().lower()
    return v if v in {"train", "val", "test"} else "train"


def _read_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            yield {k: (r.get(k) or "").strip() for k in (reader.fieldnames or [])}


def _write_rows(out_csv: Path, rows: Iterable[Dict[str, str]]) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: (r.get(c) or "") for c in COLUMNS})
            n += 1
    return n


def _filter_manifest(
    *,
    base_manifest: Path,
    sources_keep: Set[str],
) -> Iterable[Dict[str, str]]:
    for r in _read_rows(base_manifest):
        src = (r.get("source") or "").strip()
        if src not in sources_keep:
            continue
        yield {
            "image_path": (r.get("image_path") or "").strip(),
            "label": (r.get("label") or "").strip(),
            "split": _norm_split(r.get("split") or ""),
            "source": src,
            "confidence": (r.get("confidence") or "").strip(),
            "orig_image": (r.get("orig_image") or "").strip(),
            "face_id": (r.get("face_id") or "").strip(),
            "bbox_top": (r.get("bbox_top") or "").strip(),
            "bbox_left": (r.get("bbox_left") or "").strip(),
            "bbox_right": (r.get("bbox_right") or "").strip(),
            "bbox_bottom": (r.get("bbox_bottom") or "").strip(),
        }


def _dedupe_rows(
    rows: Iterable[Dict[str, str]],
    *,
    key: str,
) -> Iterable[Dict[str, str]]:
    """Drop duplicates while preserving first-seen order.

    Supported keys:
    - image_path: exact path match
    - expw_orig_face: (orig_image, face_id) if present, otherwise fallback to image_path
    """

    seen: Set[str] = set()
    for r in rows:
        if key == "image_path":
            k = (r.get("image_path") or "").strip()
        elif key == "expw_orig_face":
            orig = (r.get("orig_image") or "").strip()
            face = (r.get("face_id") or "").strip()
            if orig and face:
                k = f"orig_image={orig}|face_id={face}"
            else:
                k = (r.get("image_path") or "").strip()
        else:
            raise ValueError(f"Unsupported dedupe key: {key}")

        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        yield r


def summarize(csv_path: Path) -> Dict[str, Counter]:
    by_source = Counter()
    by_source_split = Counter()
    for r in _read_rows(csv_path):
        src = (r.get("source") or "").strip() or "<missing>"
        split = _norm_split(r.get("split") or "")
        by_source[src] += 1
        by_source_split[(src, split)] += 1
    return {"by_source": by_source, "by_source_split": by_source_split}


def main() -> None:
    p = argparse.ArgumentParser(description="Build curated train/eval manifests from existing FER manifests.")
    p.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("Training_data_cleaned/classification_manifest.csv"),
        help="Unified manifest to filter (default: Training_data_cleaned/classification_manifest.csv)",
    )
    p.add_argument(
        "--expw-hq-manifest",
        type=Path,
        default=Path("Training_data_cleaned/expw_hq_manifest.csv"),
        help="ExpW HQ manifest to optionally append (default: Training_data_cleaned/expw_hq_manifest.csv)",
    )
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("Training_data_cleaned/classification_manifest_hq_train.csv"),
        help="Output curated training manifest CSV",
    )
    p.add_argument(
        "--out-eval",
        type=Path,
        default=Path("Training_data_cleaned/classification_manifest_eval_only.csv"),
        help="Output evaluation-only manifest CSV",
    )
    p.add_argument(
        "--train-sources",
        type=str,
        default="ferplus,rafdb_basic,affectnet_full_balanced",
        help="Comma-separated source names to keep for training from base-manifest.",
    )
    p.add_argument(
        "--eval-sources",
        type=str,
        default="expw_full,rafdb_compound_mapped,rafml_argmax,affectnet_yolo_format",
        help="Comma-separated source names to keep for evaluation from base-manifest.",
    )
    p.add_argument(
        "--include-expw-hq-in-train",
        action="store_true",
        help="Append expw_hq_manifest.csv rows into the training manifest.",
    )
    p.add_argument(
        "--include-expw-hq-in-eval",
        action="store_true",
        help="Append expw_hq_manifest.csv rows into the eval manifest.",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicated rows in outputs (preserves first-seen order).",
    )
    p.add_argument(
        "--dedupe-key",
        type=str,
        default="image_path",
        choices=["image_path", "expw_orig_face"],
        help="De-duplication key (default: image_path).",
    )

    args = p.parse_args()

    base_manifest = args.base_manifest
    expw_hq_manifest = args.expw_hq_manifest

    if not base_manifest.exists():
        raise SystemExit(f"Base manifest not found: {base_manifest}")
    if (args.include_expw_hq_in_train or args.include_expw_hq_in_eval) and not expw_hq_manifest.exists():
        raise SystemExit(f"ExpW HQ manifest not found: {expw_hq_manifest}")

    train_sources = {s.strip() for s in args.train_sources.split(",") if s.strip()}
    eval_sources = {s.strip() for s in args.eval_sources.split(",") if s.strip()}

    def _iter_train() -> Iterable[Dict[str, str]]:
        rows: Iterable[Dict[str, str]]
        def _raw() -> Iterable[Dict[str, str]]:
            yield from _filter_manifest(base_manifest=base_manifest, sources_keep=train_sources)
            if args.include_expw_hq_in_train:
                yield from _filter_manifest(base_manifest=expw_hq_manifest, sources_keep={"expw_hq"})

        rows = _raw()
        if args.dedupe:
            rows = _dedupe_rows(rows, key=args.dedupe_key)
        yield from rows

    def _iter_eval() -> Iterable[Dict[str, str]]:
        rows: Iterable[Dict[str, str]]
        def _raw() -> Iterable[Dict[str, str]]:
            yield from _filter_manifest(base_manifest=base_manifest, sources_keep=eval_sources)
            if args.include_expw_hq_in_eval:
                yield from _filter_manifest(base_manifest=expw_hq_manifest, sources_keep={"expw_hq"})

        rows = _raw()
        if args.dedupe:
            rows = _dedupe_rows(rows, key=args.dedupe_key)
        yield from rows

    n_train = _write_rows(args.out_train, _iter_train())
    n_eval = _write_rows(args.out_eval, _iter_eval())

    print(f"Wrote train manifest: {args.out_train} (rows={n_train})")
    print(f"Wrote eval  manifest: {args.out_eval} (rows={n_eval})")

    # Small summaries (counts by source/split) to guide sanity checks.
    train_summary = summarize(args.out_train)
    eval_summary = summarize(args.out_eval)

    print("\nTrain by_source:")
    for k, v in train_summary["by_source"].most_common():
        print(f"  {k}: {v}")

    print("\nEval by_source:")
    for k, v in eval_summary["by_source"].most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
