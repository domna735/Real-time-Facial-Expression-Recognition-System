from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


@dataclass(frozen=True)
class SplitSummary:
    dataset: str
    split: str
    class_counts: Dict[str, int]
    total: int
    max_class: Optional[str]
    max_count: int
    min_class_nonzero: Optional[str]
    min_count_nonzero: int
    missing_classes: int
    imbalance_ratio: Optional[float]


def _count_files_nonrecursive(dir_path: Path) -> int:
    try:
        n = 0
        with os.scandir(dir_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if Path(entry.name).suffix.lower() in IMAGE_EXTS:
                    n += 1
        return n
    except FileNotFoundError:
        return 0


def _count_files_recursive(dir_path: Path) -> int:
    n = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                n += 1
    return n


def _count_images_in_class_dir(class_dir: Path) -> int:
    n = _count_files_nonrecursive(class_dir)
    if n > 0:
        return n
    return _count_files_recursive(class_dir)


def _is_class_folder_layout(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    # Heuristic: at least 2 subdirectories and some images inside them.
    subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False
    total = 0
    checked = 0
    for sd in subdirs[:12]:
        total += _count_files_nonrecursive(sd)
        checked += 1
    if checked == 0:
        return False
    return total > 0


def _find_split_dirs(dataset_root: Path, max_depth: int = 4) -> List[Path]:
    # Find directories named train/test/val/... up to max_depth.
    candidates: List[Path] = []
    dataset_root = dataset_root.resolve()

    def walk(cur: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for child in cur.iterdir():
                if not child.is_dir():
                    continue
                name = child.name.lower()
                if name in SPLIT_NAMES:
                    candidates.append(child)
                walk(child, depth + 1)
        except PermissionError:
            return

    walk(dataset_root, 0)

    # Filter to those that look like class-folder splits.
    good = [p for p in candidates if _is_class_folder_layout(p)]

    # If none look good, return raw candidates (might still be useful for parsing).
    return good if good else candidates


def _summarize_class_folder_split(dataset_name: str, split_dir: Path) -> Optional[SplitSummary]:
    if not split_dir.is_dir():
        return None
    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        return None

    counts: Dict[str, int] = {}
    for cd in class_dirs:
        counts[cd.name] = _count_images_in_class_dir(cd)

    # If it doesn't actually contain images, ignore.
    total = sum(counts.values())
    if total == 0:
        return None

    max_class = max(counts, key=lambda k: counts[k]) if counts else None
    max_count = counts[max_class] if max_class is not None else 0

    nonzero_items = [(k, v) for k, v in counts.items() if v > 0]
    if nonzero_items:
        min_class, min_count = min(nonzero_items, key=lambda kv: kv[1])
        imbalance = (max_count / min_count) if min_count > 0 else None
    else:
        min_class, min_count, imbalance = None, 0, None

    missing = sum(1 for _, v in counts.items() if v == 0)

    return SplitSummary(
        dataset=dataset_name,
        split=split_dir.name,
        class_counts=dict(sorted(counts.items(), key=lambda kv: kv[0].lower())),
        total=total,
        max_class=max_class,
        max_count=max_count,
        min_class_nonzero=min_class,
        min_count_nonzero=min_count,
        missing_classes=missing,
        imbalance_ratio=imbalance,
    )


def _try_parse_expw_label_lst(expw_root: Path) -> Optional[SplitSummary]:
    # ExpW raw layout typically has a label.lst with "path label" or similar.
    label_path = expw_root / "label.lst"
    if not label_path.exists():
        return None

    counts: Dict[str, int] = {}
    try:
        with label_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                label = parts[-1]
                counts[label] = counts.get(label, 0) + 1
    except OSError:
        return None

    if not counts:
        return None

    total = sum(counts.values())
    max_label = max(counts, key=lambda k: counts[k])
    max_count = counts[max_label]
    nonzero_items = [(k, v) for k, v in counts.items() if v > 0]
    min_label, min_count = min(nonzero_items, key=lambda kv: kv[1])
    imbalance = (max_count / min_count) if min_count > 0 else None

    return SplitSummary(
        dataset="Expression in-the-Wild (ExpW) Dataset",
        split="label.lst",
        class_counts=dict(sorted(counts.items(), key=lambda kv: kv[0])),
        total=total,
        max_class=max_label,
        max_count=max_count,
        min_class_nonzero=min_label,
        min_count_nonzero=min_count,
        missing_classes=0,
        imbalance_ratio=imbalance,
    )


def _fmt_ratio(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}"


def _write_markdown(out_path: Path, summaries: List[SplitSummary], notes: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Training_data class counts")
    lines.append("")
    lines.append("Source: `Training_data/` (auto-counted from folder structures and known label files).")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Summary (dataset + split)")
    lines.append("")
    lines.append(
        "| Dataset | Split | Total | #Classes | Missing | Max (class=count) | MinNonZero (class=count) | Imbalance (max/min) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        max_str = "N/A" if s.max_class is None else f"{s.max_class}={s.max_count}"
        min_str = (
            "N/A"
            if s.min_class_nonzero is None
            else f"{s.min_class_nonzero}={s.min_count_nonzero}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    s.dataset,
                    s.split,
                    str(s.total),
                    str(len(s.class_counts)),
                    str(s.missing_classes),
                    max_str,
                    min_str,
                    _fmt_ratio(s.imbalance_ratio),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Per-class counts")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s.dataset} — {s.split}")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|---|---:|")
        for cls, cnt in s.class_counts.items():
            lines.append(f"| {cls} | {cnt} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="Training_data",
        help="Path to Training_data root.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="table.md",
        help="Output markdown path.",
    )
    ap.add_argument(
        "--also-json",
        type=str,
        default="",
        help="Optional: also write a JSON dump to this path.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        raise SystemExit(f"Training_data root not found: {root}")

    notes: List[str] = []
    summaries: List[SplitSummary] = []

    # Parse ExpW label.lst if present.
    expw_dir = root / "Expression in-the-Wild (ExpW) Dataset"
    expw_summary = _try_parse_expw_label_lst(expw_dir)
    if expw_summary is not None:
        summaries.append(expw_summary)
        notes.append("ExpW counts are computed from label.lst label frequencies (not from image folders).")

    # Folder-based counts for each top-level dataset folder.
    for child in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        dataset_name = child.name
        # Skip ExpW here (already handled) but still allow split parsing if it has splits.
        split_dirs = _find_split_dirs(child)
        for sd in split_dirs:
            ss = _summarize_class_folder_split(dataset_name, sd)
            if ss is not None:
                summaries.append(ss)

    # De-duplicate identical dataset+split entries (can happen if multiple split dirs found).
    seen = set()
    uniq: List[SplitSummary] = []
    for s in summaries:
        key = (s.dataset, s.split, tuple(s.class_counts.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)

    # Sort for readability.
    uniq.sort(key=lambda s: (s.dataset.lower(), s.split.lower()))

    if not uniq:
        notes.append(
            "No class-folder splits detected under Training_data. If your data is primarily in Training_data_cleaned/, run a similar scan there instead."
        )

    _write_markdown(out, uniq, notes)

    if args.also_json:
        dump = [
            {
                "dataset": s.dataset,
                "split": s.split,
                "total": s.total,
                "class_counts": s.class_counts,
                "missing_classes": s.missing_classes,
                "max_class": s.max_class,
                "max_count": s.max_count,
                "min_class_nonzero": s.min_class_nonzero,
                "min_count_nonzero": s.min_count_nonzero,
                "imbalance_ratio": s.imbalance_ratio,
            }
            for s in uniq
        ]
        Path(args.also_json).write_text(json.dumps(dump, indent=2), encoding="utf-8")

    print(f"Wrote {out} with {len(uniq)} dataset/split summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
