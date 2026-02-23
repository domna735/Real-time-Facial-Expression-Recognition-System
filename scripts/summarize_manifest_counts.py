from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


CANON_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@dataclass
class ManifestCounts:
    path: str
    total_rows: int
    splits: dict[str, int]
    label_totals: dict[str, int]
    split_label: dict[str, dict[str, int]]
    top_sources: list[tuple[str, int]]


def read_counts(csv_path: Path, top_k_sources: int = 10) -> ManifestCounts:
    split = Counter()
    label = Counter()
    split_label = Counter()
    source = Counter()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = (row.get("split") or "").strip()
            lb = (row.get("label") or "").strip()
            src = (row.get("source") or "").strip()
            split[sp] += 1
            label[lb] += 1
            split_label[(sp, lb)] += 1
            source[src] += 1

    split_label_map: dict[str, dict[str, int]] = {}
    for sp in sorted(split.keys()):
        split_label_map[sp] = {lb: int(split_label.get((sp, lb), 0)) for lb in CANON_LABELS}

    return ManifestCounts(
        path=csv_path.as_posix(),
        total_rows=int(sum(split.values())),
        splits={k: int(v) for k, v in split.items()},
        label_totals={lb: int(label.get(lb, 0)) for lb in CANON_LABELS},
        split_label=split_label_map,
        top_sources=[(k, int(v)) for k, v in source.most_common(top_k_sources)],
    )


def to_markdown(counts: list[ManifestCounts]) -> str:
    lines: list[str] = []
    lines.append("# Manifest count summary\n")
    lines.append(
        "This file is auto-generated from the CSV manifests under `Training_data_cleaned/`.\n"
    )
    lines.append("\n")

    for c in counts:
        lines.append(f"## {c.path}\n")
        lines.append(f"- total_rows: {c.total_rows}\n")
        lines.append(f"- splits: {c.splits}\n")
        lines.append("- label_totals:\n")
        for lb in CANON_LABELS:
            lines.append(f"  - {lb}: {c.label_totals.get(lb, 0)}\n")
        lines.append("\n")

        header = ["split"] + CANON_LABELS + ["TOTAL"]
        lines.append("Split x label:\n\n")
        lines.append("| " + " | ".join(header) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |\n")
        for sp, by_label in c.split_label.items():
            total = sum(by_label.values())
            row = [sp] + [str(by_label.get(lb, 0)) for lb in CANON_LABELS] + [str(total)]
            lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")

        lines.append("Top sources:\n")
        for src, n in c.top_sources:
            lines.append(f"- {src}: {n}\n")
        lines.append("\n")

    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize split/label/source counts for cleaned CSV manifests.")
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[
            "Training_data_cleaned/classification_manifest.csv",
            "Training_data_cleaned/classification_manifest_hq_train.csv",
            "Training_data_cleaned/classification_manifest_eval_only.csv",
            "Training_data_cleaned/test_all_sources.csv",
        ],
        help="One or more manifest CSV paths (relative to repo root)",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/manifest_counts_summary.md",
        help="Write a Markdown summary to this path",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/manifest_counts_summary.json",
        help="Write the raw counts to this path",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-k sources to include")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    counts: list[ManifestCounts] = []
    for rel in args.csv:
        p = (repo_root / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV: {p}")
        counts.append(read_counts(p, top_k_sources=args.topk))

    out_md = (repo_root / args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(counts), encoding="utf-8")

    out_json = (repo_root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(c) for c in counts], indent=2), encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
