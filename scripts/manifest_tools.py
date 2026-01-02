from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


def _iter_rows(manifest: Path) -> Iterable[dict]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {manifest}")
        for row in reader:
            if not row:
                continue
            yield row


def _norm(v: Optional[str]) -> str:
    if v is None:
        return ""
    return str(v).strip()


def cmd_stats(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    src_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    src_split_counts: Counter[tuple[str, str]] = Counter()

    total = 0
    for row in _iter_rows(manifest):
        total += 1
        src = _norm(row.get("source"))
        split = _norm(row.get("split"))
        src_counts[src] += 1
        split_counts[split] += 1
        src_split_counts[(src, split)] += 1

    print(f"manifest: {manifest}")
    print(f"rows: {total}")

    print("\nSplits:")
    for k, v in split_counts.most_common():
        print(f"  {k or '<empty>'}: {v}")

    print("\nTop sources:")
    top_n = int(args.top)
    for k, v in src_counts.most_common(top_n):
        print(f"  {k or '<empty>'}: {v}")

    if args.source_prefix:
        pref = str(args.source_prefix)
        print(f"\nSources starting with '{pref}':")
        for k, v in sorted(src_counts.items()):
            if k.startswith(pref):
                print(f"  {k}: {v}")

        print(f"\nPer-split counts for sources starting with '{pref}':")
        for (src, split), v in sorted(src_split_counts.items()):
            if src.startswith(pref):
                print(f"  {src} / {split or '<empty>'}: {v}")

    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    out = Path(args.out)
    if not manifest.exists():
        raise SystemExit(f"Not found: {manifest}")

    include_sources = [s.strip() for s in (args.include_source or []) if s.strip()]
    include_prefixes = [p.strip() for p in (args.include_source_prefix or []) if p.strip()]
    include_split = _norm(args.split)

    if not include_sources and not include_prefixes:
        raise SystemExit("Provide --include-source and/or --include-source-prefix")

    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total = 0

    with manifest.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {manifest}")
        with out.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=list(reader.fieldnames))
            writer.writeheader()
            for row in reader:
                total += 1
                src = _norm(row.get("source"))
                split = _norm(row.get("split"))

                if include_split and split != include_split:
                    continue

                ok = False
                if src in include_sources:
                    ok = True
                if not ok:
                    for p in include_prefixes:
                        if src.startswith(p):
                            ok = True
                            break
                if not ok:
                    continue

                writer.writerow(row)
                written += 1

    print(f"in:  {manifest}")
    print(f"out: {out}")
    print(f"rows_written: {written}")
    print(f"rows_scanned: {total}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming tools for big classification manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_stats = sub.add_parser("stats", help="Print split/source counts")
    ap_stats.add_argument("--manifest", type=str, required=True)
    ap_stats.add_argument("--top", type=int, default=25)
    ap_stats.add_argument("--source-prefix", type=str, default=None)
    ap_stats.set_defaults(func=cmd_stats)

    ap_filter = sub.add_parser("filter", help="Filter manifest by source and optional split")
    ap_filter.add_argument("--manifest", type=str, required=True)
    ap_filter.add_argument("--out", type=str, required=True)
    ap_filter.add_argument("--split", type=str, default=None, help="train|val|test")
    ap_filter.add_argument("--include-source", type=str, action="append", default=[])
    ap_filter.add_argument("--include-source-prefix", type=str, action="append", default=[])
    ap_filter.set_defaults(func=cmd_filter)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
