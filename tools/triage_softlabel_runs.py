from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    macro_f1: float
    accuracy: float
    nll: Optional[float]
    ece: Optional[float]


def _fget(m: Dict[str, object], key: str) -> Optional[float]:
    v = m.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_runs(root: Path) -> List[Run]:
    runs: List[Run] = []
    for metrics_path in root.rglob("ensemble_metrics.json"):
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        macro_f1 = _fget(m, "macro_f1")
        accuracy = _fget(m, "accuracy")
        runs.append(
            Run(
                name=metrics_path.parent.name,
                path=metrics_path.parent,
                macro_f1=-1.0 if macro_f1 is None else macro_f1,
                accuracy=-1.0 if accuracy is None else accuracy,
                nll=_fget(m, "nll"),
                ece=_fget(m, "ece"),
            )
        )

    return runs


def guess_group(name: str) -> str:
    n = name.lower()
    # Order matters: more specific first.
    if "test_all_sources" in n:
        return "test_all_sources"
    if "rafdb" in n and "test" in n:
        return "rafdb_test"
    if "rafdb_basic" in n:
        return "rafdb_basic"
    if "rafdb_compound" in n:
        return "rafdb_compound"
    if "rafml_argmax" in n:
        return "rafml_argmax"
    if "ferplus" in n:
        return "ferplus"
    if "fer2013" in n:
        return "fer2013_uniform_7"
    if "expw_full" in n:
        return "expw_full"
    if "affectnet_full_balanced" in n:
        return "affectnet_full_balanced"
    if "fulltest" in n:
        return "fulltest"
    if "smoke" in n:
        return "smoke"
    if "uncleaned" in n:
        return "uncleaned"
    return "other"


def sort_key(r: Run) -> Tuple[float, float]:
    return (r.macro_f1, r.accuracy)


def fmt_row(r: Run) -> str:
    ece = "nan" if r.ece is None else f"{r.ece:.6f}"
    nll = "nan" if r.nll is None else f"{r.nll:.3f}"
    return f"{r.name} | {r.macro_f1:.6f} | {r.accuracy:.6f} | {ece} | {nll} | {r.path.as_posix()}"


def section(title: str) -> str:
    return f"\n## {title}\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/softlabels/**/ensemble_metrics.json and write a triage report "
            "(top runs + bottom/bad runs) for quick cleanup decisions."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs") / "softlabels",
        help="Root folder to scan (default: outputs/softlabels)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "softlabels" / "_ensemble_triage.md",
        help="Markdown output path (default: outputs/softlabels/_ensemble_triage.md)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top runs to show per group (default: 10)",
    )
    parser.add_argument(
        "--bottom",
        type=int,
        default=10,
        help="How many bottom runs to show per group (default: 10)",
    )
    parser.add_argument(
        "--exclude-smoke",
        action="store_true",
        help="Exclude runs whose name contains 'smoke'",
    )
    parser.add_argument(
        "--exclude-uncleaned",
        action="store_true",
        help="Exclude runs whose name contains 'uncleaned'",
    )
    parser.add_argument(
        "--bad-out",
        type=Path,
        default=Path("outputs") / "softlabels" / "_ensemble_bad_list.txt",
        help=(
            "Write a plain-text list of bottom runs (candidates to ignore/archive). "
            "Default: outputs/softlabels/_ensemble_bad_list.txt"
        ),
    )
    parser.add_argument(
        "--write-bad-out",
        action="store_true",
        help="Actually write --bad-out (off by default).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs = load_runs(args.root)

    def keep(r: Run) -> bool:
        n = r.name.lower()
        if args.exclude_smoke and "smoke" in n:
            return False
        if args.exclude_uncleaned and "uncleaned" in n:
            return False
        return True

    runs = [r for r in runs if keep(r)]

    groups: Dict[str, List[Run]] = {}
    for r in runs:
        g = guess_group(r.name)
        groups.setdefault(g, []).append(r)

    # Sort each group once.
    for g in list(groups.keys()):
        groups[g].sort(key=sort_key, reverse=True)

    md: List[str] = []
    md.append("# Ensemble Runs Triage\n")
    md.append("Generated by tools/triage_softlabel_runs.py\n")
    md.append(f"Total runs scanned: {len(runs)}\n")
    md.append("\nColumns: run | macro_f1 | acc | ece | nll | path\n")

    # Recommended keep-list: best run in each non-trivial group.
    md.append(section("Recommended (best per group)"))
    md.append("run | macro_f1 | acc | ece | nll | path\n")
    md.append("--- | ---: | ---: | ---: | ---: | ---\n")
    for g in sorted(groups.keys()):
        if g in {"smoke", "other"}:
            continue
        xs = groups[g]
        if not xs:
            continue
        md.append(fmt_row(xs[0]) + "\n")

    # Per-group top/bottom.
    bad_lines: List[str] = []
    for g in sorted(groups.keys()):
        xs = groups[g]
        if not xs:
            continue

        md.append(section(f"Group: {g} (n={len(xs)})"))

        md.append("### Top\n")
        md.append("run | macro_f1 | acc | ece | nll | path\n")
        md.append("--- | ---: | ---: | ---: | ---: | ---\n")
        for r in xs[: args.top]:
            md.append(fmt_row(r) + "\n")

        md.append("\n### Bottom (candidates to ignore/archive)\n")
        md.append("run | macro_f1 | acc | ece | nll | path\n")
        md.append("--- | ---: | ---: | ---: | ---: | ---\n")
        bottom = list(reversed(xs))[: args.bottom]
        for r in bottom:
            md.append(fmt_row(r) + "\n")
            bad_lines.append(f"{g}\t{r.macro_f1:.6f}\t{r.accuracy:.6f}\t{r.path.as_posix()}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(md), encoding="utf-8")
    print(f"WROTE\t{args.out}")

    if args.write_bad_out:
        args.bad_out.parent.mkdir(parents=True, exist_ok=True)
        # Sort for stable output (group then lowest macro_f1)
        args.bad_out.write_text("".join(sorted(bad_lines)), encoding="utf-8")
        print(f"WROTE\t{args.bad_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
