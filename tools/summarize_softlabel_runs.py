from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    macro_f1: float
    accuracy: float
    nll: float | None
    ece: float | None


def load_runs(root: Path) -> List[Run]:
    runs: List[Run] = []
    for metrics_path in root.rglob("ensemble_metrics.json"):
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        def fget(key: str) -> float | None:
            v = m.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        macro_f1 = fget("macro_f1")
        accuracy = fget("accuracy")

        runs.append(
            Run(
                name=metrics_path.parent.name,
                path=metrics_path.parent,
                macro_f1=-1.0 if macro_f1 is None else macro_f1,
                accuracy=-1.0 if accuracy is None else accuracy,
                nll=fget("nll"),
                ece=fget("ece"),
            )
        )

    return runs


def top(runs: Iterable[Run], pred: Callable[[Run], bool], k: int = 20) -> List[Run]:
    xs = [r for r in runs if pred(r)]
    xs.sort(key=lambda r: (r.macro_f1, r.accuracy), reverse=True)
    return xs[:k]


def fmt(r: Run) -> str:
    ece = "nan" if r.ece is None else f"{r.ece:.6f}"
    nll = "nan" if r.nll is None else f"{r.nll:.3f}"
    return f"{r.name}\tmacro_f1={r.macro_f1:.6f}\tacc={r.accuracy:.6f}\tece={ece}\tnll={nll}"


def main() -> int:
    root = Path("outputs") / "softlabels"
    runs = load_runs(root)
    print(f"TOTAL_RUNS\t{len(runs)}")

    print("\nTOP_FULLTEST")
    for r in top(
        runs,
        lambda x: ("fulltest" in x.name.lower()) and ("smoke" not in x.name.lower()),
        k=25,
    ):
        print(fmt(r))

    print("\nTOP_RAFDB_TEST")
    for r in top(
        runs,
        lambda x: ("rafdb" in x.name.lower())
        and ("test" in x.name.lower())
        and ("uncleaned" not in x.name.lower()),
        k=25,
    ):
        print(fmt(r))

    print("\nTOP_AFFECTNET_FULL_BALANCED")
    for r in top(runs, lambda x: "affectnet_full_balanced" in x.name.lower(), k=20):
        print(fmt(r))

    print("\nTOP_EXPW_FULL")
    for r in top(runs, lambda x: "expw_full" in x.name.lower(), k=20):
        print(fmt(r))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
