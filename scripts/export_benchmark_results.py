from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fget(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.6f}"


@dataclass(frozen=True)
class Row:
    model_kind: str
    model: str
    dataset: str
    out_dir: str
    reliabilitymetrics: str
    raw_macro_f1: Optional[float]
    raw_acc: Optional[float]
    raw_ece: Optional[float]
    raw_nll: Optional[float]
    ts_macro_f1: Optional[float]
    ts_acc: Optional[float]
    ts_ece: Optional[float]
    ts_nll: Optional[float]


def _read_rows(index: Dict[str, Any]) -> List[Row]:
    results = index.get("results")
    if not isinstance(results, list):
        raise SystemExit("benchmark_index.json missing 'results' list")

    out: List[Row] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        out.append(
            Row(
                model_kind=str(r.get("model_kind") or ""),
                model=str(r.get("model") or ""),
                dataset=str(r.get("dataset") or ""),
                out_dir=str(r.get("out_dir") or ""),
                reliabilitymetrics=str(r.get("reliabilitymetrics") or ""),
                raw_macro_f1=_fget(metrics, "raw.macro_f1"),
                raw_acc=_fget(metrics, "raw.accuracy"),
                raw_ece=_fget(metrics, "raw.ece"),
                raw_nll=_fget(metrics, "raw.nll"),
                ts_macro_f1=_fget(metrics, "ts.macro_f1"),
                ts_acc=_fget(metrics, "ts.accuracy"),
                ts_ece=_fget(metrics, "ts.ece"),
                ts_nll=_fget(metrics, "ts.nll"),
            )
        )

    return out


def _write_all_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "model_kind",
                "model",
                "dataset",
                "raw_macro_f1",
                "raw_acc",
                "raw_ece",
                "raw_nll",
                "ts_macro_f1",
                "ts_acc",
                "ts_ece",
                "ts_nll",
                "out_dir",
                "reliabilitymetrics",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.model_kind,
                    r.model,
                    r.dataset,
                    _fmt(r.raw_macro_f1),
                    _fmt(r.raw_acc),
                    _fmt(r.raw_ece),
                    _fmt(r.raw_nll),
                    _fmt(r.ts_macro_f1),
                    _fmt(r.ts_acc),
                    _fmt(r.ts_ece),
                    _fmt(r.ts_nll),
                    r.out_dir,
                    r.reliabilitymetrics,
                ]
            )


def _best_worst_by_dataset(rows: List[Row], *, metric: str) -> Tuple[List[Row], List[Row]]:
    """Return (best_rows, worst_rows) for each (model_kind, dataset) by metric."""
    def _metric_value(r: Row) -> float:
        v = getattr(r, metric)
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def _metric_value_worst(r: Row) -> float:
        v = getattr(r, metric)
        return float(v) if isinstance(v, (int, float)) else float("inf")

    best: Dict[Tuple[str, str], Row] = {}
    worst: Dict[Tuple[str, str], Row] = {}

    for r in rows:
        key = (r.model_kind, r.dataset)

        cur_best = best.get(key)
        if cur_best is None or _metric_value(r) > _metric_value(cur_best):
            best[key] = r

        cur_worst = worst.get(key)
        if cur_worst is None or _metric_value_worst(r) < _metric_value_worst(cur_worst):
            worst[key] = r

    best_rows = sorted(best.values(), key=lambda x: (x.model_kind, x.dataset, x.model))
    worst_rows = sorted(worst.values(), key=lambda x: (x.model_kind, x.dataset, x.model))
    return best_rows, worst_rows


def _write_pick_csv(rows: List[Row], out_path: Path, *, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([title])
        w.writerow(
            [
                "model_kind",
                "dataset",
                "model",
                "raw_macro_f1",
                "raw_acc",
                "ts_macro_f1",
                "ts_acc",
                "out_dir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.model_kind,
                    r.dataset,
                    r.model,
                    _fmt(r.raw_macro_f1),
                    _fmt(r.raw_acc),
                    _fmt(r.ts_macro_f1),
                    _fmt(r.ts_acc),
                    r.out_dir,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export offline benchmark suite results to CSV and compute best/worst per dataset.")
    ap.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Path like outputs/benchmarks/offline_suite__YYYYMMDD_HHMMSS",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="raw_macro_f1",
        choices=["raw_macro_f1", "raw_acc", "ts_macro_f1", "ts_acc"],
        help="Metric used to pick best/worst per dataset.",
    )

    args = ap.parse_args()

    suite_dir: Path = args.suite_dir
    if not suite_dir.is_absolute():
        suite_dir = REPO_ROOT / suite_dir
    suite_dir = suite_dir.resolve()

    index_path = suite_dir / "benchmark_index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing benchmark_index.json: {index_path}")

    index = _load_json(index_path)
    if not isinstance(index, dict):
        raise SystemExit("benchmark_index.json is not a JSON object")

    rows = _read_rows(index)

    all_csv = suite_dir / "benchmark_results.csv"
    _write_all_csv(rows, all_csv)

    best_rows, worst_rows = _best_worst_by_dataset(rows, metric=str(args.metric))

    best_csv = suite_dir / f"benchmark_best_by_dataset__{args.metric}.csv"
    worst_csv = suite_dir / f"benchmark_worst_by_dataset__{args.metric}.csv"

    _write_pick_csv(best_rows, best_csv, title=f"Best per (model_kind,dataset) by {args.metric}")
    _write_pick_csv(worst_rows, worst_csv, title=f"Worst per (model_kind,dataset) by {args.metric}")

    print(
        json.dumps(
            {
                "suite_dir": str(suite_dir),
                "wrote": [
                    str(all_csv),
                    str(best_csv),
                    str(worst_csv),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
