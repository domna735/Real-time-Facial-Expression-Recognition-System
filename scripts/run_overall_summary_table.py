from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    manifest: str
    split: str = "test"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(d: Dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _ts_metric(d: Dict[str, Any], key: str) -> float | None:
    """Return temperature-scaled metric.

    Some artifacts store this under `temperature_scaled` (current), while older
    ones may store it under `ts`.
    """

    v = _metric(d, "temperature_scaled", key)
    if v is not None:
        return v
    return _metric(d, "ts", key)


def _per_class_f1(d: Dict[str, Any], scope: str, label: str) -> float | None:
    cur = d.get(scope, {}).get("per_class_f1", {})
    if isinstance(cur, dict) and label in cur:
        v = cur[label]
        if isinstance(v, (int, float)):
            return float(v)
    return None


def eval_one(
    *,
    checkpoint: str,
    manifest: str,
    split: str,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> Path:
    out_path = (REPO_ROOT / out_dir).resolve()
    metrics_path = out_path / "reliabilitymetrics.json"
    if metrics_path.exists() and not force:
        return metrics_path

    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((REPO_ROOT / "scripts" / "eval_student_checkpoint.py").resolve()),
        "--checkpoint",
        checkpoint,
        "--eval-manifest",
        manifest,
        "--eval-split",
        split,
        "--eval-data-root",
        ".",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--out-dir",
        out_dir,
    ]

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics not found: {metrics_path}")
    return metrics_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run CE/KD/DKD checkpoint evaluations on key manifests and build an overall summary table. "
            "Outputs are derived from on-disk reliabilitymetrics.json artifacts."
        )
    )
    ap.add_argument("--date", type=str, default="20260208", help="Tag for output folder naming")
    ap.add_argument("--force", action="store_true", help="Re-run evals even if metrics exist")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    models: List[ModelSpec] = [
        ModelSpec(
            label="CE_20251223_225031",
            checkpoint="outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt",
        ),
        ModelSpec(
            label="KD_20251229_182119",
            checkpoint="outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/best.pt",
        ),
        ModelSpec(
            label="DKD_20251229_223722",
            checkpoint="outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/best.pt",
        ),
    ]

    datasets: List[DatasetSpec] = [
        DatasetSpec(key="eval_only", manifest="Training_data_cleaned/classification_manifest_eval_only.csv"),
        DatasetSpec(key="expw_full", manifest="Training_data_cleaned/expw_full_manifest.csv"),
        DatasetSpec(key="test_all_sources", manifest="Training_data_cleaned/test_all_sources.csv"),
        DatasetSpec(key="fer2013_folder", manifest="Training_data/fer2013_folder_manifest.csv"),
    ]

    out_root = Path(f"outputs/benchmarks/overall_summary__{args.date}")
    (REPO_ROOT / out_root).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    # Run evals (or reuse existing artifacts), then load metrics.
    for m in models:
        ckpt_abs = str((REPO_ROOT / m.checkpoint).resolve())
        if not Path(ckpt_abs).exists():
            raise FileNotFoundError(f"Missing checkpoint: {m.checkpoint}")

        for d in datasets:
            manifest_abs = str((REPO_ROOT / d.manifest).resolve())
            if not Path(manifest_abs).exists():
                raise FileNotFoundError(f"Missing manifest: {d.manifest}")

            out_dir = f"outputs/evals/students/overall__{m.label}__{d.key}__{d.split}__{args.date}"
            metrics_path = eval_one(
                checkpoint=m.checkpoint,
                manifest=d.manifest,
                split=d.split,
                out_dir=out_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                force=bool(args.force),
            )
            metrics = _load_json(metrics_path)

            rows.append(
                {
                    "model": m.label,
                    "dataset": d.key,
                    "manifest": d.manifest,
                    "out_dir": out_dir,
                    "raw_acc": _metric(metrics, "raw", "accuracy"),
                    "raw_macro_f1": _metric(metrics, "raw", "macro_f1"),
                    "raw_ece": _metric(metrics, "raw", "ece"),
                    "raw_nll": _metric(metrics, "raw", "nll"),
                    "ts_ece": _ts_metric(metrics, "ece"),
                    "ts_nll": _ts_metric(metrics, "nll"),
                    "fear_f1": _per_class_f1(metrics, "raw", "Fear"),
                    "disgust_f1": _per_class_f1(metrics, "raw", "Disgust"),
                }
            )

    # Write CSV
    out_csv = (REPO_ROOT / out_root / "overall_summary.csv").resolve()
    fieldnames = [
        "model",
        "dataset",
        "manifest",
        "raw_acc",
        "raw_macro_f1",
        "fear_f1",
        "disgust_f1",
        "raw_ece",
        "raw_nll",
        "ts_ece",
        "ts_nll",
        "out_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Write Markdown table
    def fmt(x: Any) -> str:
        if x is None:
            return "(missing)"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    # Pivot-ish: emit rows sorted by dataset then model
    order_ds = {d.key: i for i, d in enumerate(datasets)}
    order_m = {m.label: i for i, m in enumerate(models)}
    rows_sorted = sorted(rows, key=lambda r: (order_ds.get(r["dataset"], 999), order_m.get(r["model"], 999)))

    out_md = (REPO_ROOT / out_root / "overall_summary.md").resolve()
    lines: List[str] = []
    lines.append("# Overall sanity table (CE vs KD vs DKD)\n\n")
    lines.append(f"Date tag: `{args.date}`\n\n")
    lines.append("This table is generated from on-disk `reliabilitymetrics.json` artifacts produced by `scripts/eval_student_checkpoint.py`.\n\n")
    lines.append("Important interpretation:\n\n")
    lines.append("- `test_all_sources` / `eval_only` / `expw_full` are *domain-shift / mixture* gates; scores can be lower than clean single-dataset tests.\n")
    lines.append("- `fer2013_folder` is a folder split under `Training_data/FER2013` and is **not** the official FER2013 PublicTest/PrivateTest protocol.\n\n")

    header = [
        "Dataset",
        "Model",
        "Raw acc",
        "Raw macro-F1",
        "Fear F1",
        "Disgust F1",
        "TS ECE",
        "TS NLL",
        "Artifact out_dir",
    ]
    lines.append("| " + " | ".join(header) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |\n")
    for r in rows_sorted:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["dataset"],
                    r["model"],
                    fmt(r["raw_acc"]),
                    fmt(r["raw_macro_f1"]),
                    fmt(r["fear_f1"]),
                    fmt(r["disgust_f1"]),
                    fmt(r["ts_ece"]),
                    fmt(r["ts_nll"]),
                    r["out_dir"],
                ]
            )
            + " |\n"
        )

    out_md.write_text("".join(lines), encoding="utf-8")

    # Also write raw JSON for programmatic consumption
    out_json = (REPO_ROOT / out_root / "overall_summary.json").resolve()
    out_json.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
