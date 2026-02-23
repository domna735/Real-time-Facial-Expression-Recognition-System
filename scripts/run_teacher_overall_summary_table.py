from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TeacherSpec:
    label: str
    run_dir: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    manifest: str


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


def _load_torch_checkpoint_args(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "PyTorch (torch) is required to read checkpoint args. "
            "Run this script with the project's virtualenv Python (e.g. .venv/Scripts/python.exe)."
        ) from e

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    return args if isinstance(args, dict) else {}


def _teacher_eval_cmd(
    *,
    teacher_ckpt: Path,
    eval_manifest: Path,
    out_dir: Path,
    num_workers: int,
    batch_size: int,
) -> List[str]:
    ckpt_args = _load_torch_checkpoint_args(teacher_ckpt)

    def _get(name: str, default: Any) -> Any:
        v = ckpt_args.get(name, default)
        return default if v is None else v

    # IMPORTANT:
    # - For cross-dataset benchmarking we must NOT apply the training run's include/exclude filters.
    # - For curated manifests under Training_data_cleaned/, the correct out-root for relative paths
    #   is the manifest's parent directory.
    eval_out_root = eval_manifest.parent

    cmd: List[str] = [
        sys.executable,
        str((REPO_ROOT / "scripts" / "train_teacher.py").resolve()),
        "--model",
        str(_get("model", "resnet18")),
        "--manifest",
        str(_get("manifest", str(REPO_ROOT / "Training_data_cleaned" / "classification_manifest.csv"))),
        "--out-root",
        str(eval_out_root),
        "--image-size",
        str(int(_get("image_size", 224))),
        "--embed-dim",
        str(int(_get("embed_dim", 512))),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--seed",
        str(int(_get("seed", 1337))),
        "--val-fraction",
        str(float(_get("val_fraction", 0.05))),
        "--min-per-class",
        str(int(_get("min_per_class", 2))),
        "--cb-beta",
        str(float(_get("cb_beta", 0.9999))),
        "--temperature-scaling",
        str(_get("temperature_scaling", "global")),
        "--eval-every",
        "1",
        "--checkpoint-every",
        "0",
        "--output-dir",
        str(out_dir),
        "--resume",
        str(teacher_ckpt),
        "--evaluate-only",
        "--eval-manifest",
        str(eval_manifest),
        "--skip-env-snapshot",
        "--no-pretrained",
    ]

    if bool(_get("clahe", False)):
        cmd.append("--clahe")
        cmd.extend(["--clahe-clip", str(float(_get("clahe_clip", 2.0)))])
        cmd.extend(["--clahe-tile", str(int(_get("clahe_tile", 8)))])

    fixed_temperature = _get("fixed_temperature", None)
    if fixed_temperature is not None:
        try:
            cmd.extend(["--fixed-temperature", str(float(fixed_temperature))])
        except Exception:
            pass

    return cmd


def eval_one(
    *,
    teacher_run_dir: Path,
    manifest: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> Path:
    metrics_path = out_dir / "reliabilitymetrics.json"
    if metrics_path.exists() and not force:
        return metrics_path

    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = teacher_run_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt}")

    cmd = _teacher_eval_cmd(
        teacher_ckpt=ckpt,
        eval_manifest=manifest,
        out_dir=out_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics not found: {metrics_path}")
    return metrics_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate teacher checkpoints on hard/mixed-domain gates (eval_only/expw_full/test_all_sources) "
            "and build a teacher-only summary table. Outputs are derived from on-disk reliabilitymetrics.json artifacts."
        )
    )
    ap.add_argument("--date", type=str, default="20260209", help="Tag for output folder naming")
    ap.add_argument("--force", action="store_true", help="Re-run evals even if metrics exist")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--teacher-run",
        type=str,
        action="append",
        default=[],
        help=(
            "Teacher run directory under outputs/teachers/... (repeatable). "
            "If omitted, uses the 3 Stage-A img224 teachers listed in the final report."
        ),
    )
    args = ap.parse_args()

    default_teachers: List[TeacherSpec] = [
        TeacherSpec(label="RN18_stageA_img224", run_dir="outputs/teachers/RN18_resnet18_seed1337_stageA_img224"),
        TeacherSpec(label="B3_stageA_img224", run_dir="outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224"),
        TeacherSpec(label="CNXT_stageA_img224", run_dir="outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224"),
    ]

    teachers: List[TeacherSpec]
    if args.teacher_run:
        teachers = [TeacherSpec(label=Path(p).name, run_dir=p) for p in args.teacher_run]
    else:
        teachers = default_teachers

    datasets: List[DatasetSpec] = [
        DatasetSpec(key="eval_only", manifest="Training_data_cleaned/classification_manifest_eval_only.csv"),
        DatasetSpec(key="expw_full", manifest="Training_data_cleaned/expw_full_manifest.csv"),
        DatasetSpec(key="test_all_sources", manifest="Training_data_cleaned/test_all_sources.csv"),
    ]

    out_root = Path(f"outputs/benchmarks/teacher_overall_summary__{args.date}")
    (REPO_ROOT / out_root).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for t in teachers:
        run_dir = (REPO_ROOT / t.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing teacher run dir: {t.run_dir}")

        for d in datasets:
            manifest_path = (REPO_ROOT / d.manifest).resolve()
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {d.manifest}")

            out_dir = (REPO_ROOT / f"outputs/evals/teachers/overall__{t.label}__{d.key}__test__{args.date}").resolve()
            metrics_path = eval_one(
                teacher_run_dir=run_dir,
                manifest=manifest_path,
                out_dir=out_dir,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                force=bool(args.force),
            )
            metrics = _load_json(metrics_path)

            rows.append(
                {
                    "teacher": t.label,
                    "dataset": d.key,
                    "manifest": d.manifest,
                    "eval_rows": metrics.get("eval_rows"),
                    "out_dir": str(out_dir.relative_to(REPO_ROOT).as_posix()),
                    "raw_acc": _metric(metrics, "raw", "accuracy"),
                    "raw_macro_f1": _metric(metrics, "raw", "macro_f1"),
                    "fear_f1": _per_class_f1(metrics, "raw", "Fear"),
                    "disgust_f1": _per_class_f1(metrics, "raw", "Disgust"),
                    "raw_ece": _metric(metrics, "raw", "ece"),
                    "raw_nll": _metric(metrics, "raw", "nll"),
                    "ts_ece": _ts_metric(metrics, "ece"),
                    "ts_nll": _ts_metric(metrics, "nll"),
                }
            )

    # Write CSV
    out_csv = (REPO_ROOT / out_root / "teacher_overall_summary.csv").resolve()
    fieldnames = [
        "teacher",
        "dataset",
        "manifest",
        "eval_rows",
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

    def fmt(x: Any) -> str:
        if x is None:
            return "(missing)"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    order_ds = {d.key: i for i, d in enumerate(datasets)}
    order_t = {t.label: i for i, t in enumerate(teachers)}
    rows_sorted = sorted(rows, key=lambda r: (order_ds.get(r["dataset"], 999), order_t.get(r["teacher"], 999)))

    out_md = (REPO_ROOT / out_root / "teacher_overall_summary.md").resolve()
    lines: List[str] = []
    lines.append("# Teacher hard-gate summary (Stage-A teachers on eval_only/expw_full/test_all_sources)\n\n")
    lines.append(f"Date tag: `{args.date}`\n\n")
    lines.append("This table is generated from on-disk `reliabilitymetrics.json` artifacts produced by `scripts/train_teacher.py --evaluate-only`.\n\n")
    lines.append("Interpretation: these are **hard/mixed-domain gates**; scores can be much lower than the Stage-A in-distribution validation split.\n\n")

    header = [
        "Dataset",
        "Teacher",
        "Eval rows",
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
                    r["teacher"],
                    fmt(r.get("eval_rows")),
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

    out_json = (REPO_ROOT / out_root / "teacher_overall_summary.json").resolve()
    out_json.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
