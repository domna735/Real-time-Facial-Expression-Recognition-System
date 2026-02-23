from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest: Path


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # "teacher" | "student"
    name: str
    ckpt: Path
    run_dir: Path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _find_best_student_run(group_dir: Path) -> ModelSpec:
    if not group_dir.exists():
        raise SystemExit(f"Student group dir not found: {group_dir}")

    best: Optional[Tuple[float, float, Path]] = None  # (macro_f1, acc, run_dir)

    for run_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
        metrics_path = run_dir / "reliabilitymetrics.json"
        ckpt_path = run_dir / "best.pt"
        if not metrics_path.exists() or not ckpt_path.exists():
            continue

        try:
            rel = _load_json(metrics_path)
        except Exception:
            continue

        raw = rel.get("raw") if isinstance(rel, dict) else None
        if not isinstance(raw, dict):
            continue

        macro = raw.get("macro_f1")
        acc = raw.get("accuracy")
        if not isinstance(macro, (int, float)) or not isinstance(acc, (int, float)):
            continue

        cand = (float(macro), float(acc), run_dir)
        if best is None or cand[:2] > best[:2]:
            best = cand

    if best is None:
        raise SystemExit(f"No runnable student checkpoints found under: {group_dir}")

    _macro, _acc, best_run_dir = best
    ckpt = best_run_dir / "best.pt"

    return ModelSpec(kind="student", name=best_run_dir.name, ckpt=ckpt, run_dir=best_run_dir)


def _teacher_from_run_dir(run_dir: Path) -> ModelSpec:
    if not run_dir.exists():
        raise SystemExit(f"Teacher run dir not found: {run_dir}")
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Teacher best.pt not found: {ckpt}")
    return ModelSpec(kind="teacher", name=run_dir.name, ckpt=ckpt, run_dir=run_dir)


def _ensure_relative(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.resolve().as_posix()


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def _student_eval_cmd(*, python_exe: Path, model: ModelSpec, ds: DatasetSpec, out_dir: Path, num_workers: int) -> List[str]:
    # The manifest's image_path values are relative to the cleaned dataset root.
    # For curated manifests under Training_data_cleaned/, the correct root is the manifest's parent.
    eval_data_root = ds.manifest.parent
    return [
        str(python_exe),
        str(REPO_ROOT / "scripts" / "eval_student_checkpoint.py"),
        "--checkpoint",
        str(model.ckpt),
        "--eval-manifest",
        str(ds.manifest),
        "--eval-split",
        "test",
        "--eval-data-root",
        str(eval_data_root),
        "--num-workers",
        str(int(num_workers)),
        "--out-dir",
        str(out_dir),
    ]


def _teacher_eval_cmd(*, python_exe: Path, model: ModelSpec, ds: DatasetSpec, out_dir: Path, num_workers: int) -> List[str]:
    ckpt_args = _load_torch_checkpoint_args(model.ckpt)

    def _get(name: str, default: Any) -> Any:
        v = ckpt_args.get(name, default)
        return default if v is None else v

    cmd: List[str] = [
        str(python_exe),
        str(REPO_ROOT / "scripts" / "train_teacher.py"),
        "--model",
        str(_get("model", "resnet18")),
        "--manifest",
        str(_get("manifest", "")),
        "--out-root",
        str(_get("out_root", "Training_data_cleaned")),
        "--image-size",
        str(int(_get("image_size", 224))),
        "--embed-dim",
        str(int(_get("embed_dim", 512))),
        "--batch-size",
        str(int(_get("batch_size", 64))),
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
        str(model.ckpt),
        "--evaluate-only",
        "--eval-manifest",
        str(ds.manifest),
        "--skip-env-snapshot",
    ]

    # IMPORTANT: for cross-dataset benchmarking we must NOT apply the training run's
    # include/exclude source filters, otherwise external manifests (e.g. FER2013) can
    # be filtered down to zero rows.

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


def _read_key_metrics(rel_path: Path) -> Dict[str, float]:
    d = _load_json(rel_path)
    raw = d.get("raw", {}) if isinstance(d, dict) else {}
    ts = d.get("temperature_scaled", {}) if isinstance(d, dict) else {}

    def _fg(obj: Any, k: str) -> Optional[float]:
        if isinstance(obj, dict) and isinstance(obj.get(k), (int, float)):
            return float(obj[k])
        return None

    out: Dict[str, float] = {}
    for prefix, obj in [("raw", raw), ("ts", ts)]:
        for k in ["accuracy", "macro_f1", "ece", "nll"]:
            v = _fg(obj, k)
            if v is not None:
                out[f"{prefix}.{k}"] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run offline benchmark suite (teachers + best students) on fixed test manifests.")

    ap.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable (default: repo .venv).",
    )
    ap.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (default 0 for stability on Windows).")

    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmarks",
        help="Where to write summary + index.",
    )

    ap.add_argument(
        "--teacher-run",
        type=Path,
        action="append",
        default=[],
        help="Teacher run directory under outputs/teachers/... (repeatable)",
    )
    ap.add_argument(
        "--student-group",
        type=Path,
        action="append",
        default=[],
        help="Student group directory like outputs/students/CE (repeatable). Best run is auto-selected.",
    )

    ap.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Dataset test manifest CSV (repeatable, order matters).",
    )

    args = ap.parse_args()

    python_exe: Path = args.python
    if not python_exe.exists():
        raise SystemExit(f"python not found: {python_exe}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root: Path = args.out_root / f"offline_suite__{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.manifest:
        raise SystemExit("No manifests provided. Use --manifest ... (repeatable).")

    datasets: List[DatasetSpec] = []
    for m in args.manifest:
        mm = m if m.is_absolute() else (REPO_ROOT / m)
        if not mm.exists():
            raise SystemExit(f"manifest not found: {mm}")
        datasets.append(DatasetSpec(name=mm.stem, manifest=mm))

    teachers = [_teacher_from_run_dir(REPO_ROOT / p if not p.is_absolute() else p) for p in args.teacher_run]
    students = [_find_best_student_run(REPO_ROOT / p if not p.is_absolute() else p) for p in args.student_group]

    if not teachers and not students:
        raise SystemExit("No models provided. Use --teacher-run and/or --student-group.")

    models: List[ModelSpec] = [*teachers, *students]

    index: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": _ensure_relative(out_root),
        "datasets": [{"name": d.name, "manifest": _ensure_relative(d.manifest)} for d in datasets],
        "models": [
            {"kind": m.kind, "name": m.name, "ckpt": _ensure_relative(m.ckpt), "run_dir": _ensure_relative(m.run_dir)}
            for m in models
        ],
        "results": [],
    }

    rows_md: List[str] = []
    rows_md.append("# Offline Benchmark Suite\n")
    rows_md.append(f"Generated: {index['time']}\n")
    rows_md.append("\nDatasets (order):\n")
    for d in datasets:
        rows_md.append(f"- {d.name}: `{_ensure_relative(d.manifest)}`\n")

    rows_md.append("\n## Summary (raw + temperature-scaled)\n")
    rows_md.append("model_kind | model | dataset | raw_macro_f1 | raw_acc | ts_macro_f1 | ts_acc | out_dir\n")
    rows_md.append("---|---|---:|---:|---:|---:|---:|---\n")

    for m in models:
        for ds in datasets:
            eval_out = out_root / "evals" / m.kind / m.name / ds.name
            eval_out.mkdir(parents=True, exist_ok=True)

            if m.kind == "student":
                cmd = _student_eval_cmd(python_exe=python_exe, model=m, ds=ds, out_dir=eval_out, num_workers=int(args.num_workers))
            else:
                cmd = _teacher_eval_cmd(python_exe=python_exe, model=m, ds=ds, out_dir=eval_out, num_workers=int(args.num_workers))

            print(f"\n=== {m.kind}:{m.name} on {ds.name} ===")
            _run(cmd)

            rel_path = eval_out / "reliabilitymetrics.json"
            if not rel_path.exists():
                raise SystemExit(f"Missing reliabilitymetrics.json: {rel_path}")

            km = _read_key_metrics(rel_path)
            result_item = {
                "model_kind": m.kind,
                "model": m.name,
                "dataset": ds.name,
                "out_dir": _ensure_relative(eval_out),
                "reliabilitymetrics": _ensure_relative(rel_path),
                "metrics": km,
            }
            index["results"].append(result_item)

            rows_md.append(
                f"{m.kind} | {m.name} | {ds.name} | "
                f"{km.get('raw.macro_f1', float('nan')):.6f} | {km.get('raw.accuracy', float('nan')):.6f} | "
                f"{km.get('ts.macro_f1', float('nan')):.6f} | {km.get('ts.accuracy', float('nan')):.6f} | "
                f"`{_ensure_relative(eval_out)}`\n"
            )

    (out_root / "benchmark_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (out_root / "benchmark_summary.md").write_text("".join(rows_md), encoding="utf-8")

    print("\nDONE")
    print(json.dumps({"out_root": str(out_root), "summary": str(out_root / 'benchmark_summary.md')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
