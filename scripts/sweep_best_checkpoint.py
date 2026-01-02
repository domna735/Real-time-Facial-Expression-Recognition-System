from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_teacher.py"


@dataclass(frozen=True)
class SweepResult:
    ckpt_path: Path
    epoch: int
    macro_f1_raw: float
    acc_raw: float
    nll_raw: float
    ece_raw: float
    macro_f1_t: float
    acc_t: float
    nll_t: float
    ece_t: float
    t_star: float


def _load_checkpoint_any(path: Path) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    last = run_dir / "checkpoint_last.pt"
    best = run_dir / "best.pt"
    if last.exists():
        ckpts.append(last)
    if best.exists() and best not in ckpts:
        ckpts.append(best)
    # De-dup while preserving order
    seen = set()
    out: List[Path] = []
    for p in ckpts:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _args_from_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    args = ckpt.get("args") or {}
    if not isinstance(args, dict):
        return {}
    return args


def _build_eval_command(
    *,
    python_exe: Path,
    ckpt_path: Path,
    ckpt_args: Dict[str, Any],
    eval_out_dir: Path,
    eval_manifest: Optional[Path],
    fixed_temperature: Optional[float],
    num_workers: int,
) -> List[str]:
    # We re-run train_teacher.py in --evaluate-only mode, loading the weights from this checkpoint.
    # This uses the same transforms + split logic as training.
    def _get(name: str, default: Any) -> Any:
        v = ckpt_args.get(name, default)
        return default if v is None else v

    cmd: List[str] = [
        str(python_exe),
        str(TRAIN_SCRIPT),
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
        str(eval_out_dir),
        "--resume",
        str(ckpt_path),
        "--evaluate-only",
    ]

    include_sources = str(_get("include_sources", ""))
    exclude_sources = str(_get("exclude_sources", ""))
    if include_sources:
        cmd.extend(["--include-sources", include_sources])
    if exclude_sources:
        cmd.extend(["--exclude-sources", exclude_sources])

    # Keep CLAHE behavior consistent.
    if bool(_get("clahe", False)):
        cmd.append("--clahe")
        cmd.extend(["--clahe-clip", str(float(_get("clahe_clip", 2.0)))])
        cmd.extend(["--clahe-tile", str(int(_get("clahe_tile", 8)))])

    # Optional fixed temperature for report-style evaluation.
    if fixed_temperature is not None:
        cmd.extend(["--fixed-temperature", str(float(fixed_temperature))])

    if eval_manifest is not None:
        cmd.extend(["--eval-manifest", str(eval_manifest)])

    return cmd


def _read_reliability(path: Path) -> SweepResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    ts = data["temperature_scaled"]
    return SweepResult(
        ckpt_path=Path(""),
        epoch=int(data.get("epoch", -1)),
        macro_f1_raw=float(raw["macro_f1"]),
        acc_raw=float(raw["accuracy"]),
        nll_raw=float(raw["nll"]),
        ece_raw=float(raw["ece"]),
        macro_f1_t=float(ts["macro_f1"]),
        acc_t=float(ts["accuracy"]),
        nll_t=float(ts["nll"]),
        ece_t=float(ts["ece"]),
        t_star=float(ts.get("global_temperature", 1.0) or 1.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate every saved checkpoint in a teacher run directory and report which checkpoint is truly best. "
            "This helps when training used --eval-every 10 so best.pt may have missed an in-between best epoch."
        )
    )
    ap.add_argument("--run-dir", type=Path, required=True, help="Path like outputs/teachers/B3_*_stageA_img224")
    ap.add_argument(
        "--python",
        type=Path,
        default=REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        help="Python executable to use (default: repo .venv).",
    )
    ap.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional manifest to evaluate on (e.g. Training_data_cleaned/classification_manifest_eval_only.csv).",
    )
    ap.add_argument(
        "--fixed-temperature",
        type=float,
        default=None,
        help="Optional fixed temperature (global) to match report-style evaluation (e.g., 1.2).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        choices=["macro_f1_raw", "macro_f1_t"],
        default="macro_f1_raw",
        help="Which metric to maximize when selecting best checkpoint.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Num dataloader workers for evaluation (default 0 for stability on Windows).",
    )
    ap.add_argument(
        "--write-best",
        action="store_true",
        help="If set, copy the best checkpoint to <run-dir>/best.pt (backing up existing best.pt).",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write a CSV of all checkpoint scores.",
    )

    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    if not args.python.exists():
        raise SystemExit(f"python not found: {args.python}")

    ckpts = _find_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under: {run_dir}")

    sweep_root = run_dir / "_eval_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[Path, SweepResult]] = []

    for ckpt_path in ckpts:
        ckpt = _load_checkpoint_any(ckpt_path)
        ckpt_args = _args_from_checkpoint(ckpt)

        out_dir = sweep_root / ckpt_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_eval_command(
            python_exe=args.python,
            ckpt_path=ckpt_path,
            ckpt_args=ckpt_args,
            eval_out_dir=out_dir,
            eval_manifest=args.eval_manifest,
            fixed_temperature=args.fixed_temperature,
            num_workers=int(args.num_workers),
        )

        print(f"\n=== Evaluating: {ckpt_path.name} ===")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        if p.returncode != 0:
            print(p.stdout)
            raise SystemExit(f"Evaluation failed for {ckpt_path}: exit={p.returncode}")

        rel_path = out_dir / "reliabilitymetrics.json"
        if not rel_path.exists():
            print(p.stdout)
            raise SystemExit(f"Expected reliabilitymetrics.json not found: {rel_path}")

        r = _read_reliability(rel_path)
        # patch back the true ckpt path
        r = SweepResult(
            ckpt_path=ckpt_path,
            epoch=r.epoch,
            macro_f1_raw=r.macro_f1_raw,
            acc_raw=r.acc_raw,
            nll_raw=r.nll_raw,
            ece_raw=r.ece_raw,
            macro_f1_t=r.macro_f1_t,
            acc_t=r.acc_t,
            nll_t=r.nll_t,
            ece_t=r.ece_t,
            t_star=r.t_star,
        )

        score = getattr(r, args.metric)
        print(
            f"done in {dt:.1f}s | epoch={r.epoch:>3d} | "
            f"macro_f1_raw={r.macro_f1_raw:.4f} | macro_f1_t={r.macro_f1_t:.4f} | T={r.t_star:.3f} | score={score:.4f}"
        )
        results.append((ckpt_path, r))

    # Select best
    def _score(item: Tuple[Path, SweepResult]) -> float:
        return float(getattr(item[1], args.metric))

    results_sorted = sorted(results, key=_score, reverse=True)
    best_ckpt, best_r = results_sorted[0]

    print("\n=== Best checkpoint ===")
    print(f"metric={args.metric}")
    print(f"checkpoint={best_ckpt}")
    print(f"epoch={best_r.epoch}")
    print(f"macro_f1_raw={best_r.macro_f1_raw:.6f} acc_raw={best_r.acc_raw:.6f}")
    print(f"macro_f1_t  ={best_r.macro_f1_t:.6f} acc_t  ={best_r.acc_t:.6f} (T={best_r.t_star:.3f})")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ckpt",
                    "epoch",
                    "macro_f1_raw",
                    "acc_raw",
                    "nll_raw",
                    "ece_raw",
                    "macro_f1_t",
                    "acc_t",
                    "nll_t",
                    "ece_t",
                    "t_star",
                ]
            )
            for ckpt_path, r in results_sorted:
                w.writerow(
                    [
                        str(ckpt_path),
                        r.epoch,
                        r.macro_f1_raw,
                        r.acc_raw,
                        r.nll_raw,
                        r.ece_raw,
                        r.macro_f1_t,
                        r.acc_t,
                        r.nll_t,
                        r.ece_t,
                        r.t_star,
                    ]
                )
        print(f"Wrote CSV: {args.csv_out}")

    if args.write_best:
        dst = run_dir / "best.pt"
        if dst.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            bak = run_dir / f"best.pt.bak_{stamp}"
            dst.replace(bak)
            print(f"Backed up existing best.pt -> {bak.name}")

        # Copy bytes
        dst.write_bytes(best_ckpt.read_bytes())
        print(f"Wrote best.pt from: {best_ckpt.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
