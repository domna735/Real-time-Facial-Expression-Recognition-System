from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str


@dataclass(frozen=True)
class SplitSpec:
    key: str
    manifest: str
    out_dir: str


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


def _count_manifest_rows(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return sum(1 for _ in r)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize on-disk FER2013 official PublicTest/PrivateTest evaluations for standard CE/KD/DKD checkpoints. "
            "This script does not run evaluation; it reads existing reliabilitymetrics.json artifacts."
        )
    )
    ap.add_argument("--date", type=str, default="20260209", help="Tag for output folder naming")
    ap.add_argument(
        "--protocols",
        type=str,
        default="singlecrop,tencrop",
        help="Comma-separated evaluation protocols to include (e.g., 'singlecrop,tencrop')",
    )
    args = ap.parse_args()

    protocols = [p.strip() for p in str(args.protocols).split(",") if p.strip()]
    if not protocols:
        raise SystemExit("--protocols must contain at least one protocol")

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

    splits: List[SplitSpec] = [
        SplitSpec(
            key="fer2013_publictest",
            manifest="Training_data/FER2013_official_from_csv/manifest__publictest.csv",
            out_dir="outputs/evals/students/fer2013_official__{model}__publictest__test__{date}__{protocol}",
        ),
        SplitSpec(
            key="fer2013_privatetest",
            manifest="Training_data/FER2013_official_from_csv/manifest__privatetest.csv",
            out_dir="outputs/evals/students/fer2013_official__{model}__privatetest__test__{date}__{protocol}",
        ),
    ]

    out_root = REPO_ROOT / f"outputs/benchmarks/fer2013_official_summary__{args.date}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for m in models:
        ckpt_abs = (REPO_ROOT / m.checkpoint).resolve()
        if not ckpt_abs.exists():
            raise FileNotFoundError(f"Missing checkpoint: {m.checkpoint}")

        for s in splits:
            manifest_abs = (REPO_ROOT / s.manifest).resolve()
            if not manifest_abs.exists():
                raise FileNotFoundError(f"Missing manifest: {s.manifest}")
            n = _count_manifest_rows(manifest_abs)

            for protocol in protocols:
                eval_out_dir = REPO_ROOT / s.out_dir.format(model=m.label, date=args.date, protocol=protocol)
                metrics_path = eval_out_dir / "reliabilitymetrics.json"

                # Backward-compat: historical single-crop directories had no __singlecrop suffix.
                if protocol == "singlecrop" and not metrics_path.exists():
                    legacy_dir = REPO_ROOT / s.out_dir.format(model=m.label, date=args.date, protocol="").rstrip("_")
                    legacy_metrics = legacy_dir / "reliabilitymetrics.json"
                    if legacy_metrics.exists():
                        metrics_path = legacy_metrics

                if not metrics_path.exists():
                    raise FileNotFoundError(
                        f"Missing metrics artifact: {metrics_path} (run scripts/eval_student_checkpoint.py first)"
                    )

                metrics = _load_json(metrics_path)

                rows.append(
                    {
                        "model": m.label,
                        "dataset": s.key,
                        "protocol": protocol,
                        "n": n,
                        "manifest": s.manifest,
                        "metrics": str(metrics_path.relative_to(REPO_ROOT).as_posix()),
                        "raw_acc": _metric(metrics, "raw", "accuracy"),
                        "raw_macro_f1": _metric(metrics, "raw", "macro_f1"),
                        "raw_ece": _metric(metrics, "raw", "ece"),
                        "raw_nll": _metric(metrics, "raw", "nll"),
                        "ts_ece": _ts_metric(metrics, "ece"),
                        "ts_nll": _ts_metric(metrics, "nll"),
                    }
                )

    out_csv = out_root / "fer2013_official_summary.csv"
    fieldnames = [
        "model",
        "dataset",
        "protocol",
        "n",
        "raw_acc",
        "raw_macro_f1",
        "raw_ece",
        "raw_nll",
        "ts_ece",
        "ts_nll",
        "manifest",
        "metrics",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Markdown table
    def fmt(x: Any) -> str:
        if x is None:
            return "(missing)"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    out_md = out_root / "fer2013_official_summary.md"
    lines: List[str] = []
    lines.append("# FER2013 official split summary\n")
    lines.append(f"Date tag: `{args.date}`\n")
    lines.append("\n")
    lines.append("This table is generated from on-disk `reliabilitymetrics.json` artifacts produced by `scripts/eval_student_checkpoint.py`.\n")
    lines.append("\n")
    lines.append("| Model | Split | Protocol | n | Acc | Macro-F1 | ECE | NLL | TS-ECE | TS-NLL | Metrics artifact |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")

    # Stable sort: PublicTest then PrivateTest, singlecrop then tencrop, then CE/KD/DKD
    order_ds = {"fer2013_publictest": 0, "fer2013_privatetest": 1}
    order_p = {"singlecrop": 0, "tencrop": 1}
    order_m = {m.label: i for i, m in enumerate(models)}
    for r in sorted(
        rows,
        key=lambda x: (
            order_ds.get(str(x["dataset"]), 999),
            order_p.get(str(x.get("protocol") or ""), 999),
            order_m.get(str(x["model"]), 999),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["dataset"]),
                    str(r.get("protocol") or ""),
                    str(r["n"]),
                    fmt(r["raw_acc"]),
                    fmt(r["raw_macro_f1"]),
                    fmt(r["raw_ece"]),
                    fmt(r["raw_nll"]),
                    fmt(r["ts_ece"]),
                    fmt(r["ts_nll"]),
                    str(r["metrics"]),
                ]
            )
            + " |\n"
        )

    out_md.write_text("".join(lines), encoding="utf-8")

    out_json = out_root / "fer2013_official_summary.json"
    out_json.write_text(json.dumps({"date": args.date, "rows": rows}, indent=2), encoding="utf-8")

    print(str(out_md.relative_to(REPO_ROOT).as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
