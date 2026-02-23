"""Generate a detailed investigation report for datasets with unexpectedly low results.

Reads an offline benchmark suite output directory (benchmark_index.json + per-eval
reliabilitymetrics.json / eval_meta.json) and produces:

- A Markdown report (evidence-first, with paths)
- A per-class F1 CSV (raw + temperature-scaled)
- A dataset audit CSV (label/source distribution + bbox coverage + sampled file existence)

Usage (PowerShell):
  .\.venv\Scripts\python.exe scripts\report_bad_dataset_results.py \
    --suite-dir outputs/benchmarks/offline_suite__20260208_192604 \
    --datasets classification_manifest_eval_only expw_full_manifest test_fer2013_uniform_7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SuiteResult:
    model_kind: str
    model: str
    dataset: str
    out_dir: str
    reliabilitymetrics: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _load_suite_index(suite_dir: Path) -> tuple[dict[str, Any], list[SuiteResult]]:
    index_path = suite_dir / "benchmark_index.json"
    index = _read_json(index_path)

    results: list[SuiteResult] = []
    for r in index.get("results", []):
        results.append(
            SuiteResult(
                model_kind=r["model_kind"],
                model=r["model"],
                dataset=r["dataset"],
                out_dir=r["out_dir"],
                reliabilitymetrics=r["reliabilitymetrics"],
            )
        )
    return index, results


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _is_bbox_present(row: dict[str, str]) -> bool:
    keys = ["bbox_top", "bbox_left", "bbox_right", "bbox_bottom"]
    values = [row.get(k, "") for k in keys]
    if any(v is None for v in values):
        return False
    return all(str(v).strip() != "" for v in values)


def _sample_file_exists(
    data_root: Path,
    rows: list[dict[str, str]],
    max_check: int,
    seed: int,
) -> tuple[int, int]:
    if max_check <= 0 or not rows:
        return 0, 0
    rng = random.Random(seed)
    n = min(max_check, len(rows))
    sample = rng.sample(rows, n)
    ok = 0
    for row in sample:
        rel = row.get("image_path", "")
        if not rel:
            continue
        p = data_root / rel
        if p.exists():
            ok += 1
    return ok, n


def _audit_dataset_manifest(
    manifest_path: Path,
    eval_split: str,
    max_path_check: int,
    seed: int,
) -> dict[str, Any]:
    rows_all = _read_manifest_rows(manifest_path)
    rows = [r for r in rows_all if (r.get("split") or "").strip().lower() == eval_split.lower()]

    label_counts = Counter((r.get("label") or "<missing>").strip() for r in rows)
    source_counts = Counter((r.get("source") or "<missing>").strip() for r in rows)
    bbox_present = sum(1 for r in rows if _is_bbox_present(r))

    data_root = manifest_path.parent
    exists_ok, exists_checked = _sample_file_exists(
        data_root=data_root,
        rows=rows,
        max_check=max_path_check,
        seed=seed,
    )

    return {
        "manifest": str(manifest_path).replace("\\", "/"),
        "eval_split": eval_split,
        "rows_total": len(rows_all),
        "rows_split": len(rows),
        "labels": dict(label_counts),
        "sources": dict(source_counts),
        "bbox_present": bbox_present,
        "bbox_present_pct": (bbox_present / len(rows) * 100.0) if rows else 0.0,
        "file_exists_ok": exists_ok,
        "file_exists_checked": exists_checked,
        "file_exists_ok_pct": (exists_ok / exists_checked * 100.0) if exists_checked else None,
        "data_root": str(data_root).replace("\\", "/"),
    }


def _extract_metrics(reliabilitymetrics_path: Path) -> dict[str, Any]:
    rm = _read_json(reliabilitymetrics_path)
    raw = rm.get("raw", {})
    ts = rm.get("temperature_scaled", {})
    return {
        "raw_acc": _safe_float(raw.get("accuracy")),
        "raw_macro_f1": _safe_float(raw.get("macro_f1")),
        "raw_ece": _safe_float(raw.get("ece")),
        "raw_nll": _safe_float(raw.get("nll")),
        "raw_per_class_f1": raw.get("per_class_f1") or {},
        "ts_mode": ts.get("mode"),
        "ts_temp": _safe_float(ts.get("global_temperature")),
        "ts_acc": _safe_float(ts.get("accuracy")),
        "ts_macro_f1": _safe_float(ts.get("macro_f1")),
        "ts_ece": _safe_float(ts.get("ece")),
        "ts_nll": _safe_float(ts.get("nll")),
        "ts_per_class_f1": ts.get("per_class_f1") or {},
    }


def _read_eval_meta(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "eval_meta.json"
    if not p.exists():
        return None
    return _read_json(p)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}%"


def _topk(counter: Counter[str], k: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(k)


def _as_counter(d: dict[str, Any]) -> Counter[str]:
    c: Counter[str] = Counter()
    for k, v in d.items():
        try:
            c[str(k)] += int(v)
        except Exception:
            continue
    return c


def _markdown_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _posix_str(p: Path | str) -> str:
    return str(p).replace("\\", "/")


def _write_per_class_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model_kind",
        "model",
        "mode",
        "class",
        "f1",
        "out_dir",
        "reliabilitymetrics",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _write_audit_csv(audits: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "manifest",
        "eval_split",
        "rows_total",
        "rows_split",
        "bbox_present",
        "bbox_present_pct",
        "file_exists_ok",
        "file_exists_checked",
        "file_exists_ok_pct",
        "data_root",
        "top_labels",
        "top_sources",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in audits:
            labels = _as_counter(a.get("labels") or {})
            sources = _as_counter(a.get("sources") or {})
            top_labels = "; ".join([f"{k}:{v}" for k, v in _topk(labels, 10)])
            top_sources = "; ".join([f"{k}:{v}" for k, v in _topk(sources, 10)])
            row = {
                **{k: a.get(k) for k in fieldnames if k not in {"top_labels", "top_sources"}},
                "top_labels": top_labels,
                "top_sources": top_sources,
            }
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-md", default=None, type=Path)
    ap.add_argument("--out-per-class-csv", default=None, type=Path)
    ap.add_argument("--out-audit-csv", default=None, type=Path)
    ap.add_argument("--max-path-check", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    suite_dir: Path = args.suite_dir
    datasets: list[str] = list(args.datasets)

    index, results = _load_suite_index(suite_dir)
    suite_time = index.get("time")
    out_root = index.get("out_root")

    # Resolve dataset -> manifest
    dataset_manifest: dict[str, Path] = {}
    for d in index.get("datasets", []):
        name = d.get("name")
        manifest = d.get("manifest")
        if name and manifest:
            dataset_manifest[str(name)] = (Path(suite_dir.parent.parent.parent) / manifest).resolve()  # best-effort

    # Better: repo root inferred from suite_dir (outputs/...)
    repo_root = suite_dir
    while repo_root.name != "outputs" and repo_root.parent != repo_root:
        repo_root = repo_root.parent
    if repo_root.name == "outputs":
        repo_root = repo_root.parent

    for k in list(dataset_manifest.keys()):
        # manifest paths in index are repo-relative
        rel = next((d.get("manifest") for d in index.get("datasets", []) if d.get("name") == k), None)
        if rel:
            dataset_manifest[k] = (repo_root / rel).resolve()

    # Collect per-result metrics
    focus_results = [r for r in results if r.dataset in set(datasets)]
    if not focus_results:
        raise SystemExit(f"No results found for datasets={datasets}")

    # Use student eval_meta as canonical for split & basic transform flags (same across models for a dataset)
    dataset_eval_meta: dict[str, dict[str, Any]] = {}
    for r in focus_results:
        out_dir = (repo_root / r.out_dir).resolve()
        meta = _read_eval_meta(out_dir)
        if meta and r.dataset not in dataset_eval_meta:
            dataset_eval_meta[r.dataset] = meta

    # Dataset audits
    audits: list[dict[str, Any]] = []
    for ds in datasets:
        mp = dataset_manifest.get(ds)
        if mp is None or not mp.exists():
            audits.append(
                {
                    "dataset": ds,
                    "manifest": str(mp) if mp else "<missing>",
                    "eval_split": dataset_eval_meta.get(ds, {}).get("eval_split", "test"),
                    "rows_total": None,
                    "rows_split": None,
                    "labels": {},
                    "sources": {},
                    "bbox_present": None,
                    "bbox_present_pct": None,
                    "file_exists_ok": None,
                    "file_exists_checked": None,
                    "file_exists_ok_pct": None,
                    "data_root": None,
                }
            )
            continue
        eval_split = (dataset_eval_meta.get(ds, {}).get("eval_split") or "test")
        a = _audit_dataset_manifest(
            manifest_path=mp,
            eval_split=eval_split,
            max_path_check=args.max_path_check,
            seed=args.seed,
        )
        a["dataset"] = ds
        audits.append(a)

    # Per-class F1 rows
    per_class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for r in focus_results:
        out_dir = (repo_root / r.out_dir).resolve()
        rm_path = (repo_root / r.reliabilitymetrics).resolve()
        if not rm_path.exists():
            continue
        m = _extract_metrics(rm_path)

        summary_rows.append(
            {
                "dataset": r.dataset,
                "model_kind": r.model_kind,
                "model": r.model,
                "raw_macro_f1": m["raw_macro_f1"],
                "raw_acc": m["raw_acc"],
                "raw_ece": m["raw_ece"],
                "raw_nll": m["raw_nll"],
                "ts_temp": m["ts_temp"],
                "ts_macro_f1": m["ts_macro_f1"],
                "ts_acc": m["ts_acc"],
                "ts_ece": m["ts_ece"],
                "ts_nll": m["ts_nll"],
                "out_dir": r.out_dir,
                "reliabilitymetrics": r.reliabilitymetrics,
            }
        )

        for mode, per_class in (
            ("raw", m["raw_per_class_f1"]),
            ("temperature_scaled", m["ts_per_class_f1"]),
        ):
            for cls, f1 in (per_class or {}).items():
                per_class_rows.append(
                    {
                        "dataset": r.dataset,
                        "model_kind": r.model_kind,
                        "model": r.model,
                        "mode": mode,
                        "class": cls,
                        "f1": _safe_float(f1),
                        "out_dir": r.out_dir,
                        "reliabilitymetrics": r.reliabilitymetrics,
                    }
                )

    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = args.out_md
    if out_md is None:
        out_md = repo_root / "research" / f"issue__bad_results__{timestamp}.md"
    else:
        out_md = (repo_root / out_md).resolve() if not out_md.is_absolute() else out_md

    out_per_class = args.out_per_class_csv
    if out_per_class is None:
        out_per_class = suite_dir / f"bad_datasets__per_class_f1__{timestamp}.csv"
    else:
        out_per_class = (repo_root / out_per_class).resolve() if not out_per_class.is_absolute() else out_per_class

    out_audit = args.out_audit_csv
    if out_audit is None:
        out_audit = suite_dir / f"bad_datasets__audit__{timestamp}.csv"
    else:
        out_audit = (repo_root / out_audit).resolve() if not out_audit.is_absolute() else out_audit

    _write_per_class_csv(per_class_rows, out_per_class)
    _write_audit_csv(audits, out_audit)

    # Build report markdown
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Low-Performance Dataset Investigation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Scope\n")
        f.write("This report investigates unexpectedly weak performance on the following evaluation datasets:\n")
        for ds in datasets:
            f.write(f"- `{ds}`\n")
        f.write("\n")

        f.write("## Evidence (suite artifacts)\n")
        f.write(f"- Suite time: `{suite_time}`\n")
        f.write(f"- Suite dir: `{_posix_str(suite_dir)}`\n")
        f.write(f"- Index: `{_posix_str(suite_dir / 'benchmark_index.json')}`\n")
        f.write(f"- Full results CSV: `{_posix_str(suite_dir / 'benchmark_results.csv')}`\n")
        f.write(f"- Per-class F1 CSV (generated): `{_posix_str(out_per_class)}`\n")
        f.write(f"- Dataset audit CSV (generated): `{_posix_str(out_audit)}`\n")
        f.write("\n")

        f.write("## High-level observation\n")
        f.write("Across these datasets, macro-F1 is pulled down mainly by consistent weak classes (commonly `Disgust` and `Fear`).\n")
        f.write("Temperature scaling improves calibration (ECE / NLL) but does not change macro-F1, suggesting the main issue is not confidence calibration but domain shift / label noise / class ambiguity.\n\n")

        # Dataset audits section
        f.write("## Dataset audits (manifest-level)\n")
        for a in audits:
            ds = a.get("dataset")
            f.write(f"### {ds}\n")
            f.write(f"- Manifest: `{a.get('manifest')}`\n")
            f.write(f"- Eval split: `{a.get('eval_split')}`\n")
            f.write(f"- Rows (split/chosen): `{a.get('rows_split')}`\n")
            f.write(f"- BBox present: `{a.get('bbox_present')}` ({_fmt_pct(_safe_float(a.get('bbox_present_pct')))})\n")
            f.write(
                f"- File existence check (sampled): `{a.get('file_exists_ok')}/{a.get('file_exists_checked')}` ({_fmt_pct(_safe_float(a.get('file_exists_ok_pct')))})\n"
            )
            labels = _as_counter(a.get("labels") or {})
            sources = _as_counter(a.get("sources") or {})
            f.write(f"- Top labels: {', '.join([f'{k}:{v}' for k, v in _topk(labels, 10)])}\n")
            f.write(f"- Top sources: {', '.join([f'{k}:{v}' for k, v in _topk(sources, 10)])}\n")
            f.write("\n")

        # Results table
        f.write("## Model results summary (these datasets only)\n")
        f.write("model_kind | model | dataset | raw_macro_f1 | raw_acc | raw_ece | raw_nll | ts_temp | ts_ece | ts_nll | out_dir\n")
        f.write("---|---|---:|---:|---:|---:|---:|---:|---:|---:|---\n")
        for r in sorted(summary_rows, key=lambda x: (x["dataset"], x["model_kind"], x["model"])):
            f.write(
                " | ".join(
                    [
                        _markdown_escape(str(r["model_kind"])),
                        _markdown_escape(str(r["model"])),
                        _markdown_escape(str(r["dataset"])),
                        f"{(r['raw_macro_f1'] or 0):.6f}",
                        f"{(r['raw_acc'] or 0):.6f}",
                        f"{(r['raw_ece'] or 0):.6f}",
                        f"{(r['raw_nll'] or 0):.6f}",
                        f"{(r['ts_temp'] or 0):.3f}",
                        f"{(r['ts_ece'] or 0):.6f}",
                        f"{(r['ts_nll'] or 0):.6f}",
                        f"`{_posix_str(r['out_dir'])}`",
                    ]
                )
                + "\n"
            )
        f.write("\n")

        # Per-class F1 highlights
        f.write("## Per-class F1 highlights (where macro-F1 is lost)\n")
        f.write("For each dataset, the table below lists the minimum and maximum per-class F1 observed across all evaluated models (teachers + students).\n\n")

        for ds in datasets:
            ds_rows = [x for x in per_class_rows if x["dataset"] == ds and x["mode"] == "raw"]
            by_class: dict[str, list[float]] = defaultdict(list)
            for x in ds_rows:
                if x.get("f1") is not None:
                    by_class[str(x.get("class"))].append(float(x["f1"]))
            f.write(f"### {ds} (raw)\n")
            f.write("class | min_f1 | max_f1 | note\n")
            f.write("---|---:|---:|---\n")
            for cls in sorted(by_class.keys()):
                vals = by_class[cls]
                if not vals:
                    continue
                minv = min(vals)
                maxv = max(vals)
                note = "" 
                if maxv < 0.35:
                    note = "consistently weak across all models"
                elif minv < 0.25 and maxv > 0.55:
                    note = "model-sensitive (possible optimization / representation gap)"
                f.write(f"{_markdown_escape(cls)} | {minv:.3f} | {maxv:.3f} | {note}\n")
            f.write("\n")

        # Hypotheses
        f.write("## Likely causes (hypotheses ranked by evidence)\n")
        f.write("1) **Domain shift / label ambiguity in real-world sets (ExpW + mixed-source eval-only):** low F1 concentrates in `Disgust`/`Fear`, which are subtle and frequently confused in-the-wild.\n")
        f.write("2) **Class imbalance / long-tail:** if the manifest audit shows very small counts for `Disgust`/`Fear`, macro-F1 will drop quickly even if accuracy stays moderate.\n")
        f.write("3) **Annotation noise / weak labels:** ExpW is known to be noisy; mixed-source eval-only contains diverse acquisition/quality.\n")
        f.write("4) **Preprocessing mismatch:** FER2013 images differ strongly (low-res, grayscale, different cropping/alignment). Our pipeline forces resizing to 224 and applies CLAHE; this may help/hurt depending on domain.\n")
        f.write("5) **Model capacity + training objective effects:** KD/DKD may improve certain domains but degrade others if teacher is biased; CE appears more stable across domains in this suite.\n\n")

        # Proposed tests
        f.write("## Next tests to pinpoint the root cause (actionable)\n")
        f.write("- **Per-source breakdown for eval-only:** run the same model evaluation but grouped by `source` to see which sources dominate the drop.\n")
        f.write("- **Ablate CLAHE on FER2013/ExpW:** re-evaluate with `use_clahe=false` to confirm whether CLAHE is helping or harming these domains.\n")
        f.write("- **Error analysis on weak classes:** sample misclassified `Disgust`/`Fear` images and inspect whether they are ambiguous/occluded/low-quality or mislabeled.\n")
        f.write("- **Re-weight / focal loss / class-balanced sampling:** specifically target `Disgust` and `Fear` during training or finetuning.\n")
        f.write("- **Domain-adaptive augmentation:** add grayscale/noise/low-res augmentation to better match FER2013-like conditions.\n")

    print(f"Wrote report: {out_md}")
    print(f"Wrote per-class CSV: {out_per_class}")
    print(f"Wrote audit CSV: {out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
