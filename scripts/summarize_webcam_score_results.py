"""Summarize webcam run score artifacts into a single Markdown table.

Inputs:
- demo/outputs/*/score_results.json

Outputs:
- outputs/domain_shift/webcam_summary.md

This script is intentionally minimal and evidence-oriented: it only reports
numbers already stored in the per-run score_results.json artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_OUTPUTS_DIR = REPO_ROOT / "demo" / "outputs"
OUT_MD = REPO_ROOT / "outputs" / "domain_shift" / "webcam_summary.md"


@dataclass(frozen=True)
class Row:
    run_id: str
    score_path: str
    scored_frames: int
    raw_acc: float
    raw_macro_f1_present: float
    raw_minority_f1_lowest3: float
    sm_acc: float
    sm_macro_f1_present: float
    sm_minority_f1_lowest3: float
    jitter_flips_per_min: float


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _safe_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    raise TypeError(f"Expected number, got {type(x)}")


def load_row(score_json_path: Path) -> Row:
    data = json.loads(score_json_path.read_text(encoding="utf-8"))

    run_id = score_json_path.parent.name

    scored_frames = int(_safe_float(_get(data, "metrics.scored_frames_for_f1")))

    raw_acc = _safe_float(_get(data, "metrics.raw.accuracy"))
    raw_macro_f1_present = _safe_float(_get(data, "metrics.raw.macro_f1_present"))
    raw_minority_f1_lowest3 = _safe_float(_get(data, "metrics.raw.minority_f1_lowest3"))

    sm_acc = _safe_float(_get(data, "metrics.smoothed.accuracy"))
    sm_macro_f1_present = _safe_float(_get(data, "metrics.smoothed.macro_f1_present"))
    sm_minority_f1_lowest3 = _safe_float(
        _get(data, "metrics.smoothed.minority_f1_lowest3")
    )

    jitter = _safe_float(_get(data, "jitter_flips_per_min"))

    score_path_rel = score_json_path.relative_to(REPO_ROOT).as_posix()

    return Row(
        run_id=run_id,
        score_path=score_path_rel,
        scored_frames=scored_frames,
        raw_acc=raw_acc,
        raw_macro_f1_present=raw_macro_f1_present,
        raw_minority_f1_lowest3=raw_minority_f1_lowest3,
        sm_acc=sm_acc,
        sm_macro_f1_present=sm_macro_f1_present,
        sm_minority_f1_lowest3=sm_minority_f1_lowest3,
        jitter_flips_per_min=jitter,
    )


def main() -> int:
    score_files = sorted(DEMO_OUTPUTS_DIR.glob("*/score_results.json"))
    rows: list[Row] = []
    for score_file in score_files:
        try:
            rows.append(load_row(score_file))
        except Exception:
            # Keep the script robust: skip malformed/partial artifacts.
            continue

    rows.sort(key=lambda r: r.run_id)

    lines: list[str] = []
    lines.append("# Webcam domain-shift scoring summary")
    lines.append("")
    lines.append("Source artifacts: `demo/outputs/*/score_results.json`.")
    lines.append("Interpretation note (scope): These are deployment-facing webcam runs and are not directly comparable unless the same scoring protocol and labeling regime are used.")
    lines.append("")

    if not rows:
        lines.append("No `score_results.json` artifacts found.")
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    lines.append(
        "| Run | Scored frames | Raw acc | Raw macro-F1 (present) | Raw minority-F1 (lowest-3) | Smoothed acc | Smoothed macro-F1 (present) | Smoothed minority-F1 (lowest-3) | Jitter flips/min | Evidence |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )

    for r in rows:
        lines.append(
            "| {run_id} | {scored_frames} | {raw_acc:.6f} | {raw_mf1:.6f} | {raw_min3:.6f} | {sm_acc:.6f} | {sm_mf1:.6f} | {sm_min3:.6f} | {jitter:.4f} | `{evidence}` |".format(
                run_id=r.run_id,
                scored_frames=r.scored_frames,
                raw_acc=r.raw_acc,
                raw_mf1=r.raw_macro_f1_present,
                raw_min3=r.raw_minority_f1_lowest3,
                sm_acc=r.sm_acc,
                sm_mf1=r.sm_macro_f1_present,
                sm_min3=r.sm_minority_f1_lowest3,
                jitter=r.jitter_flips_per_min,
                evidence=r.score_path,
            )
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {OUT_MD}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
