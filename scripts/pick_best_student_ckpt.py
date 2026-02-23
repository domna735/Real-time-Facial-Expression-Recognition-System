from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BestStudent:
    ckpt: str
    metrics: str
    macro_f1: float
    accuracy: float


def _find_best_student(root: Path) -> Optional[BestStudent]:
    best_macro = -1.0
    best_acc = -1.0
    best_ckpt: Optional[Path] = None
    best_metrics: Optional[Path] = None

    for metrics_path in root.rglob("reliabilitymetrics.json"):
        run_dir = metrics_path.parent
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            continue

        try:
            rel: Any = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        raw = rel.get("raw") if isinstance(rel, dict) else None
        if not isinstance(raw, dict):
            continue

        macro = raw.get("macro_f1")
        acc = raw.get("accuracy")
        if not isinstance(macro, (int, float)) or not isinstance(acc, (int, float)):
            continue

        macro_f = float(macro)
        acc_f = float(acc)

        if (macro_f, acc_f) > (best_macro, best_acc):
            best_macro, best_acc = macro_f, acc_f
            best_ckpt = ckpt_path
            best_metrics = metrics_path

    if best_ckpt is None or best_metrics is None:
        return None

    # Use workspace-relative, forward-slash paths so PowerShell + JSON parsing is painless.
    ckpt_rel = best_ckpt.relative_to(REPO_ROOT).as_posix()
    metrics_rel = best_metrics.relative_to(REPO_ROOT).as_posix()

    return BestStudent(
        ckpt=ckpt_rel,
        metrics=metrics_rel,
        macro_f1=best_macro,
        accuracy=best_acc,
    )


def main() -> int:
    root = REPO_ROOT / "outputs" / "students"
    best = _find_best_student(root) if root.exists() else None

    if best is None:
        print(json.dumps({"error": "no_student_checkpoints_found"}))
        return 2

    print(
        json.dumps(
            {
                "ckpt": best.ckpt,
                "metrics": best.metrics,
                "raw": {"macro_f1": best.macro_f1, "accuracy": best.accuracy},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
