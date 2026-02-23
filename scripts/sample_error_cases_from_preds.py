"""Sample misclassified cases from an evaluation `preds.csv`.

This script is intentionally dependency-free (stdlib only).

Expected input columns (from `scripts/eval_student_checkpoint.py --save-preds`):
- image_path, source, y_true, y_pred, pred_prob, true_prob, margin_top1_top2, correct

It filters rows where correct == 0 and optionally by y_true/y_pred/source,
then sorts and writes the top K rows.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PredRow:
    image_path: str
    source: str
    y_true: str
    y_pred: str
    pred_prob: float
    true_prob: float
    margin_top1_top2: float
    correct: int


def _split_csv_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return set(items) if items else None


def _read_preds_csv(path: Path) -> Iterable[PredRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "image_path",
            "source",
            "y_true",
            "y_pred",
            "pred_prob",
            "true_prob",
            "margin_top1_top2",
            "correct",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for row in reader:
            yield PredRow(
                image_path=row["image_path"],
                source=row["source"],
                y_true=row["y_true"],
                y_pred=row["y_pred"],
                pred_prob=float(row["pred_prob"]),
                true_prob=float(row["true_prob"]),
                margin_top1_top2=float(row["margin_top1_top2"]),
                correct=int(row["correct"]),
            )


def _matches(row: PredRow, *, y_true: Optional[set[str]], y_pred: Optional[set[str]], source: Optional[set[str]]) -> bool:
    if row.correct != 0:
        return False
    if y_true is not None and row.y_true not in y_true:
        return False
    if y_pred is not None and row.y_pred not in y_pred:
        return False
    if source is not None and row.source not in source:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-csv", required=True, help="Path to preds.csv")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument(
        "--y-true",
        default=None,
        help="Comma-separated true labels to keep (e.g. 'Fear,Disgust'). If omitted, keeps all.",
    )
    ap.add_argument(
        "--y-pred",
        default=None,
        help="Comma-separated predicted labels to keep. If omitted, keeps all.",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Comma-separated sources to keep (e.g. 'expw_hq'). If omitted, keeps all.",
    )
    ap.add_argument("--top-k", type=int, default=50, help="Number of rows to output")
    ap.add_argument(
        "--sort-by",
        choices=["pred_prob_desc", "margin_asc", "true_prob_asc"],
        default="pred_prob_desc",
        help="Sort key for sampled errors.",
    )

    args = ap.parse_args()

    preds_path = Path(args.preds_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = _split_csv_list(args.y_true)
    y_pred = _split_csv_list(args.y_pred)
    source = _split_csv_list(args.source)

    rows = [
        r
        for r in _read_preds_csv(preds_path)
        if _matches(r, y_true=y_true, y_pred=y_pred, source=source)
    ]

    if args.sort_by == "pred_prob_desc":
        rows.sort(key=lambda r: r.pred_prob, reverse=True)
    elif args.sort_by == "margin_asc":
        rows.sort(key=lambda r: r.margin_top1_top2)
    elif args.sort_by == "true_prob_asc":
        rows.sort(key=lambda r: r.true_prob)

    rows = rows[: max(0, args.top_k)]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "source",
                "y_true",
                "y_pred",
                "pred_prob",
                "true_prob",
                "margin_top1_top2",
                "correct",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "source": r.source,
                    "y_true": r.y_true,
                    "y_pred": r.y_pred,
                    "pred_prob": f"{r.pred_prob:.6f}",
                    "true_prob": f"{r.true_prob:.6f}",
                    "margin_top1_top2": f"{r.margin_top1_top2:.6f}",
                    "correct": r.correct,
                }
            )

    print(
        {
            "preds_csv": str(preds_path),
            "out_csv": str(out_path),
            "selected": len(rows),
            "filters": {"y_true": sorted(y_true) if y_true else None, "y_pred": sorted(y_pred) if y_pred else None, "source": sorted(source) if source else None},
            "sort_by": args.sort_by,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
