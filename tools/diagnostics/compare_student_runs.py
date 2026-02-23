from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    label: str
    mode: Optional[str]
    epochs: Optional[int]
    negl: Optional[Dict[str, Any]]
    nl: Optional[Dict[str, Any]]
    raw: Dict[str, Any]
    ts: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_last_aux(history_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[int]]:
    try:
        hist = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    if not isinstance(hist, list) or not hist:
        return None, None, None

    last = hist[-1] if isinstance(hist[-1], dict) else None
    if not isinstance(last, dict):
        return None, None, None

    epochs = None
    try:
        epochs = int(last.get("epoch", 0)) + 1
    except Exception:
        epochs = None

    negl = last.get("negl")
    nl = last.get("nl")
    if isinstance(negl, dict):
        negl_out: Optional[Dict[str, Any]] = negl
    else:
        negl_out = None

    nl_out: Optional[Dict[str, Any]] = nl if isinstance(nl, dict) else None

    return negl_out, nl_out, epochs


def _infer_mode_from_dirname(name: str) -> Optional[str]:
    u = name.upper()
    if "_CE_" in u:
        return "ce"
    if "_KD_" in u:
        return "kd"
    if "_DKD_" in u:
        return "dkd"
    return None


def load_run(run_dir: Path, *, label: Optional[str] = None) -> RunSummary:
    run_dir = run_dir.resolve()
    rel_path = run_dir / "reliabilitymetrics.json"
    if not rel_path.exists():
        raise SystemExit(f"Missing reliabilitymetrics.json: {rel_path}")

    rel = _read_json(rel_path)
    raw = rel.get("raw") if isinstance(rel.get("raw"), dict) else {}
    ts = rel.get("temperature_scaled") if isinstance(rel.get("temperature_scaled"), dict) else {}

    hist_path = run_dir / "history.json"
    negl = None
    nl = None
    epochs = None
    if hist_path.exists():
        negl, nl, epochs = _maybe_read_last_aux(hist_path)

    mode = _infer_mode_from_dirname(run_dir.name)

    return RunSummary(
        run_dir=run_dir,
        label=label or run_dir.name,
        mode=mode,
        epochs=epochs,
        negl=negl,
        nl=nl,
        raw=raw,
        ts=ts,
    )


def _fmt_float(v: Any, *, nd: int = 6) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def _get_per_class_f1(d: Dict[str, Any]) -> Dict[str, float]:
    pc = d.get("per_class_f1")
    if not isinstance(pc, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in pc.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def _minority_f1(per_class_f1: Dict[str, float], *, k: int = 3) -> Optional[float]:
    if not per_class_f1:
        return None
    vals = sorted(per_class_f1.values())
    if not vals:
        return None
    kk = max(1, min(int(k), len(vals)))
    return float(sum(vals[:kk]) / kk)


def render_markdown_table(runs: List[RunSummary]) -> str:
    headers = [
        "Label",
        "Mode",
        "Epochs",
        "NegL",
        "NL",
        "Raw acc",
        "Raw macro-F1",
        "Raw ECE",
        "Raw NLL",
        "TS ECE",
        "TS NLL",
        "Minority-F1 (lowest-3)",
        "Run dir",
    ]

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for r in runs:
        raw_pc = _get_per_class_f1(r.raw)
        mf1 = _minority_f1(raw_pc, k=3)

        negl_tag = "off"
        if isinstance(r.negl, dict):
            try:
                negl_tag = (
                    f"on (w={r.negl.get('weight')}, ratio={r.negl.get('ratio')}, gate={r.negl.get('gate')})"
                )
            except Exception:
                negl_tag = "on"

        nl_tag = "off"
        if isinstance(r.nl, dict):
            try:
                kind = r.nl.get("kind")
                if kind == "proto":
                    dim = r.nl.get("dim")
                    mom = r.nl.get("momentum")
                    thr = r.nl.get("consistency_thresh")
                    w = r.nl.get("weight")
                    nl_tag = f"on (proto dim={dim}, m={mom}, thr={thr}, w={w})"
                elif kind == "negl_gate":
                    hd = r.nl.get("hidden_dim")
                    layers = r.nl.get("layers")
                    nl_tag = f"on (negl_gate hd={hd}, layers={layers})"
                else:
                    # Backward-compat for older history.json
                    hd = r.nl.get("hidden_dim")
                    layers = r.nl.get("layers")
                    if hd is not None or layers is not None:
                        nl_tag = f"on (hd={hd}, layers={layers})"
                    else:
                        nl_tag = "on"
            except Exception:
                nl_tag = "on"

        row = [
            r.label,
            r.mode or "-",
            str(r.epochs) if r.epochs is not None else "-",
            negl_tag,
            nl_tag,
            _fmt_float(r.raw.get("accuracy")),
            _fmt_float(r.raw.get("macro_f1")),
            _fmt_float(r.raw.get("ece")),
            _fmt_float(r.raw.get("nll")),
            _fmt_float(r.ts.get("ece")),
            _fmt_float(r.ts.get("nll")),
            _fmt_float(mf1) if mf1 is not None else "-",
            str(r.run_dir),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare student runs by reading reliabilitymetrics.json (+ NegL settings).")
    ap.add_argument("run_dirs", nargs="+", type=Path, help="One or more outputs/students/<RUN> directories")
    ap.add_argument("--label", action="append", default=[], help="Optional label per run_dir (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="Optional markdown output path")
    args = ap.parse_args()

    labels: List[str] = list(args.label or [])
    while len(labels) < len(args.run_dirs):
        labels.append("")

    runs = [load_run(d, label=(labels[i] or None)) for i, d in enumerate(args.run_dirs)]
    md = render_markdown_table(runs)

    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
