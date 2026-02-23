"""Find duplicate KD student runs by comparing their saved training args.

This is intended for quick workspace cleanup on Windows.

- Reads `checkpoint_last.pt` (preferred) to extract `ckpt["args"]`.
- Builds a stable signature from a selected subset of args ("same knobs").
- Groups runs by signature and marks all but one per group as redundant.
- Keeps any run explicitly requested, plus runs referenced in markdown files.

Output:
- JSON to stdout with keys: keep, move, groups.

Note:
- This does NOT delete anything.
- The caller (PowerShell) is responsible for moving folders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _norm_path(p: Any) -> Optional[str]:
    if p is None:
        return None
    s = str(p)
    s = s.replace("\\", "/")
    return s


_SIGNATURE_KEYS_FULL: Tuple[str, ...] = (
    # identity / data
    "mode",
    "model",
    "image_size",
    "seed",
    "manifest",
    "data_root",
    "softlabels",
    "softlabels_index",
    # training
    "epochs",
    "batch_size",
    "num_workers",
    "use_clahe",
    "use_amp",
    "lr",
    "weight_decay",
    "warmup_epochs",
    # KD knobs
    "temperature",
    "alpha",
    # NegL knobs
    "use_negl",
    "negl_weight",
    "negl_ratio",
    "negl_gate",
    "negl_entropy_thresh",
    # NL knobs
    "use_nl",
    "nl_kind",
    "nl_embed",
    "nl_dim",
    "nl_momentum",
    "nl_proto_gate",
    "nl_consistency_thresh",
    "nl_topk_frac",
    "nl_weight",
    "nl_hidden_dim",
    "nl_layers",
)

_SIGNATURE_KEYS_KNOBS: Tuple[str, ...] = (
    # identity / data
    "mode",
    "model",
    "image_size",
    "seed",
    "manifest",
    "data_root",
    "softlabels",
    "softlabels_index",
    # training budget
    "epochs",
    # preprocessing/runtime that changes the computation
    "use_clahe",
    "use_amp",
    # KD knobs
    "temperature",
    "alpha",
    # NegL knobs
    "use_negl",
    "negl_weight",
    "negl_ratio",
    "negl_gate",
    "negl_entropy_thresh",
    # NL knobs
    "use_nl",
    "nl_kind",
    "nl_embed",
    "nl_dim",
    "nl_momentum",
    "nl_proto_gate",
    "nl_consistency_thresh",
    "nl_topk_frac",
    "nl_weight",
    "nl_hidden_dim",
    "nl_layers",
)


def _build_signature(args_dict: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    sig: Dict[str, Any] = {}
    for k in keys:
        if k not in args_dict:
            continue
        v = args_dict.get(k)
        if k in {"manifest", "data_root", "softlabels", "softlabels_index"}:
            sig[k] = _norm_path(v)
        else:
            sig[k] = v
    return sig


def _signature_key(sig: Dict[str, Any]) -> str:
    # Stable JSON string key.
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _prune_irrelevant_signature_fields(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Remove knobs that don't affect training given other enabled flags.

    This helps identify *truly redundant* runs where e.g. `use_negl=False` but
    `negl_entropy_thresh` differs due to defaults/CLI noise.
    """

    pruned = dict(sig)

    use_negl = bool(pruned.get("use_negl"))
    if not use_negl:
        for k in list(pruned.keys()):
            if k.startswith("negl_") and k != "use_negl":
                pruned.pop(k, None)
    else:
        gate = pruned.get("negl_gate")
        if gate != "entropy":
            pruned.pop("negl_entropy_thresh", None)

    use_nl = bool(pruned.get("use_nl"))
    if not use_nl:
        for k in list(pruned.keys()):
            if k.startswith("nl_") and k != "use_nl":
                pruned.pop(k, None)
    else:
        nl_kind = pruned.get("nl_kind")
        # Currently we only care about proto-specific knobs; if a different kind
        # is ever used, drop proto-only fields to avoid blocking duplicates.
        if nl_kind != "proto":
            for k in (
                "nl_momentum",
                "nl_proto_gate",
                "nl_consistency_thresh",
                "nl_topk_frac",
            ):
                pruned.pop(k, None)
        else:
            gate = pruned.get("nl_proto_gate")
            if gate == "topk":
                # top-k selection does not use a threshold
                pruned.pop("nl_consistency_thresh", None)
            elif gate == "threshold":
                # threshold selection does not use top-k fraction
                pruned.pop("nl_topk_frac", None)

    return pruned


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_md_keeps(md_paths: Sequence[Path]) -> List[str]:
    # Keep any run dirs mentioned like outputs/students/KD/<name>/
    keep: List[str] = []
    pat = re.compile(r"outputs/students/KD/([A-Za-z0-9_\-]+)")
    for mp in md_paths:
        try:
            text = mp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in pat.finditer(text.replace("\\", "/")):
            keep.append(m.group(1))
    return sorted(set(keep))


def _torch_load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception:
        return None


@dataclass
class RunInfo:
    run_dir: Path
    name: str
    mtime: float
    signature: Dict[str, Any]
    signature_key: str
    raw_macro_f1: Optional[float]


def _read_macro_f1(run_dir: Path) -> Optional[float]:
    rel = run_dir / "reliabilitymetrics.json"
    data = _safe_load_json(rel)
    if not isinstance(data, dict):
        return None
    raw = data.get("raw")
    if not isinstance(raw, dict):
        return None
    mf1 = raw.get("macro_f1")
    try:
        return float(mf1)
    except Exception:
        return None


def _collect_kd_runs(kd_root: Path, keys: Tuple[str, ...]) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for d in sorted(kd_root.iterdir() if kd_root.exists() else []):
        if not d.is_dir():
            continue
        if "_KD_" not in d.name:
            continue

        ckpt_path = d / "checkpoint_last.pt"
        args_dict: Dict[str, Any] = {}

        ckpt = _torch_load_checkpoint(ckpt_path) if ckpt_path.exists() else None
        if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
            args_dict = dict(ckpt["args"])  # shallow copy
        else:
            # Fallback: infer minimal signature from history.json (nl/negl only)
            hist = _safe_load_json(d / "history.json")
            if isinstance(hist, list) and hist and isinstance(hist[0], dict):
                args_dict = {
                    "mode": "kd",
                    "model": args_dict.get("model"),
                    "image_size": args_dict.get("image_size"),
                    "seed": args_dict.get("seed"),
                }
                first = hist[0]
                if isinstance(first.get("nl"), dict):
                    nl = first["nl"]
                    args_dict.update(
                        {
                            "use_nl": bool(nl.get("enabled")),
                            "nl_kind": nl.get("kind"),
                            "nl_embed": nl.get("embed"),
                            "nl_dim": nl.get("dim"),
                            "nl_momentum": nl.get("momentum"),
                            "nl_proto_gate": nl.get("proto_gate"),
                            "nl_consistency_thresh": nl.get("consistency_thresh"),
                            "nl_topk_frac": nl.get("topk_frac"),
                            "nl_weight": nl.get("weight"),
                        }
                    )
                if isinstance(first.get("negl"), dict):
                    negl = first["negl"]
                    args_dict.update(
                        {
                            "use_negl": bool(negl.get("enabled")),
                            "negl_weight": negl.get("weight"),
                            "negl_ratio": negl.get("ratio"),
                            "negl_gate": negl.get("gate"),
                            "negl_entropy_thresh": negl.get("entropy_thresh"),
                        }
                    )

        sig = _prune_irrelevant_signature_fields(_build_signature(args_dict, keys))
        skey = _signature_key(sig)
        runs.append(
            RunInfo(
                run_dir=d,
                name=d.name,
                mtime=d.stat().st_mtime,
                signature=sig,
                signature_key=skey,
                raw_macro_f1=_read_macro_f1(d),
            )
        )

    return runs


def _pick_keep(runs: List[RunInfo], prefer: str, forced_keep_names: set[str]) -> RunInfo:
    # Any forced-keep inside this signature group wins.
    for r in runs:
        if r.name in forced_keep_names:
            return r

    if prefer == "oldest":
        return sorted(runs, key=lambda r: (r.mtime, r.name))[0]
    if prefer == "best":
        # prefer best macro_f1; fallback to newest
        def score(r: RunInfo) -> Tuple[int, float, float]:
            has = 1 if r.raw_macro_f1 is not None else 0
            mf1 = r.raw_macro_f1 if r.raw_macro_f1 is not None else -1.0
            return (has, mf1, r.mtime)

        return sorted(runs, key=score, reverse=True)[0]

    # newest
    return sorted(runs, key=lambda r: (r.mtime, r.name), reverse=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--kd-root", type=str, default="outputs/students/KD")
    ap.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Run dir name or path to always keep. Can be repeated.",
    )
    ap.add_argument(
        "--keep-from-md",
        action="append",
        default=[],
        help="Markdown file to scan for outputs/students/KD/<dir> references; all referenced dirs will be kept.",
    )
    ap.add_argument(
        "--prefer",
        choices=["newest", "oldest", "best"],
        default="newest",
        help="Which run to keep within each duplicate group.",
    )
    ap.add_argument(
        "--signature-level",
        choices=["knobs", "full"],
        default="knobs",
        help="How strict the duplicate detection is: 'knobs' ignores runtime knobs; 'full' is stricter.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    kd_root = Path(args.kd_root)
    if not kd_root.is_absolute():
        kd_root = repo_root / kd_root

    forced_keep_names: set[str] = set()
    for k in args.keep:
        p = Path(k)
        name = p.name
        forced_keep_names.add(name)

    md_paths = [Path(p) for p in args.keep_from_md]
    md_paths = [p if p.is_absolute() else (repo_root / p) for p in md_paths]
    forced_keep_names.update(_extract_md_keeps(md_paths))

    keys = _SIGNATURE_KEYS_KNOBS if args.signature_level == "knobs" else _SIGNATURE_KEYS_FULL
    runs = _collect_kd_runs(kd_root, keys)
    by_sig: Dict[str, List[RunInfo]] = {}
    for r in runs:
        by_sig.setdefault(r.signature_key, []).append(r)

    keep_dirs: List[str] = []
    move_dirs: List[str] = []
    groups_out: List[Dict[str, Any]] = []

    for skey, group in sorted(by_sig.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        if len(group) <= 1:
            keep_dirs.append(str(group[0].run_dir))
            continue

        keeper = _pick_keep(group, args.prefer, forced_keep_names)
        keep_dirs.append(str(keeper.run_dir))
        redundant = [g for g in group if g.run_dir != keeper.run_dir]
        move_dirs.extend(str(g.run_dir) for g in redundant)

        groups_out.append(
            {
                "count": len(group),
                "keep": str(keeper.run_dir),
                "move": [str(g.run_dir) for g in redundant],
                "signature": keeper.signature,
            }
        )

    out = {
        "kd_root": str(kd_root),
        "prefer": args.prefer,
        "signature_level": args.signature_level,
        "keep": sorted(set(keep_dirs)),
        "move": sorted(set(move_dirs)),
        "groups": groups_out,
    }

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
