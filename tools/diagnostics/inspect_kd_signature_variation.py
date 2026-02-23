"""Inspect KD run signature variation.

This is a debugging helper for the KD duplicate-cleanup tooling.
It prints:
- how many runs were found
- how many duplicate groups exist at the given signature level
- which signature fields vary the most

Usage:
  python tools/diagnostics/inspect_kd_signature_variation.py --signature-level knobs
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _norm_path(p: Any) -> str | None:
    if p is None:
        return None
    return str(p).replace("\\", "/")


_SIGNATURE_KEYS_KNOBS: Tuple[str, ...] = (
    "mode",
    "model",
    "image_size",
    "seed",
    "manifest",
    "data_root",
    "softlabels",
    "softlabels_index",
    "epochs",
    "use_clahe",
    "use_amp",
    "temperature",
    "alpha",
    "use_negl",
    "negl_weight",
    "negl_ratio",
    "negl_gate",
    "negl_entropy_thresh",
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

_SIGNATURE_KEYS_FULL: Tuple[str, ...] = (
    *_SIGNATURE_KEYS_KNOBS,
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "warmup_epochs",
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
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _torch_load_checkpoint(path: Path) -> Dict[str, Any] | None:
    try:
        import torch  # type: ignore

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _iter_kd_run_dirs(kd_root: Path) -> Iterable[Path]:
    for d in sorted(kd_root.iterdir() if kd_root.exists() else []):
        if d.is_dir() and "_KD_" in d.name:
            yield d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--kd-root", type=str, default="outputs/students/KD")
    ap.add_argument("--signature-level", choices=["knobs", "full"], default="knobs")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    kd_root = Path(args.kd_root)
    if not kd_root.is_absolute():
        kd_root = repo_root / kd_root

    keys = _SIGNATURE_KEYS_KNOBS if args.signature_level == "knobs" else _SIGNATURE_KEYS_FULL

    sig_keys: List[str] = []
    sigs: List[Dict[str, Any]] = []
    names: List[str] = []

    for d in _iter_kd_run_dirs(kd_root):
        ckpt_path = d / "checkpoint_last.pt"
        ckpt = _torch_load_checkpoint(ckpt_path) if ckpt_path.exists() else None
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get("args"), dict):
            continue
        args_dict = dict(ckpt["args"])
        sig = _build_signature(args_dict, keys)
        sigs.append(sig)
        sig_keys.append(_signature_key(sig))
        names.append(d.name)

    total = len(names)
    by = Counter(sig_keys)
    dup_groups = [k for k, c in by.items() if c > 1]

    print(f"KD root: {kd_root}")
    print(f"Runs with checkpoint args: {total}")
    print(f"Unique signatures: {len(by)}")
    print(f"Duplicate groups: {len(dup_groups)}")

    if total == 0:
        return 0

    # Which fields vary the most?
    field_values: Dict[str, set[str]] = defaultdict(set)
    for sig in sigs:
        for k in keys:
            if k not in sig:
                continue
            v = sig[k]
            field_values[k].add(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v))

    var = sorted(((k, len(vs)) for k, vs in field_values.items()), key=lambda kv: kv[1], reverse=True)
    print("\nMost-varying signature fields:")
    for k, n in var[: args.top]:
        print(f"  {k}: {n} unique")

    # Show examples for path fields.
    for k in ("manifest", "data_root", "softlabels", "softlabels_index"):
        if k in field_values:
            vals = sorted(field_values[k])
            print(f"\n{k} examples ({min(len(vals), 8)} of {len(vals)}):")
            for v in vals[:8]:
                print(f"  {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
