"""Inspect training checkpoint metadata and (optionally) compare weights.

Why this exists:
- `alignmentreport.json` can reflect the *latest invocation* (e.g., a resume),
  so it may show `init_from=null` even if the run originally started from Stage A.
- This script loads one or two .pt files and:
  - prints obvious metadata (epoch, args/config if present)
  - compares a few tensor weights to estimate similarity

Usage (PowerShell):
  python scripts/inspect_checkpoint_provenance.py --ckpt-a outputs/teachers/.../best.pt
  python scripts/inspect_checkpoint_provenance.py --ckpt-a <stageA_best> --ckpt-b <stageB_checkpoint>

Notes:
- Similarity is heuristic; metadata (if present) is the authoritative signal.
"""

from __future__ import annotations

import argparse
import math
import pathlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def _load_checkpoint(path: Path) -> Any:
    """Load a .pt checkpoint in a way that works across PyTorch versions.

    PyTorch 2.6+ defaults `weights_only=True` which can reject some objects
    inside checkpoints (e.g., `pathlib.WindowsPath`). These checkpoints are
    produced by this repo, so we allowlist `WindowsPath` and retry.
    """

    # Allowlist WindowsPath for weights-only loader (PyTorch 2.6+ safety).
    try:
        torch.serialization.add_safe_globals([pathlib.WindowsPath])
    except Exception:
        pass

    # Try the safest mode first.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without `weights_only`.
        return torch.load(path, map_location="cpu")
    except Exception:
        # Fallback: full unpickling (only for trusted, locally-produced checkpoints).
        return torch.load(path, map_location="cpu", weights_only=False)


def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    return obj if isinstance(obj, dict) else None


def _find_state_dict(ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(ckpt, dict):
        # common conventions
        for key in ("state_dict", "model", "model_state", "net", "encoder"):
            val = ckpt.get(key)
            if isinstance(val, dict) and val and all(isinstance(v, torch.Tensor) for v in val.values()):
                return val

        # sometimes the checkpoint is *just* a state_dict
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]

    return None


def _find_args_like(ckpt: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    for key in ("args", "hparams", "config", "train_args", "run_args", "cli_args", "argv"):
        if key in ckpt:
            return key, ckpt[key]
    return None, None


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return _safe_float((a @ b) / denom)


def _l2_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = a.norm().clamp_min(1e-12)
    return _safe_float((a - b).norm() / denom)


def _pick_compare_keys(sd: Dict[str, torch.Tensor]) -> Iterable[str]:
    # Try common backbone + head keys across timm/torchvision.
    preferred = [
        "conv1.weight",
        "layer1.0.conv1.weight",
        "layer4.1.conv2.weight",
        "fc.weight",
        "fc.bias",
        "head.fc.weight",
        "head.bias",
        "classifier.weight",
        "classifier.bias",
    ]

    for k in preferred:
        if k in sd:
            yield k

    # If none hit, fall back to first few tensor keys.
    if not any(k in sd for k in preferred):
        count = 0
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                yield k
                count += 1
                if count >= 5:
                    break


def _print_metadata(name: str, ckpt_path: Path, ckpt: Any) -> None:
    print(f"\n=== {name} ===")
    print(f"path: {ckpt_path}")
    print(f"type: {type(ckpt)}")

    if not isinstance(ckpt, dict):
        sd = _find_state_dict(ckpt)
        print(f"dict checkpoint: no")
        print(f"state_dict present: {'yes' if sd is not None else 'no'}")
        return

    print(f"dict checkpoint: yes (keys={len(ckpt)})")

    for k in ("epoch", "global_step", "step"):
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")

    args_key, args_val = _find_args_like(ckpt)
    if args_key is not None:
        print(f"args-like key: {args_key} (type={type(args_val)})")
        if isinstance(args_val, dict):
            for kk in ("init_from", "resume", "output_dir", "model", "image_size", "seed", "exclude_sources"):
                if kk in args_val:
                    print(f"  {kk}: {args_val[kk]}")
        else:
            # argv is often a list of strings
            s = repr(args_val)
            print(f"  value repr: {s[:300]}")

    sd = _find_state_dict(ckpt)
    print(f"state_dict present: {'yes' if sd is not None else 'no'}")
    if sd is not None:
        print(f"state_dict tensors: {len(sd)}")


def _compare(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> None:
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    common = sorted(keys_a.intersection(keys_b))

    if not common:
        print("\nNo common state_dict keys; cannot compare tensors.")
        return

    print("\n=== Weight similarity (heuristic) ===")
    compared = 0
    for k in _pick_compare_keys(sd_a):
        if k not in sd_b:
            continue
        a = sd_a[k]
        b = sd_b[k]
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            continue
        if a.shape != b.shape:
            continue
        cos = _cosine_similarity(a, b)
        rel = _l2_relative(a, b)
        print(f"{k}: cosine={cos:.6f}  rel_l2={rel:.6f}  shape={tuple(a.shape)}")
        compared += 1

    if compared == 0:
        print("No comparable tensors found (shape/key mismatch).")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", type=Path, required=True, help="First checkpoint (.pt)")
    p.add_argument("--ckpt-b", type=Path, default=None, help="Optional second checkpoint (.pt) to compare")
    args = p.parse_args()

    ckpt_a = _load_checkpoint(args.ckpt_a)
    _print_metadata("A", args.ckpt_a, ckpt_a)

    if args.ckpt_b is None:
        return 0

    ckpt_b = _load_checkpoint(args.ckpt_b)
    _print_metadata("B", args.ckpt_b, ckpt_b)

    sd_a = _find_state_dict(ckpt_a)
    sd_b = _find_state_dict(ckpt_b)
    if sd_a is not None and sd_b is not None:
        _compare(sd_a, sd_b)
    else:
        print("\nMissing state_dict in one of the checkpoints; cannot compare weights.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
