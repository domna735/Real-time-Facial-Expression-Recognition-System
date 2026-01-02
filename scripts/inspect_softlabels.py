from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _summary(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().flatten().float().cpu()
    if x.numel() == 0:
        return {"n": 0.0}
    return {
        "n": float(x.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p01": float(x.quantile(0.01).item()),
        "p50": float(x.quantile(0.50).item()),
        "p99": float(x.quantile(0.99).item()),
    }


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect softlabels.npz sharpness / temperature effects.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--temperatures", type=str, default="1,1.2,2,4")
    args = ap.parse_args()

    data = np.load(args.npz)
    logits = torch.from_numpy(data["logits"]).float()
    y = torch.from_numpy(data["y"]).long()

    temps: List[float] = []
    for part in str(args.temperatures).split(","):
        part = part.strip()
        if not part:
            continue
        temps.append(float(part))
    if not temps:
        temps = [1.0]

    payload: Dict[str, object] = {
        "npz": str(args.npz),
        "logits": _summary(logits),
        "y": {"n": int(y.numel()), "min": int(y.min().item()) if y.numel() else None, "max": int(y.max().item()) if y.numel() else None},
        "by_temperature": {},
    }

    for t in temps:
        scaled = logits / float(t)
        p = F.softmax(scaled, dim=1)
        conf, pred = p.max(dim=1)
        ent = _entropy(p)
        acc = float((pred == y).float().mean().item())
        payload["by_temperature"][str(t)] = {
            "accuracy": acc,
            "max_prob": _summary(conf),
            "entropy": _summary(ent),
        }

    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(text)

    # Quick hint for KD: if max_prob p99 is ~1.0 at t=1, try higher T.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
