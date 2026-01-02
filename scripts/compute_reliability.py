from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, *, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = (pred == y).to(torch.float32)

    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            prop = in_bin.to(torch.float32).mean()
            acc = correct[in_bin].mean()
            avg_conf = conf[in_bin].mean()
            ece = ece + prop * (avg_conf - acc).abs()

    return float(ece.item())


def confusion_from_logits(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    idx = (y * num_classes + pred).to(torch.int64)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    num_classes = cm.shape[0]
    per: Dict[str, float] = {}
    f1s: List[float] = []
    for i in range(num_classes):
        tp = float(cm[i, i].item())
        fp = float(cm[:, i].sum().item() - tp)
        fn = float(cm[i, :].sum().item() - tp)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        per[CANONICAL_7[i]] = float(f1)
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s))), per


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int) -> Dict[str, object]:
    y = y.long()
    probs = F.softmax(logits, dim=1)

    cm = confusion_from_logits(logits, y, num_classes=num_classes)
    correct = float(cm.diag().sum().item())
    total = float(cm.sum().item())
    acc = correct / max(1.0, total)

    macro_f1, per_f1 = f1_from_confusion(cm)
    nll = float(F.cross_entropy(logits, y).item())
    ece = expected_calibration_error(probs, y)

    # Multi-class Brier score: mean over samples of sum_k (p_k - 1[y==k])^2
    y_onehot = F.one_hot(y, num_classes=num_classes).to(probs.dtype)
    brier = float(((probs - y_onehot) ** 2).sum(dim=1).mean().item())

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": dict(per_f1),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
    }


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, *, init_t: float = 1.2) -> float:
    import math

    logits = logits.float()
    y = y.long()

    log_t = torch.tensor([math.log(init_t)], dtype=torch.float32, requires_grad=True)

    def nll() -> torch.Tensor:
        t = torch.exp(log_t)
        return F.cross_entropy(logits / t, y)

    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = nll()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).detach().cpu().item())
    return float(max(0.5, min(5.0, t)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute reliability metrics (acc/macro-F1/ECE/NLL/Brier) for logits.")
    ap.add_argument("--npz", type=Path, required=True, help="NPZ containing logits and y arrays")
    ap.add_argument("--out", type=Path, required=True, help="Output reliabilitymetrics.json")
    ap.add_argument("--temperature", type=str, default="global", choices=["none", "global"], help="Temperature scaling")
    ap.add_argument("--fixed-temperature", type=float, default=None)
    args = ap.parse_args()

    import numpy as np

    data = np.load(args.npz)
    logits = torch.from_numpy(data["logits"]).float()
    y = torch.from_numpy(data["y"]).long()

    raw = metrics_from_logits(logits, y, num_classes=len(CANONICAL_7))

    t_star: float = 1.0
    logits_scaled = logits
    if args.temperature == "global":
        if args.fixed_temperature is not None:
            t_star = float(args.fixed_temperature)
        else:
            t_star = fit_temperature(logits, y, init_t=1.2)
        logits_scaled = logits / float(t_star)

    scaled = metrics_from_logits(logits_scaled, y, num_classes=len(CANONICAL_7))

    payload = {
        "raw": raw,
        "temperature_scaled": {
            "mode": str(args.temperature),
            "global_temperature": float(t_star),
            **scaled,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
