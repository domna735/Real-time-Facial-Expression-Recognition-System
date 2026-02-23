from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import (  # noqa: E402
    CANONICAL_7,
    ManifestImageDataset,
    build_splits,
    read_manifest,
)
from src.fer.nl.memory import AssociativeMemory  # noqa: E402
from src.fer.negl.losses import complementary_negative_loss  # noqa: E402
from src.fer.utils.device import get_best_device  # noqa: E402


try:
    import timm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("timm is required for student training.") from e


def _write_run_lock(output_dir: Path, *, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "argv": sys.argv,
        "model": getattr(args, "model", None),
        "image_size": getattr(args, "image_size", None),
    }
    try:
        lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    return lock_path


def _remove_run_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except Exception:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def lr_for_step(step: int, *, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": dict(per_f1),
        "nll": float(nll),
        "ece": float(ece),
    }


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, *, init_t: float = 1.2) -> float:
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


def kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    t: float,
) -> torch.Tensor:
    # KL(teacher||student) with temperature scaling.
    p_t = F.softmax(teacher_logits / t, dim=1)
    log_p_s = F.log_softmax(student_logits / t, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y: torch.Tensor,
    *,
    t: float,
    beta: float,
) -> torch.Tensor:
    """A simple DKD variant.

    This is intentionally minimal: target vs non-target decoupling.
    """

    y = y.long()
    num_classes = student_logits.shape[1]

    s = student_logits / t
    te = teacher_logits / t

    # Masks
    y_onehot = F.one_hot(y, num_classes=num_classes).to(s.dtype)
    not_y = 1.0 - y_onehot

    # TCKD: KL on target-vs-others binary distribution
    s_prob = F.softmax(s, dim=1)
    t_prob = F.softmax(te, dim=1)

    s_t = (s_prob * y_onehot).sum(dim=1, keepdim=True)
    s_nt = (s_prob * not_y).sum(dim=1, keepdim=True)
    t_t = (t_prob * y_onehot).sum(dim=1, keepdim=True)
    t_nt = (t_prob * not_y).sum(dim=1, keepdim=True)

    s_bin = torch.cat([s_t, s_nt], dim=1).clamp_min(1e-8)
    t_bin = torch.cat([t_t, t_nt], dim=1).clamp_min(1e-8)
    tckd = F.kl_div(s_bin.log(), t_bin, reduction="batchmean")

    # NCKD: KL among non-target classes only
    s_prob_nt = (s_prob * not_y) / (s_nt + 1e-8)
    t_prob_nt = (t_prob * not_y) / (t_nt + 1e-8)
    nckd = F.kl_div((s_prob_nt + 1e-8).log(), t_prob_nt, reduction="batchmean")

    return tckd + beta * nckd


def _load_softlabels_index(index_path: Path) -> Dict[str, int]:
    """Return mapping: image_path -> row index in softlabels.npz."""
    mapping: Dict[str, int] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            p = rec.get("image_path")
            i = rec.get("i")
            if not isinstance(p, str) or not p:
                continue
            try:
                ii = int(i)
            except Exception:
                continue
            mapping[p] = ii
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description="Train student model with KD/DKD using exported softlabels.")

    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "Training_data_cleaned" / "classification_manifest_hq_train.csv",
    )
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "Training_data_cleaned")
    ap.add_argument(
        "--softlabels",
        type=Path,
        default=None,
        help="softlabels.npz from scripts/export_softlabels.py (required for kd/dkd; not needed for ce)",
    )
    ap.add_argument(
        "--softlabels-index",
        type=Path,
        default=None,
        help="softlabels_index.jsonl from scripts/export_softlabels.py (defaults to sibling file)",
    )

    ap.add_argument("--model", type=str, default="mobilenetv3_large_100", help="timm model name")
    ap.add_argument("--image-size", type=int, default=224)

    ap.add_argument("--use-clahe", action="store_true")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)

    ap.add_argument("--alpha", type=float, default=0.5, help="KD/DKD weight")
    ap.add_argument("--beta", type=float, default=4.0, help="DKD beta (weight on non-target KD)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--mode", type=str, default="kd", choices=["ce", "kd", "dkd"])

    # NegL (complementary-label negative learning) - optional auxiliary loss.
    ap.add_argument("--use-negl", action="store_true")
    ap.add_argument("--negl-weight", type=float, default=0.1, help="lambda: weight of NegL term")
    ap.add_argument(
        "--negl-ratio",
        type=float,
        default=0.5,
        help="fraction of batch to apply NegL on (0..1); remaining samples have weight=0",
    )
    ap.add_argument("--negl-gate", type=str, default="entropy", choices=["none", "entropy"])
    ap.add_argument(
        "--negl-entropy-thresh",
        type=float,
        default=0.7,
        help="apply NegL only when normalized entropy >= thresh (0..1), if gate=entropy",
    )

    # Optional: use extra manifest columns for self-learning buffers.
    ap.add_argument(
        "--manifest-use-weights",
        action="store_true",
        help="If set: use per-row 'weight' column as a per-sample weight for the CE term (defaults to 1.0 if missing).",
    )
    ap.add_argument(
        "--manifest-use-neg-label",
        action="store_true",
        help=(
            "If set: use per-row 'neg_label' column to define NegL targets (applies NegL only where provided; uses -1 sentinel otherwise)."
        ),
    )

    # NL (two modes):
    # - proto: prototype memory + momentum smoothing + consistency-gated auxiliary loss (stable default)
    # - negl_gate: learned gate for NegL sample weights (legacy; kept for reproducibility)
    ap.add_argument("--use-nl", action="store_true", help="Enable NL auxiliary module (see --nl-kind).")
    ap.add_argument(
        "--nl-kind",
        type=str,
        default="proto",
        choices=["proto", "negl_gate"],
        help="NL variant to use.",
    )

    # NL(proto)
    ap.add_argument(
        "--nl-embed",
        type=str,
        default="penultimate",
        choices=["penultimate", "logits"],
        help="Representation source for NL(proto): penultimate features (recommended) or logits (legacy).",
    )
    ap.add_argument("--nl-dim", type=int, default=32, help="Prototype memory dimension (32-64 recommended).")
    ap.add_argument("--nl-momentum", type=float, default=0.9, help="EMA momentum for prototypes (0..1).")
    ap.add_argument(
        "--nl-proto-gate",
        type=str,
        default="fixed",
        choices=["fixed", "topk"],
        help="NL(proto) gating strategy: fixed threshold on (1-cos) vs top-k most inconsistent per batch.",
    )
    ap.add_argument(
        "--nl-consistency-thresh",
        type=float,
        default=0.2,
        help="Apply NL proto loss only when (1 - cosine_sim) >= thresh.",
    )
    ap.add_argument(
        "--nl-topk-frac",
        type=float,
        default=0.1,
        help="If --nl-proto-gate topk: target fraction of batch to apply NL on (0..1).",
    )
    ap.add_argument("--nl-weight", type=float, default=0.1, help="Weight of NL proto auxiliary loss.")

    # NL(negl_gate) legacy args
    ap.add_argument("--nl-hidden-dim", type=int, default=32)
    ap.add_argument("--nl-layers", type=int, default=1)

    # LP loss (Deep Locality-Preserving loss; Paper #5 Track A)
    ap.add_argument(
        "--lp-weight",
        type=float,
        default=0.0,
        help="lambda: weight of locality-preserving (LP) loss term (0 disables)",
    )
    ap.add_argument(
        "--lp-k",
        type=int,
        default=20,
        help="k nearest neighbors within-class (batch approximation) for LP loss",
    )
    ap.add_argument(
        "--lp-embed",
        type=str,
        default="penultimate",
        choices=["penultimate", "logits"],
        help="Representation source for LP loss.",
    )

    # Optional: after training, run the existing eval script to produce standard gate artifacts
    # (outputs/evals/students/*/reliabilitymetrics.json) for eval-only + ExpW.
    ap.add_argument(
        "--post-eval",
        action="store_true",
        help="After training, evaluate best checkpoint on eval-only + ExpW using scripts/eval_student_checkpoint.py",
    )
    ap.add_argument(
        "--post-eval-evalonly-manifest",
        type=Path,
        default=REPO_ROOT / "Training_data_cleaned" / "classification_manifest_eval_only.csv",
    )
    ap.add_argument(
        "--post-eval-expw-manifest",
        type=Path,
        default=REPO_ROOT / "Training_data_cleaned" / "expw_full_manifest.csv",
    )

    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)

    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt) and start fresh (no optimizer/scaler/epoch resume). "
            "Use this for conservative fine-tuning / domain adaptation."
        ),
    )
    ap.add_argument(
        "--tune",
        type=str,
        default="all",
        choices=["all", "head", "bn", "lastblock_head"],
        help=(
            "Which parameters to update. all=full training; head=classifier head only; bn=BatchNorm affine only; "
            "lastblock_head=last backbone block + head (heuristic)."
        ),
    )

    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-val-batches", type=int, default=0)

    args = ap.parse_args()

    if bool(args.use_nl) and str(args.nl_kind) == "negl_gate" and (not bool(args.use_negl)):
        raise SystemExit("--use-nl --nl-kind negl_gate requires --use-negl")
    if bool(args.manifest_use_neg_label) and (not bool(args.use_negl)):
        raise SystemExit("--manifest-use-neg-label requires --use-negl")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = REPO_ROOT / "outputs" / "students" / f"{args.model}_img{args.image_size}_seed{args.seed}_{stamp}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _write_run_lock(args.output_dir, args=args)
    atexit.register(_remove_run_lock, lock_path)

    device_info = get_best_device()
    device = device_info.device

    use_amp = bool(args.use_amp) and device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"

    # Load manifest rows and split.
    rows_all = read_manifest(args.manifest)
    train_rows, val_rows, _test_rows = build_splits(rows_all, out_root=args.data_root)

    # Softlabels are only required for KD/DKD.
    teacher_logits_cpu: Optional[torch.Tensor] = None
    path_to_i: Optional[Dict[str, int]] = None
    if args.mode != "ce":
        if args.softlabels is None:
            raise SystemExit("--softlabels is required when --mode is kd/dkd")
        if not args.softlabels.exists():
            raise SystemExit(f"Softlabels file not found: {args.softlabels}")

        import numpy as np

        sl = np.load(args.softlabels)
        if "logits" not in sl:
            raise SystemExit(f"softlabels.npz missing key 'logits': {args.softlabels}")
        teacher_logits_cpu = torch.from_numpy(sl["logits"]).float()

        # Load index mapping
        index_path = args.softlabels_index
        if index_path is None:
            index_path = args.softlabels.parent / "softlabels_index.jsonl"
        if not index_path.exists():
            raise SystemExit(f"Softlabels index not found: {index_path}")
        path_to_i = _load_softlabels_index(index_path)
        if not path_to_i:
            raise SystemExit(f"Softlabels index is empty/unreadable: {index_path}")

        # Coverage check on train split (fail-fast if export doesn't match manifest/split).
        missing = 0
        for r in train_rows:
            if r.image_path not in path_to_i:
                missing += 1
                if missing <= 5:
                    print(f"Missing softlabel for image_path: {r.image_path}")
        if missing:
            raise SystemExit(
                f"Softlabels coverage mismatch: missing {missing}/{len(train_rows)} train rows. "
                "Ensure export_softlabels used the same --manifest/--data-root/--split=train."
            )

    # Datasets
    # Keep transforms consistent with teacher eval transforms (center crop pipeline).
    # Import by module name so Windows DataLoader workers can pickle custom transforms.
    import importlib

    train_teacher = importlib.import_module("scripts.train_teacher")

    train_tf = train_teacher.build_transforms(
        image_size=int(args.image_size),
        train=True,
        use_clahe=bool(args.use_clahe),
        clahe_clip=float(args.clahe_clip),
        clahe_tile=int(args.clahe_tile),
    )
    val_tf = train_teacher.build_transforms(
        image_size=int(args.image_size),
        train=False,
        use_clahe=bool(args.use_clahe),
        clahe_clip=float(args.clahe_clip),
        clahe_tile=int(args.clahe_tile),
    )

    return_meta = bool(args.manifest_use_weights) or bool(args.manifest_use_neg_label)
    train_ds = ManifestImageDataset(
        train_rows,
        out_root=args.data_root,
        transform=train_tf,
        return_path=True,
        return_meta=return_meta,
    )
    val_ds = ManifestImageDataset(val_rows, out_root=args.data_root, transform=val_tf)

    # Windows can hit "Couldn't open shared file mapping" (error 1455) when too many
    # prefetched batches are in-flight (large batch_size * num_workers). Reduce prefetch.
    num_workers = int(args.num_workers)
    prefetch_factor = 1 if (os.name == "nt" and num_workers > 0) else 2

    train_dl_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": True,
    }
    val_dl_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        train_dl_kwargs["prefetch_factor"] = prefetch_factor
        val_dl_kwargs["prefetch_factor"] = prefetch_factor
        # Avoid re-spawning worker processes every epoch (especially important on Windows).
        train_dl_kwargs["persistent_workers"] = True
        val_dl_kwargs["persistent_workers"] = True

    train_dl = DataLoader(train_ds, **train_dl_kwargs)
    val_dl = DataLoader(val_ds, **val_dl_kwargs)

    # Student model
    # If we're initializing from an existing checkpoint, don't download pretrained weights.
    use_pretrained = (args.init_from is None)
    student = timm.create_model(str(args.model), pretrained=bool(use_pretrained), num_classes=len(CANONICAL_7)).to(device)

    # Optional: initialize weights from an existing checkpoint but start fresh (no optimizer/scaler resume).
    if args.init_from is not None:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise SystemExit(f"--init-from checkpoint not found: {init_path}")
        try:
            init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        except TypeError:
            init_ckpt = torch.load(init_path, map_location="cpu")
        student.load_state_dict(init_ckpt.get("model", {}), strict=True)
        print(f"[init-from] Loaded model weights from {init_path}")

    def _infer_head_modules(model: nn.Module) -> List[nn.Module]:
        mods: List[nn.Module] = []

        # timm models often expose get_classifier() which returns a module or a name.
        if hasattr(model, "get_classifier"):
            try:
                c = getattr(model, "get_classifier")()
                if isinstance(c, str):
                    m = getattr(model, c, None)
                    if isinstance(m, nn.Module):
                        mods.append(m)
                elif isinstance(c, nn.Module):
                    mods.append(c)
                elif isinstance(c, (tuple, list)):
                    for item in c:
                        if isinstance(item, str):
                            m = getattr(model, item, None)
                            if isinstance(m, nn.Module):
                                mods.append(m)
                        elif isinstance(item, nn.Module):
                            mods.append(item)
            except Exception:
                pass

        # Heuristic fallbacks.
        for attr in ("classifier", "fc", "head"):
            m = getattr(model, attr, None)
            if isinstance(m, nn.Module):
                mods.append(m)

        # Dedupe while preserving order.
        seen: set[int] = set()
        out: List[nn.Module] = []
        for m in mods:
            mid = id(m)
            if mid in seen:
                continue
            seen.add(mid)
            out.append(m)
        return out

    def _infer_last_block_module(model: nn.Module) -> Optional[nn.Module]:
        # Common timm patterns.
        for attr in ("blocks", "stages", "layers"):
            m = getattr(model, attr, None)
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                return m[-1]
            if isinstance(m, (list, tuple)) and len(m) > 0 and isinstance(m[-1], nn.Module):
                return m[-1]

        # ResNet-like.
        for attr in ("layer4", "stage4", "layer3"):
            m = getattr(model, attr, None)
            if isinstance(m, nn.Module):
                return m

        # MobileNet/EfficientNet-like.
        m = getattr(model, "features", None)
        if isinstance(m, nn.Sequential) and len(m) > 0:
            return m[-1]

        return None

    def _apply_tune_policy(model: nn.Module, tune: str) -> None:
        tune = str(tune)

        # Default: everything trainable.
        for p in model.parameters():
            p.requires_grad = True
        if tune == "all":
            return

        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad = False

        if tune == "bn":
            for mod in model.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    for p in mod.parameters(recurse=False):
                        p.requires_grad = True
            return

        # Head-only or last-block+head.
        head_mods = _infer_head_modules(model)
        for hm in head_mods:
            for p in hm.parameters():
                p.requires_grad = True

        if tune == "lastblock_head":
            last_block = _infer_last_block_module(model)
            if last_block is None:
                print("[tune] lastblock_head requested but last block not found; falling back to head-only")
            else:
                for p in last_block.parameters():
                    p.requires_grad = True

    _apply_tune_policy(student, str(args.tune))

    def _freeze_bn_running_stats(model: nn.Module) -> None:
        """Keep BatchNorm layers in eval mode to avoid drifting running stats.

        Important for conservative fine-tuning / domain adaptation on small buffers
        (e.g., webcam self-learning), where updating BN running_mean/var can cause
        large distribution shifts and harm generalization.
        """

        for mod in model.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.eval()

    def _nl_extract_penultimate_and_logits(
        model: nn.Module, x: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (logits, penultimate) if available; otherwise None.

        Uses timm-style forward_features/forward_head when present. The penultimate output
        is forced to be 2D (B, D).
        """

        if not hasattr(model, "forward_features"):
            return None

        forward_features = getattr(model, "forward_features")
        feats = forward_features(x)
        if hasattr(model, "forward_head"):
            forward_head = getattr(model, "forward_head")
            pre = forward_head(feats, pre_logits=True)
            logits = forward_head(feats, pre_logits=False)
        else:
            # Fallback: derive a pooled vector from forward_features output.
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if not isinstance(feats, torch.Tensor):
                return None
            if feats.ndim == 4:
                pre = feats.mean(dim=(2, 3))
            elif feats.ndim == 3:
                pre = feats.mean(dim=1)
            else:
                pre = feats.flatten(1)
            # Worst-case: compute logits via full forward (extra compute).
            logits = model(x)

        if isinstance(pre, torch.Tensor) and pre.ndim > 2:
            pre = pre.flatten(1)
        if not (isinstance(pre, torch.Tensor) and isinstance(logits, torch.Tensor)):
            return None
        return logits, pre

    def _lp_loss_batch(
        feats: torch.Tensor,
        y: torch.Tensor,
        *,
        k: int,
    ) -> Tuple[torch.Tensor, float]:
        """Compute the locality-preserving loss (LP loss) on a mini-batch.

        Paper form (mini-batch-friendly approximation):

            L_lp = (1 / (2n)) * sum_i || x_i - (1/k) * sum_{x in N_k{x_i}} x ||_2^2

        Here N_k{x_i} is the set of k-nearest neighbors within the same class.
        We approximate neighbors within the current batch.

        Returns: (lp_loss, included_frac)
        where included_frac is the fraction of batch samples with >=1 same-class neighbor.
        """

        if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
            raise ValueError("feats must be a 2D Tensor (B, D)")

        bs = int(feats.shape[0])
        if bs <= 1:
            z = feats.sum() * 0.0
            return z, 0.0

        k = int(k)
        if k < 1:
            z = feats.sum() * 0.0
            return z, 0.0

        feats_f = feats.float()
        y = y.long()

        total = torch.zeros((), device=feats.device, dtype=feats_f.dtype)
        n_total = 0

        # Process per class to keep neighbor search intra-class.
        for cls in torch.unique(y.detach()):
            mask = (y == cls)
            n = int(mask.sum().item())
            if n < 2:
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            f = feats_f.index_select(0, idx)  # (n, D)

            kk = min(k, n - 1)
            # Pairwise squared L2 distances.
            dist = torch.cdist(f, f, p=2).pow(2)
            # Exclude self from neighbor list.
            dist.fill_diagonal_(float("inf"))
            nn_idx = torch.topk(dist, k=kk, largest=False, dim=1).indices  # (n, kk)
            nn_mean = f[nn_idx].mean(dim=1)  # (n, D)

            diff = (f - nn_mean)
            total = total + diff.pow(2).sum(dim=1).sum()
            n_total += n

        if n_total == 0:
            z = feats.sum() * 0.0
            return z, 0.0

        lp = 0.5 * total / float(n_total)
        included_frac = float(n_total) / float(max(1, bs))
        return lp.to(feats.dtype), included_frac

    # NL state
    nl_kind = str(args.nl_kind)
    nl_gate_memory: Optional[AssociativeMemory] = None
    nl_proj: Optional[nn.Linear] = None
    nl_prototypes: Optional[torch.Tensor] = None
    nl_seen: Optional[torch.Tensor] = None

    if bool(args.use_nl) and nl_kind == "negl_gate":
        # Features: [entropy_norm, max_prob, margin, step_frac]
        nl_gate_memory = AssociativeMemory(hidden_dim=int(args.nl_hidden_dim), layers=int(args.nl_layers), input_dim=4).to(
            device
        )

    if bool(args.use_nl) and nl_kind == "proto":
        nl_embed = str(args.nl_embed)
        if nl_embed == "penultimate":
            # Infer penultimate feature dimension with a tiny dummy forward.
            student_was_training = student.training
            student.eval()
            with torch.no_grad():
                dummy = torch.zeros((1, 3, int(args.image_size), int(args.image_size)), device=device)
                out = _nl_extract_penultimate_and_logits(student, dummy)
                feat_dim = int(out[1].shape[1]) if out is not None else int(getattr(student, "num_features", 0) or 0)
            student.train(student_was_training)
            if feat_dim <= 0:
                feat_dim = int(len(CANONICAL_7))
        else:
            feat_dim = int(len(CANONICAL_7))
        nl_dim = int(args.nl_dim)
        if nl_dim < 1:
            raise SystemExit("--nl-dim must be >= 1")

        nl_proj = nn.Linear(feat_dim, nl_dim).to(device)
        nl_prototypes = torch.zeros((len(CANONICAL_7), nl_dim), device=device, dtype=torch.float32)
        nl_seen = torch.zeros((len(CANONICAL_7),), device=device, dtype=torch.int64)

    params = [p for p in student.parameters() if p.requires_grad]
    if nl_gate_memory is not None:
        params += list(nl_gate_memory.parameters())
    if nl_proj is not None:
        params += list(nl_proj.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = GradScaler("cuda", enabled=use_amp)

    global_step = 0
    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = -1

    def save_ckpt(path: Path, *, epoch: int) -> None:
        nl_ckpt: Optional[Dict[str, Any]] = None
        if bool(args.use_nl) and nl_kind == "proto" and nl_proj is not None and nl_prototypes is not None and nl_seen is not None:
            nl_ckpt = {
                "kind": "proto",
                "proj": nl_proj.state_dict(),
                "prototypes": nl_prototypes.detach().float().cpu(),
                "seen": nl_seen.detach().cpu(),
                "cfg": {
                    "embed": str(args.nl_embed),
                    "feat_dim": int(nl_proj.in_features),
                    "dim": int(args.nl_dim),
                    "momentum": float(args.nl_momentum),
                    "proto_gate": str(args.nl_proto_gate),
                    "consistency_thresh": float(args.nl_consistency_thresh),
                    "topk_frac": float(args.nl_topk_frac),
                    "weight": float(args.nl_weight),
                },
            }
        elif bool(args.use_nl) and nl_kind == "negl_gate" and nl_gate_memory is not None:
            nl_ckpt = {
                "kind": "negl_gate",
                "memory": nl_gate_memory.state_dict(),
                "cfg": {"hidden_dim": int(args.nl_hidden_dim), "layers": int(args.nl_layers)},
            }

        ckpt = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": student.state_dict(),
            # Backward-compat: legacy key used by NL(negl_gate)
            "nl_memory": (nl_gate_memory.state_dict() if nl_gate_memory is not None else None),
            # New structured NL checkpoint payload
            "nl": nl_ckpt,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "best": {"macro_f1": float(best_macro_f1), "epoch": int(best_epoch)},
        }
        torch.save(ckpt, path)

    # Resume
    ckpt_path: Optional[Path] = None
    if args.resume is not None:
        ckpt_path = args.resume
    else:
        auto = args.output_dir / "checkpoint_last.pt"
        if auto.exists():
            ckpt_path = auto

    if ckpt_path is not None and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        student.load_state_dict(ckpt.get("model", {}), strict=True)

        # NL resume
        ckpt_nl = ckpt.get("nl")
        if bool(args.use_nl) and isinstance(ckpt_nl, dict):
            if nl_kind == "proto" and str(ckpt_nl.get("kind")) == "proto":
                if nl_proj is not None and isinstance(ckpt_nl.get("proj"), dict):
                    try:
                        nl_proj.load_state_dict(ckpt_nl["proj"], strict=True)
                    except Exception as e:
                        print(f"[NL] Skipping proj resume due to shape mismatch: {e}")
                if nl_prototypes is not None and isinstance(ckpt_nl.get("prototypes"), torch.Tensor):
                    pt = ckpt_nl["prototypes"].detach().float()
                    if pt.shape == nl_prototypes.shape:
                        nl_prototypes.copy_(pt.to(device))
                if nl_seen is not None and isinstance(ckpt_nl.get("seen"), torch.Tensor):
                    se = ckpt_nl["seen"].detach().to(torch.int64)
                    if se.shape == nl_seen.shape:
                        nl_seen.copy_(se.to(device))
            elif nl_kind == "negl_gate" and str(ckpt_nl.get("kind")) == "negl_gate":
                if nl_gate_memory is not None and isinstance(ckpt_nl.get("memory"), dict):
                    nl_gate_memory.load_state_dict(ckpt_nl["memory"], strict=True)

        # Backward-compat: legacy NL(negl_gate) checkpoints
        if bool(args.use_nl) and nl_kind == "negl_gate" and nl_gate_memory is not None and isinstance(ckpt.get("nl_memory"), dict):
            nl_gate_memory.load_state_dict(ckpt["nl_memory"], strict=True)

        # Optimizer/scaler resume:
        # When resuming across stages (e.g., KD -> DKD), we intentionally DO NOT restore
        # optimizer/scaler state. This avoids parameter-group mismatches and is closer to
        # the intended meaning of "continue training weights under a new loss".
        ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
        ckpt_mode = ckpt_args.get("mode")
        same_mode = (ckpt_mode is None) or (str(ckpt_mode) == str(args.mode))
        if str(args.tune) != "all":
            same_mode = False
            print(f"[resume] tune={args.tune}: skipping optimizer/scaler resume (fresh fine-tune optimizer)")
        if same_mode:
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except ValueError as e:
                    print(f"[resume] Skipping optimizer state due to incompatibility: {e}")
            if "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    print(f"[resume] Skipping scaler state due to incompatibility: {e}")
        else:
            print(f"[resume] Checkpoint mode={ckpt_mode} != current mode={args.mode}; skipping optimizer/scaler resume")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best = ckpt.get("best") or {}
        best_macro_f1 = float(best.get("macro_f1", best_macro_f1))
        best_epoch = int(best.get("epoch", best_epoch))
        print(f"Resumed from {ckpt_path} -> start_epoch={start_epoch}")

    # History
    history_path = args.output_dir / "history.json"
    history: List[Dict[str, object]] = []
    if history_path.exists():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = [r for r in loaded if isinstance(r, dict)]
        except Exception:
            pass

    total_steps = int(args.epochs) * max(1, len(train_dl))
    warmup_steps = int(args.warmup_epochs) * max(1, len(train_dl))

    def eval_student() -> Dict[str, object]:
        student.eval()
        all_logits: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        with torch.no_grad():
            for bi, batch in enumerate(val_dl):
                x, y, _src = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(autocast_device, enabled=use_amp):
                    logits = student(x)
                all_logits.append(logits.detach().float().cpu())
                all_y.append(y.detach().cpu())
                if args.max_val_batches and (bi + 1) >= int(args.max_val_batches):
                    break
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        raw = metrics_from_logits(logits, y, num_classes=len(CANONICAL_7))

        t_star = fit_temperature(logits, y, init_t=1.2)
        scaled_logits = logits / float(t_star)
        scaled = metrics_from_logits(scaled_logits, y, num_classes=len(CANONICAL_7))

        calib = {
            "mode": "global",
            "global_temperature": float(t_star),
        }
        (args.output_dir / "calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")

        rel = {
            "raw": raw,
            "temperature_scaled": {"mode": "global", "global_temperature": float(t_star), **scaled},
        }
        (args.output_dir / "reliabilitymetrics.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
        return {"raw": raw, "temperature_scaled": scaled, "t_star": float(t_star)}

    # Training loop
    for epoch in range(start_epoch, int(args.epochs)):
        student.train()
        if str(args.tune) != "all":
            _freeze_bn_running_stats(student)
        epoch_loss = 0.0
        epoch_negl = 0.0
        epoch_negl_applied = 0.0
        epoch_negl_entropy = 0.0
        epoch_nl_gate = 0.0
        epoch_nl_gate_applied = 0.0
        epoch_nl_proto_loss = 0.0
        epoch_nl_proto_applied = 0.0
        epoch_nl_proto_sim = 0.0
        epoch_lp_loss = 0.0
        epoch_lp_included = 0.0
        t_epoch = time.time()

        for step, batch in enumerate(train_dl):
            meta = None
            if return_meta:
                x, y, _src, rel_path, meta = batch
            else:
                x, y, _src, rel_path = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # LR schedule
            lr = lr_for_step(global_step, total_steps=total_steps, base_lr=float(args.lr), warmup_steps=warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast(autocast_device, enabled=use_amp):
                nl_penultimate: Optional[torch.Tensor] = None
                lp_penultimate: Optional[torch.Tensor] = None

                need_penultimate = (
                    (bool(args.use_nl) and nl_kind == "proto" and str(args.nl_embed) == "penultimate" and nl_proj is not None)
                    or (float(args.lp_weight) != 0.0 and str(args.lp_embed) == "penultimate")
                )
                if need_penultimate:
                    out = _nl_extract_penultimate_and_logits(student, x)
                    if out is not None:
                        student_logits, lp_penultimate = out
                        # Reuse for NL(proto) if requested.
                        if bool(args.use_nl) and nl_kind == "proto" and str(args.nl_embed) == "penultimate" and nl_proj is not None:
                            nl_penultimate = lp_penultimate
                    else:
                        student_logits = student(x)
                else:
                    student_logits = student(x)
                if meta is not None and bool(args.manifest_use_weights):
                    ce_per = F.cross_entropy(student_logits, y, reduction="none")
                    w = meta["weight"].to(device=device, dtype=ce_per.dtype)
                    w = w.clamp_min(0.0)
                    denom = w.sum().clamp_min(1e-12)
                    ce = (ce_per * w).sum() / denom
                else:
                    ce = F.cross_entropy(student_logits, y)

                loss = ce
                if args.mode != "ce":
                    # Lookup teacher logits by image_path.
                    assert path_to_i is not None
                    assert teacher_logits_cpu is not None
                    idxs = [path_to_i[str(p)] for p in rel_path]
                    t_logits = teacher_logits_cpu[idxs].to(device, non_blocking=True)
                    t = float(args.temperature)
                    if args.mode == "kd":
                        distill = kd_kl(student_logits, t_logits, t=t)
                    else:
                        distill = dkd_loss(
                            student_logits,
                            t_logits,
                            y,
                            t=t,
                            beta=float(args.beta),
                        )

                    # Match report style: (1-α) CE + α * T^2 * distill
                    alpha = float(args.alpha)
                    loss = (1.0 - alpha) * ce + alpha * (t * t) * distill

                negl_loss_val = None
                negl_applied_frac = None
                negl_entropy_mean = None
                nl_gate_mean = None
                nl_gate_applied_mean = None

                nl_proto_loss_val = None
                nl_proto_applied_frac = None
                nl_proto_sim_mean = None

                lp_loss_val = None
                lp_included_frac = None

                # NL(proto): prototype memory with momentum smoothing + consistency gating.
                if bool(args.use_nl) and nl_kind == "proto" and nl_proj is not None and nl_prototypes is not None and nl_seen is not None:
                    nl_w = float(args.nl_weight)
                    if nl_w != 0.0:
                        rep = nl_penultimate if nl_penultimate is not None else student_logits.float()
                        z = nl_proj(rep.float())
                        z = F.normalize(z, dim=1, eps=1e-6)

                        proto_y = nl_prototypes[y].to(z.dtype)
                        proto_y = F.normalize(proto_y, dim=1, eps=1e-6)
                        sim = F.cosine_similarity(z, proto_y, dim=1).clamp(-1.0, 1.0)
                        incons = (1.0 - sim)

                        gate = str(args.nl_proto_gate)
                        if gate == "topk":
                            frac = float(args.nl_topk_frac)
                            frac = max(0.0, min(1.0, frac))
                            bs = int(incons.shape[0])
                            k = int(math.ceil(frac * float(bs)))
                            if k <= 0:
                                nl_apply = torch.zeros_like(incons, dtype=z.dtype)
                            else:
                                k = min(bs, k)
                                with torch.no_grad():
                                    idx = torch.topk(incons.detach(), k=k, largest=True, sorted=False).indices
                                nl_apply = torch.zeros_like(incons, dtype=z.dtype)
                                nl_apply[idx] = 1.0
                        else:
                            thr = float(args.nl_consistency_thresh)
                            thr = max(0.0, min(2.0, thr))
                            nl_apply = (incons >= thr).to(z.dtype)

                        nl_loss = (nl_apply * incons).mean()
                        loss = loss + nl_w * nl_loss

                        nl_proto_loss_val = float(nl_loss.detach().cpu().item())
                        nl_proto_applied_frac = float(nl_apply.detach().mean().cpu().item())
                        nl_proto_sim_mean = float(sim.detach().mean().cpu().item())

                        # Momentum update prototypes using batch embeddings.
                        with torch.no_grad():
                            m = float(args.nl_momentum)
                            m = max(0.0, min(1.0, m))
                            for cls in torch.unique(y.detach()):
                                cls_i = int(cls.item())
                                mask_c = (y == cls)
                                if not bool(mask_c.any()):
                                    continue
                                mean_z = z.detach()[mask_c].mean(dim=0)
                                mean_z = F.normalize(mean_z, dim=0, eps=1e-6)
                                if int(nl_seen[cls_i].item()) == 0:
                                    nl_prototypes[cls_i].copy_(mean_z.to(nl_prototypes.dtype))
                                else:
                                    nl_prototypes[cls_i].mul_(m).add_((1.0 - m) * mean_z.to(nl_prototypes.dtype))
                                nl_seen[cls_i] += int(mask_c.sum().item())

                # LP loss (Paper #5 Track A): supervised locality preserving loss in feature space.
                lp_w = float(args.lp_weight)
                if lp_w != 0.0:
                    rep: torch.Tensor
                    if str(args.lp_embed) == "penultimate":
                        rep = lp_penultimate if lp_penultimate is not None else student_logits
                    else:
                        rep = student_logits
                    if rep.ndim > 2:
                        rep = rep.flatten(1)
                    lp_loss, lp_frac = _lp_loss_batch(rep, y, k=int(args.lp_k))
                    loss = loss + lp_w * lp_loss
                    lp_loss_val = float(lp_loss.detach().cpu().item())
                    lp_included_frac = float(lp_frac)
                if bool(args.use_negl):
                    c = int(student_logits.shape[1])
                    bs = int(y.shape[0])

                    # Select subset by ratio.
                    ratio = float(args.negl_ratio)
                    ratio = max(0.0, min(1.0, ratio))
                    apply_mask = (torch.rand((bs,), device=device) < ratio)

                    use_manifest_neg = bool(args.manifest_use_neg_label) and (meta is not None)
                    if use_manifest_neg:
                        neg_y_raw = meta["neg_y"].to(device=device)
                        has_neg = (neg_y_raw >= 0)
                        # Replace missing neg_y with a safe value; weight will be forced to 0 by apply_mask.
                        neg_y = torch.where(has_neg, neg_y_raw, torch.zeros_like(neg_y_raw))
                        apply_mask = apply_mask & has_neg
                    else:
                        # Sample a complementary label (uniform wrong class).
                        neg_y = torch.randint(0, c - 1, (bs,), device=device)
                        neg_y = neg_y + (neg_y >= y).to(neg_y.dtype)

                    # Optional entropy gate: apply NegL only for uncertain predictions.
                    if str(args.negl_gate) == "entropy":
                        with torch.no_grad():
                            p = F.softmax(student_logits.detach().float(), dim=1).clamp_min(1e-12)
                            ent = -(p * p.log()).sum(dim=1)
                            ent_norm = ent / float(math.log(max(2, c)))
                            thr = float(args.negl_entropy_thresh)
                            thr = max(0.0, min(1.0, thr))
                            apply_mask = apply_mask & (ent_norm >= thr)
                            negl_entropy_mean = float(ent_norm.mean().detach().cpu().item())

                    w = apply_mask.to(student_logits.dtype)
                    if nl_gate_memory is not None:
                        # Learn a per-sample gate to weight NegL.
                        with torch.no_grad():
                            p = F.softmax(student_logits.detach().float(), dim=1).clamp_min(1e-12)
                            p_sorted, _ = p.sort(dim=1, descending=True)
                            max_prob = p_sorted[:, 0]
                            margin = p_sorted[:, 0] - p_sorted[:, 1]
                            ent = -(p * p.log()).sum(dim=1)
                            ent_norm = ent / float(math.log(max(2, c)))
                            step_frac = torch.full(
                                (bs,),
                                float(global_step) / float(max(1, total_steps)),
                                device=device,
                                dtype=p.dtype,
                            )
                            feats = torch.stack([ent_norm, max_prob, margin, step_frac], dim=1)

                        gate = nl_gate_memory(feats).squeeze(1).to(student_logits.dtype)
                        w = w * gate

                        nl_gate_mean = float(gate.detach().mean().cpu().item())
                        if apply_mask.any():
                            nl_gate_applied_mean = float(gate.detach()[apply_mask].mean().cpu().item())
                        else:
                            nl_gate_applied_mean = 0.0

                    negl = complementary_negative_loss(student_logits, neg_y, weight=w)
                    lam = float(args.negl_weight)
                    if lam != 0.0:
                        loss = loss + lam * negl

                    negl_loss_val = float(negl.detach().cpu().item())
                    negl_applied_frac = float(apply_mask.to(torch.float32).mean().detach().cpu().item())
                    if nl_gate_mean is not None:
                        nl_gate_mean = float(nl_gate_mean)
                    if nl_gate_applied_mean is not None:
                        nl_gate_applied_mean = float(nl_gate_applied_mean)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu().item())
            if negl_loss_val is not None:
                epoch_negl += float(negl_loss_val)
            if negl_applied_frac is not None:
                epoch_negl_applied += float(negl_applied_frac)
            if negl_entropy_mean is not None:
                epoch_negl_entropy += float(negl_entropy_mean)
            if nl_gate_mean is not None:
                epoch_nl_gate += float(nl_gate_mean)
            if nl_gate_applied_mean is not None:
                epoch_nl_gate_applied += float(nl_gate_applied_mean)
            if nl_proto_loss_val is not None:
                epoch_nl_proto_loss += float(nl_proto_loss_val)
            if nl_proto_applied_frac is not None:
                epoch_nl_proto_applied += float(nl_proto_applied_frac)
            if nl_proto_sim_mean is not None:
                epoch_nl_proto_sim += float(nl_proto_sim_mean)
            if lp_loss_val is not None:
                epoch_lp_loss += float(lp_loss_val)
            if lp_included_frac is not None:
                epoch_lp_included += float(lp_included_frac)
            global_step += 1

        epoch_sec = time.time() - t_epoch

        # Save last checkpoint every epoch
        save_ckpt(args.output_dir / "checkpoint_last.pt", epoch=epoch)

        rec: Dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(epoch_loss / max(1, len(train_dl))),
            "epoch_sec": float(epoch_sec),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if bool(args.use_negl):
            rec["negl"] = {
                "weight": float(args.negl_weight),
                "ratio": float(args.negl_ratio),
                "gate": str(args.negl_gate),
                "entropy_thresh": float(args.negl_entropy_thresh),
                "train_negl_loss": float(epoch_negl / max(1, len(train_dl))),
                "applied_frac": float(epoch_negl_applied / max(1, len(train_dl))),
                "entropy_mean": float(epoch_negl_entropy / max(1, len(train_dl))) if epoch_negl_entropy > 0 else None,
            }

        if bool(args.use_nl):
            if nl_kind == "negl_gate" and nl_gate_memory is not None:
                rec["nl"] = {
                    "enabled": True,
                    "kind": "negl_gate",
                    "hidden_dim": int(args.nl_hidden_dim),
                    "layers": int(args.nl_layers),
                    "gate_mean": float(epoch_nl_gate / max(1, len(train_dl))),
                    "gate_applied_mean": float(epoch_nl_gate_applied / max(1, len(train_dl))),
                }
            elif nl_kind == "proto" and nl_proj is not None:
                rec["nl"] = {
                    "enabled": True,
                    "kind": "proto",
                    "embed": str(args.nl_embed),
                    "dim": int(args.nl_dim),
                    "momentum": float(args.nl_momentum),
                    "proto_gate": str(args.nl_proto_gate),
                    "consistency_thresh": float(args.nl_consistency_thresh),
                    "topk_frac": float(args.nl_topk_frac),
                    "weight": float(args.nl_weight),
                    "train_nl_loss": float(epoch_nl_proto_loss / max(1, len(train_dl))),
                    "applied_frac": float(epoch_nl_proto_applied / max(1, len(train_dl))),
                    "sim_mean": float(epoch_nl_proto_sim / max(1, len(train_dl))),
                }

        if float(args.lp_weight) != 0.0:
            rec["lp"] = {
                "enabled": True,
                "weight": float(args.lp_weight),
                "k": int(args.lp_k),
                "embed": str(args.lp_embed),
                "train_lp_loss": float(epoch_lp_loss / max(1, len(train_dl))),
                "included_frac": float(epoch_lp_included / max(1, len(train_dl))),
            }

        if int(args.eval_every) and ((epoch + 1) % int(args.eval_every) == 0):
            eval_payload = eval_student()
            rec["val"] = eval_payload
            macro_f1 = float(eval_payload["raw"]["macro_f1"])  # type: ignore[index]
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = int(epoch)
                save_ckpt(args.output_dir / "best.pt", epoch=epoch)

        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"epoch {epoch:03d} | loss {rec['train_loss']:.4f} | lr {rec['lr']:.2e} | epoch_sec {epoch_sec:.1f}"
        )

    print(f"Done. Output: {args.output_dir}")

    # Optional: produce standard gate artifacts (eval-only + ExpW) by running the existing eval script.
    if bool(args.post_eval):
        ckpt = args.output_dir / "best.pt"
        if not ckpt.exists():
            ckpt = args.output_dir / "checkpoint_last.pt"

        post_eval_results: List[Dict[str, object]] = []
        eval_jobs = [
            {"name": "eval_only", "manifest": Path(args.post_eval_evalonly_manifest)},
            {"name": "expw", "manifest": Path(args.post_eval_expw_manifest)},
        ]

        for job in eval_jobs:
            manifest = Path(job["manifest"])  # type: ignore[arg-type]
            if not manifest.exists():
                post_eval_results.append({"name": job["name"], "ok": False, "error": f"manifest not found: {manifest}"})
                continue

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_student_checkpoint.py"),
                "--checkpoint",
                str(ckpt),
                "--eval-manifest",
                str(manifest),
                "--eval-split",
                "test",
                "--eval-data-root",
                str(args.data_root),
                "--batch-size",
                str(int(args.batch_size)),
                "--num-workers",
                str(int(args.num_workers)),
                "--seed",
                str(int(args.seed)),
            ]

            r = subprocess.run(cmd, capture_output=True, text=True)
            rec_job: Dict[str, object] = {
                "name": job["name"],
                "manifest": str(manifest),
                "checkpoint": str(ckpt),
                "returncode": int(r.returncode),
            }
            if r.returncode == 0:
                # Try parse the JSON printed by eval_student_checkpoint.py.
                out_dir = None
                try:
                    last = (r.stdout or "").strip().splitlines()[-1]
                    payload = json.loads(last)
                    if isinstance(payload, dict) and "out_dir" in payload:
                        out_dir = payload.get("out_dir")
                        rec_job["out_dir"] = out_dir
                        rec_job["raw"] = payload.get("raw")
                        rec_job["ts"] = payload.get("ts")
                except Exception:
                    pass
                rec_job["ok"] = True
            else:
                rec_job["ok"] = False
                rec_job["stderr"] = (r.stderr or "")[:4000]

            post_eval_results.append(rec_job)

        (args.output_dir / "post_eval.json").write_text(json.dumps(post_eval_results, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
