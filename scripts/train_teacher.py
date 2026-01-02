from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import (  # noqa: E402
    CANONICAL_7,
    LABEL_TO_INDEX,
    ManifestImageDataset,
    build_splits,
    read_manifest,
)
from src.fer.utils.device import get_best_device  # noqa: E402


try:
    import timm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "timm is required for teacher training. Ensure it's installed in the active .venv."
    ) from e


try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("numpy is required for teacher training.") from e


try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None, timeout_sec: int = 10) -> Optional[str]:
    try:
        p = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception:
        return None
    out = (p.stdout or "").strip()
    return out or None


def _git_commit(repo_root: Path) -> Optional[str]:
    return _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)


def _pip_freeze() -> Optional[List[str]]:
    out = _run_cmd([sys.executable, "-m", "pip", "freeze"], timeout_sec=60)
    if not out:
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines or None


def _write_environment_snapshot(*, output_dir: Path, repo_root: Path, device_info: object) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " ").strip(),
        },
        "git": {
            "commit": _git_commit(repo_root),
        },
        "packages": {
            "pip_freeze": _pip_freeze(),
        },
        "torch": {
            "version": getattr(torch, "__version__", None),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        },
        "device": {
            "backend": getattr(device_info, "backend", None),
            "detail": getattr(device_info, "detail", None),
        },
    }
    try:
        (output_dir / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort only.
        pass


def _write_run_lock(output_dir: Path, *, args: argparse.Namespace) -> Path:
    """Create a small lock file to indicate an active training/export run.

    This prevents helper scripts from deleting an output directory while
    checkpoints/ONNX exports are being written.
    """

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
        # Best-effort only.
        pass
    return lock_path


def _remove_run_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except TypeError:
        # Python <3.8 fallback (not expected in this repo).
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
    except Exception:
        pass


class CLAHETransform:
    def __init__(self, *, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not _HAS_CV2:
            return img
        arr = np.array(img)
        # RGB -> LAB
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=(self.tile_grid_size, self.tile_grid_size)
        )
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb2)


def build_transforms(
    *,
    image_size: int,
    train: bool,
    use_clahe: bool,
    clahe_clip: float,
    clahe_tile: int,
) -> T.Compose:
    ops: List[object] = []

    if train:
        ops.extend(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.3333)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            ]
        )
    else:
        resize = int(round(image_size * 1.15))
        ops.extend([T.Resize((resize, resize)), T.CenterCrop((image_size, image_size))])

    if use_clahe:
        ops.append(CLAHETransform(clip_limit=clahe_clip, tile_grid_size=clahe_tile))

    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return T.Compose(ops)  # type: ignore[arg-type]


def effective_number_weights(counts: List[int], *, beta: float = 0.9999) -> torch.Tensor:
    # Class-Balanced Loss weights: (1-beta)/(1-beta^n)
    w = []
    for n in counts:
        n = max(1, int(n))
        w_i = (1.0 - beta) / (1.0 - (beta**n))
        w.append(w_i)
    w = np.array(w, dtype=np.float64)
    w = w / w.sum() * len(w)
    return torch.tensor(w, dtype=torch.float32)


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: List[int],
        *,
        num_classes: int,
        batch_size: int,
        min_per_class: int = 2,
        seed: int = 1337,
        drop_last: bool = True,
    ) -> None:
        if batch_size < num_classes * min_per_class:
            raise ValueError(
                f"batch_size must be >= num_classes*min_per_class ({num_classes*min_per_class}), got {batch_size}"
            )
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.seed = seed
        self.drop_last = drop_last

        self.indices_by_class: List[List[int]] = [[] for _ in range(num_classes)]
        for idx, y in enumerate(labels):
            if 0 <= y < num_classes:
                self.indices_by_class[y].append(idx)

        for c in range(num_classes):
            if not self.indices_by_class[c]:
                raise ValueError(f"No samples found for class {c}.")

        self.num_batches = len(labels) // batch_size
        if not drop_last and len(labels) % batch_size:
            self.num_batches += 1

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed)
        all_classes = list(range(self.num_classes))

        for _ in range(self.num_batches):
            batch: List[int] = []

            # Ensure per-class minimum.
            for c in all_classes:
                pool = self.indices_by_class[c]
                for _k in range(self.min_per_class):
                    batch.append(pool[rng.randrange(len(pool))])

            # Fill the rest by sampling classes uniformly (then indices within class).
            remaining = self.batch_size - len(batch)
            for _ in range(remaining):
                c = all_classes[rng.randrange(self.num_classes)]
                pool = self.indices_by_class[c]
                batch.append(pool[rng.randrange(len(pool))])

            rng.shuffle(batch)
            yield batch


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, s: float = 30.0, m: float = 0.35) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, margin: float) -> torch.Tensor:
        # x: [B,D]
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if margin <= 0:
            return cosine * self.s

        # cos(theta + m) = cos* cos(m) - sin* sin(m)
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        sin_theta = torch.sqrt((1.0 - cosine * cosine).clamp_min(0.0))
        phi = cosine * cos_m - sin_theta * sin_m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1, 1), 1.0)
        out = cosine * (1.0 - one_hot) + phi * one_hot
        return out * self.s

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        # Inference-time logits: s * cos(theta) with normalized features/weights.
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, w)
        return cosine * self.s


class TeacherNet(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        embed_dim: int,
        arc_s: float,
        arc_m: float,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        # NOTE: For ViTs, forcing `global_pool='avg'` can change the module graph and cause
        # pretrained weight mismatches (e.g., unexpected `norm.*` keys). Keep default pooling.
        is_vit = model_name.startswith("vit_") or "deit" in model_name.lower()
        if is_vit:
            self.backbone = timm.create_model(model_name, pretrained=bool(pretrained), num_classes=0)
        else:
            self.backbone = timm.create_model(
                model_name, pretrained=bool(pretrained), num_classes=0, global_pool="avg"
            )

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError(f"Cannot determine num_features for timm model: {model_name}")

        self.proj = nn.Linear(int(feat_dim), int(embed_dim))
        self.bn = nn.BatchNorm1d(int(embed_dim))
        self.dropout = nn.Dropout(p=0.2)

        self.linear_head = nn.Linear(int(embed_dim), num_classes)
        self.arc_head = ArcMarginProduct(int(embed_dim), num_classes, s=arc_s, m=arc_m)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = feats.flatten(1)
        z = self.proj(feats)
        z = self.bn(z)
        z = F.silu(z)
        z = self.dropout(z)
        return z

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.linear_head(z)

    def forward_arcface(self, x: torch.Tensor, y: torch.Tensor, *, margin: float) -> torch.Tensor:
        z = self.forward_features(x)
        return self.arc_head(z, y, margin=margin)

    def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.arc_head.infer(z)


def margin_for_epoch(
    epoch: int,
    *,
    warmup_epochs_plain_logits: int,
    ramp_start: int,
    ramp_end: int,
    m_max: float,
) -> float:
    # During plain-logits warmup, we don't apply ArcFace at all.
    if epoch < warmup_epochs_plain_logits:
        return 0.0
    if epoch <= ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return m_max
    t = (epoch - ramp_start) / max(1, (ramp_end - ramp_start))
    return float(m_max * t)


def lr_for_step(step: int, *, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # cosine decay after warmup
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_for_step_min(step: int, *, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    lr = lr_for_step(step, total_steps=total_steps, base_lr=base_lr, warmup_steps=warmup_steps)
    return float(max(float(min_lr), float(lr)))


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    nll: float
    ece: float


def confusion_from_logits(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    idx = (y * num_classes + pred).to(torch.int64)
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    # cm[row=true, col=pred]
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


def evaluate(
    model: TeacherNet,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
    temperature: float = 1.0,
    warmup_plain_logits: bool,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    autocast_device = "cuda" if use_amp else "cpu"
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x, y, _src = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(autocast_device, enabled=use_amp):
                if warmup_plain_logits:
                    logits = model.forward_logits(x)
                else:
                    # IMPORTANT: evaluation uses inference-time logits (no ArcFace margin).
                    logits = model.forward_infer(x)
                logits = logits / float(temperature)

            all_logits.append(logits.detach().float().cpu())
            all_y.append(y.detach().cpu())

            if max_batches and (bi + 1) >= max_batches:
                break

    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, *, init_t: float = 1.2) -> float:
    # Optimize a single scalar T>0 to minimize NLL on val.
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
    # Clamp to sane range.
    return float(max(0.5, min(5.0, t)))


def fit_temperature_vector(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int, init_t: float = 1.2) -> List[float]:
    # Vector temperature scaling: one positive T per class applied elementwise to logits.
    logits = logits.float()
    y = y.long()

    init = float(init_t)
    log_t = torch.full((int(num_classes),), math.log(init), dtype=torch.float32, requires_grad=True)

    def nll() -> torch.Tensor:
        t = torch.exp(log_t).clamp(0.5, 5.0)
        scaled = logits / t.view(1, -1)
        return F.cross_entropy(scaled, y)

    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=75, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = nll()
        loss.backward()
        return loss

    opt.step(closure)
    t = torch.exp(log_t.detach()).clamp(0.5, 5.0).cpu().tolist()
    return [float(v) for v in t]


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int) -> Tuple[float, float, Dict[str, float], float, float]:
    y = y.long()
    probs = F.softmax(logits, dim=1)

    cm = confusion_from_logits(logits, y, num_classes=num_classes)
    correct = float(cm.diag().sum().item())
    total = float(cm.sum().item())
    acc = correct / max(1.0, total)

    macro_f1, per_f1 = f1_from_confusion(cm)

    nll = float(F.cross_entropy(logits, y).item())
    ece = expected_calibration_error(probs, y)

    return acc, macro_f1, per_f1, nll, ece


def _split_csv_list(value: str) -> List[str]:
    parts = [p.strip() for p in (value or "").split(",")]
    return [p for p in parts if p]


def filter_rows_by_source(
    rows: List[object],
    *,
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
) -> List[object]:
    include_set = {s.strip() for s in (include_sources or []) if s.strip()}
    exclude_set = {s.strip() for s in (exclude_sources or []) if s.strip()}

    if not include_set and not exclude_set:
        return rows

    out: List[object] = []
    for r in rows:
        src = getattr(r, "source", None)
        if src is None:
            continue
        if include_set and (src not in include_set):
            continue
        if exclude_set and (src in exclude_set):
            continue
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a teacher model with ArcFace protocol (reconstruction stage)")

    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest_hq_train.csv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
        help="Root used to resolve relative image paths from the manifest",
    )

    ap.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="timm model name (e.g., resnet18, resnet50, tf_efficientnet_b3, convnext_tiny, vit_base_patch16_224)",
    )
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--embed-dim", type=int, default=512)

    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Do not load timm pretrained weights. Note: when resuming or using --init-from, "
            "pretrained weights are automatically disabled to avoid unnecessary downloads."
        ),
    )

    # ArcFace protocol
    ap.add_argument("--arcface-m", type=float, default=0.35)
    ap.add_argument("--arcface-s", type=float, default=30.0)
    ap.add_argument("--plain-logits-warmup-epochs", type=int, default=5)
    ap.add_argument("--margin-ramp-start", type=int, default=5)
    ap.add_argument("--margin-ramp-end", type=int, default=15)

    # Optim + schedule
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--lr-warmup-epochs", type=int, default=2)
    ap.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum LR floor after warmup+cosine (default: 0.0). Use 1e-5 to match older reports.",
    )

    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a numbered checkpoint every N epochs (best.pt and checkpoint_last.pt are always maintained).",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint (.pt) to resume from. If omitted, will auto-resume from <output-dir>/checkpoint_last.pt when it exists.",
    )

    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt) but start a NEW run (epoch=0, fresh optimizer/scaler). "
            "Use this for Stage B finetuning. Mutually exclusive with --resume."
        ),
    )

    ap.add_argument(
        "--export-onnx-only",
        action="store_true",
        help="Load from --resume (or auto-resume) and export ONNX, then exit without training.",
    )

    ap.add_argument(
        "--skip-onnx-during-train",
        action="store_true",
        help="Do not export ONNX during training (exports can still be generated at end or via --export-onnx-only).",
    )

    ap.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation + temperature scaling every N epochs (default: 1).",
    )

    ap.add_argument(
        "--temperature-scaling",
        type=str,
        choices=["global", "vector", "none"],
        default="global",
        help=(
            "Temperature scaling mode for calibration on the validation split. "
            "global=single scalar T, vector=per-class vector T applied elementwise to logits, none=disable."
        ),
    )

    ap.add_argument(
        "--fixed-temperature",
        type=float,
        default=None,
        help=(
            "Optional fixed temperature T to use instead of fitting temperature scaling (only applies when "
            "--temperature-scaling=global). Useful to reproduce older reports that used a fixed T (e.g., 1.2)."
        ),
    )

    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip training. Load weights (resume/init-from/auto-resume) and recompute metrics+calibration, "
            "writing artifacts into --output-dir."
        ),
    )
    ap.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="When using --evaluate-only, optionally evaluate on a different manifest (e.g., classification_manifest_eval_only.csv).",
    )

    ap.add_argument(
        "--skip-env-snapshot",
        action="store_true",
        help="Skip writing environment.json (git commit + pip freeze).",
    )

    # Stage A/B convenience: filter sources by `source` column in manifest.
    ap.add_argument(
        "--include-sources",
        type=str,
        default="",
        help="Comma-separated list of sources to keep (e.g., 'ferplus,rafdb_basic,affectnet_full_balanced,expw_hq'). Empty = keep all.",
    )
    ap.add_argument(
        "--exclude-sources",
        type=str,
        default="",
        help="Comma-separated list of sources to drop (e.g., 'ferplus').",
    )

    # Data
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)

    # Balanced mini-batches + class-balanced loss
    ap.add_argument("--min-per-class", type=int, default=2, help="Ensure >=N samples/class per batch")
    ap.add_argument("--cb-beta", type=float, default=0.9999, help="Effective-number weighting beta")

    # Aug
    ap.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing (recommended)")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)

    # Smoke / limiting
    ap.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--max-val-batches", type=int, default=0, help="0 = full val")
    ap.add_argument("--smoke", action="store_true", help="Shortcut: 1 epoch + limited batches")

    # Output
    ap.add_argument("--output-dir", type=Path, default=None)

    args = ap.parse_args()

    if args.resume is not None and args.init_from is not None:
        raise SystemExit("ERROR: --resume and --init-from are mutually exclusive")

    if args.smoke:
        args.max_epochs = 1
        if args.max_train_batches == 0:
            args.max_train_batches = 50
        if args.max_val_batches == 0:
            args.max_val_batches = 25

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device_info = get_best_device(prefer="cuda")
    device = device_info.device
    use_amp = device_info.backend == "cuda"

    if device_info.backend == "cuda":
        # Safe speedups for NVIDIA GPUs.
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Prepare output
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model}_img{args.image_size}_seed{args.seed}_{stamp}"
        args.output_dir = Path("outputs") / "teachers" / run_name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Mark the output directory as "in use".
    lock_path = _write_run_lock(args.output_dir, args=args)
    atexit.register(_remove_run_lock, lock_path)

    if not bool(args.skip_env_snapshot):
        _write_environment_snapshot(output_dir=args.output_dir, repo_root=REPO_ROOT, device_info=device_info)

    # Determine whether we'll load weights from a checkpoint (resume/init-from/auto-resume).
    init_from_path: Optional[Path] = args.init_from
    ckpt_path: Optional[Path] = None
    if init_from_path is None:
        if args.resume is not None:
            ckpt_path = args.resume
        else:
            auto = args.output_dir / "checkpoint_last.pt"
            if auto.exists():
                ckpt_path = auto

    will_load_ckpt = bool(init_from_path is not None) or bool(ckpt_path is not None and ckpt_path.exists())

    # If we're loading from a checkpoint, don't fetch pretrained weights (avoid unnecessary HF/timm downloads).
    use_pretrained = (not bool(args.no_pretrained)) and (not will_load_ckpt)

    # Model
    model = TeacherNet(
        model_name=args.model,
        num_classes=len(CANONICAL_7),
        embed_dim=args.embed_dim,
        arc_s=args.arcface_s,
        arc_m=args.arcface_m,
        pretrained=use_pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if use_amp else "cpu"

    # Global step for LR schedule continuity (stored in checkpoints).
    global_step = 0

    def _save_checkpoint(path: Path, *, epoch: int, best_macro_f1: float, best_epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": int(global_step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "best": {"macro_f1": best_macro_f1, "epoch": best_epoch},
        }
        torch.save(ckpt, path)

    def _export_onnx(path: Path) -> None:
        model.eval()
        dummy = torch.zeros(1, 3, int(args.image_size), int(args.image_size), device=device)
        wrapper = nn.Module()

        # Attach the model as a submodule so ONNX exporter can trace it.
        wrapper.model = model

        def _forward(x: torch.Tensor) -> torch.Tensor:
            return wrapper.model.forward_infer(x)

        wrapper.forward = _forward  # type: ignore[assignment]

        # Force the legacy exporter when available to avoid requiring `onnxscript`.
        export_kwargs = dict(
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        try:
            import inspect

            if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                export_kwargs["dynamo"] = False
        except Exception:
            pass

        torch.onnx.export(wrapper, dummy, str(path), **export_kwargs)

    def _load_model_from_checkpoint(path: Path) -> None:
        try:
            ckpt_any = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_any = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt_any.get("model", {}), strict=True)

    def _load_checkpoint_any(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    # Resume if requested (or if checkpoint_last exists in output-dir)
    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = -1
    if init_from_path is not None:
        if not init_from_path.exists():
            raise SystemExit(f"ERROR: --init-from not found: {init_from_path}")
        ckpt_init = _load_checkpoint_any(init_from_path)
        model.load_state_dict(ckpt_init.get("model", {}), strict=True)
        start_epoch = 0
        best_macro_f1 = -1.0
        best_epoch = -1
        print(f"Initialized weights from: {init_from_path} (new run: epoch=0, fresh optimizer/scaler)")
    else:
        if ckpt_path is not None and ckpt_path.exists():
            ckpt = _load_checkpoint_any(ckpt_path)
            model.load_state_dict(ckpt.get("model", {}), strict=True)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best = ckpt.get("best") or {}
            best_macro_f1 = float(best.get("macro_f1", best_macro_f1))
            best_epoch = int(best.get("epoch", best_epoch))

            # Keep LR schedule continuity when resuming the SAME run.
            if "global_step" in ckpt:
                try:
                    global_step_ckpt = int(ckpt.get("global_step", 0))
                    global_step = max(0, global_step_ckpt)
                except Exception:
                    pass

            print(
                f"Resumed from: {ckpt_path} (next_epoch={start_epoch}, best_macro_f1={best_macro_f1:.4f} @ {best_epoch})"
            )

    if args.export_onnx_only:
        # Export from the currently loaded weights (resume/init-from/auto-resume already applied).
        try:
            _export_onnx(args.output_dir / "last.onnx")
            print("Exported:", args.output_dir / "last.onnx")
        except Exception as e:
            print("ERROR: ONNX export failed:", e)
            return 2
        return 0

    # Data split
    manifest_path_for_train = args.manifest
    rows_all = read_manifest(manifest_path_for_train)
    include_sources = _split_csv_list(args.include_sources)
    exclude_sources = _split_csv_list(args.exclude_sources)
    rows = filter_rows_by_source(rows_all, include_sources=include_sources, exclude_sources=exclude_sources)

    def _count_sources(rr: List[object]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rr:
            src = getattr(r, "source", None)
            if not src:
                src = "(missing)"
            out[str(src)] = int(out.get(str(src), 0) + 1)
        return out
    train_rows, val_rows, _test_rows = build_splits(
        rows,
        out_root=args.out_root,
        val_fraction_for_sources_without_val=args.val_fraction,
        seed=args.seed,
    )

    if not train_rows or not val_rows:
        raise RuntimeError(f"Not enough data. train={len(train_rows)} val={len(val_rows)}")

    train_tfm = build_transforms(
        image_size=args.image_size,
        train=True,
        use_clahe=bool(args.clahe),
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
    )
    val_tfm = build_transforms(
        image_size=args.image_size,
        train=False,
        use_clahe=bool(args.clahe),
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
    )

    train_ds = ManifestImageDataset(train_rows, out_root=args.out_root, transform=train_tfm)
    val_ds = ManifestImageDataset(val_rows, out_root=args.out_root, transform=val_tfm)

    # Labels for balanced batch sampler
    train_labels = [LABEL_TO_INDEX[r.label] for r in train_rows]

    batch_sampler = BalancedBatchSampler(
        train_labels,
        num_classes=len(CANONICAL_7),
        batch_size=args.batch_size,
        min_per_class=args.min_per_class,
        seed=args.seed,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device_info.backend == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_info.backend == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    class_counts = [0] * len(CANONICAL_7)
    for y in train_labels:
        class_counts[y] += 1
    class_w = effective_number_weights(class_counts, beta=args.cb_beta).to(device)

    # Precompute scheduler steps
    steps_per_epoch = len(train_loader)
    if args.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = max(1, steps_per_epoch * args.max_epochs)
    warmup_steps = int(round((args.lr_warmup_epochs * steps_per_epoch)))

    # Write alignment report (repro + integrity)
    class_counts_map = {CANONICAL_7[i]: int(class_counts[i]) for i in range(len(CANONICAL_7))}
    align = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "device": {"backend": device_info.backend, "detail": device_info.detail},
        "paths": {
            "manifest": str(manifest_path_for_train),
            "manifest_sha256": sha256_file(manifest_path_for_train) if manifest_path_for_train.exists() else None,
            "out_root": str(args.out_root),
            "output_dir": str(args.output_dir),
        },
        "seed": int(args.seed),
        "model": {
            "name": args.model,
            "pretrained": bool(use_pretrained),
            "image_size": int(args.image_size),
            "embed_dim": int(args.embed_dim),
        },
        "data": {
            "rows_total_before_filter": int(len(rows_all)),
            "rows_total_after_filter": int(len(rows)),
            "source_counts_after_filter": _count_sources(rows),
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "class_counts": class_counts_map,
            "source_filter": {
                "include_sources": include_sources,
                "exclude_sources": exclude_sources,
            },
        },
        "protocol": {
            "plain_logits_warmup_epochs": int(args.plain_logits_warmup_epochs),
            "margin_ramp_start": int(args.margin_ramp_start),
            "margin_ramp_end": int(args.margin_ramp_end),
            "arcface_m_max": float(args.arcface_m),
            "arcface_s": float(args.arcface_s),
        },
        "optim": {
            "name": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lr_warmup_epochs": float(args.lr_warmup_epochs),
            "min_lr": float(args.min_lr),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
        },
        "batching": {
            "batch_size": int(args.batch_size),
            "accum_steps": int(args.accum_steps),
            "min_per_class": int(args.min_per_class),
            "cb_beta": float(args.cb_beta),
        },
        "aug": {
            "clahe": bool(args.clahe),
            "clahe_clip": float(args.clahe_clip),
            "clahe_tile": int(args.clahe_tile),
        },
        "calibration": {
            "eval_every": int(args.eval_every),
            "temperature_scaling": str(args.temperature_scaling),
        },
    }

    # Preserve original init_from across resume runs.
    # On a resumed run, `init_from_path` can be None (because weights come from the resume checkpoint),
    # but we still want alignmentreport.json to reflect the original Stage A -> Stage B initialization.
    preserved_init_from: str | None = None
    existing_align_path = args.output_dir / "alignmentreport.json"
    if existing_align_path.exists():
        try:
            existing_align = json.loads(existing_align_path.read_text(encoding="utf-8"))
            if isinstance(existing_align, dict):
                existing_init = existing_align.get("init")
                if isinstance(existing_init, dict):
                    existing_init_from = existing_init.get("init_from")
                    if isinstance(existing_init_from, str) and existing_init_from.strip():
                        preserved_init_from = existing_init_from
        except Exception:
            preserved_init_from = None

    align["init"] = {
        "init_from": str(init_from_path) if init_from_path is not None else preserved_init_from,
        "resume": str(ckpt_path) if ckpt_path is not None else None,
    }
    (args.output_dir / "alignmentreport.json").write_text(json.dumps(align, indent=2), encoding="utf-8")

    # Evaluate-only mode (no training loop)
    if bool(args.evaluate_only):
        eval_manifest = args.eval_manifest if args.eval_manifest is not None else manifest_path_for_train
        rows_eval_all = read_manifest(eval_manifest)
        rows_eval = filter_rows_by_source(rows_eval_all, include_sources=include_sources, exclude_sources=exclude_sources)
        _train_rows_e, val_rows_e, test_rows_e = build_splits(
            rows_eval,
            out_root=args.out_root,
            val_fraction_for_sources_without_val=args.val_fraction,
            seed=args.seed,
        )
        eval_rows = test_rows_e if test_rows_e else val_rows_e
        if not eval_rows:
            raise RuntimeError(f"Not enough eval data. eval_rows={len(eval_rows)}")

        eval_tfm = build_transforms(
            image_size=args.image_size,
            train=False,
            use_clahe=bool(args.clahe),
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
        )
        eval_ds = ManifestImageDataset(eval_rows, out_root=args.out_root, transform=eval_tfm)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device_info.backend == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        logits, y = evaluate(
            model,
            eval_loader,
            device=device,
            use_amp=use_amp,
            max_batches=args.max_val_batches,
            temperature=1.0,
            warmup_plain_logits=False,
            margin=0.0,
        )

        acc, macro_f1, per_f1, nll, ece = metrics_from_logits(logits, y, num_classes=len(CANONICAL_7))

        cal_mode = str(args.temperature_scaling)
        t_star: float = 1.0
        t_vec: Optional[List[float]] = None
        logits_scaled = logits

        if cal_mode == "global":
            if args.fixed_temperature is not None:
                t_star = float(args.fixed_temperature)
            else:
                t_star = fit_temperature(logits, y, init_t=1.2)
            logits_scaled = logits / float(t_star)
        elif cal_mode == "vector":
            t_vec = fit_temperature_vector(logits, y, num_classes=len(CANONICAL_7), init_t=1.2)
            t_tensor = torch.tensor(t_vec, dtype=logits.dtype).view(1, -1)
            logits_scaled = logits / t_tensor
        elif cal_mode == "none":
            pass

        acc_t, macro_f1_t, per_f1_t, nll_t, ece_t = metrics_from_logits(logits_scaled, y, num_classes=len(CANONICAL_7))

        reliability = {
            "epoch": int(start_epoch) - 1,
            "mode": "evaluate_only",
            "eval_manifest": str(eval_manifest),
            "eval_rows": int(len(eval_rows)),
            "raw": {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_f1, "nll": nll, "ece": ece},
            "temperature_scaled": {
                "mode": cal_mode,
                "global_temperature": float(t_star),
                "temperature_vector": t_vec,
                "accuracy": acc_t,
                "macro_f1": macro_f1_t,
                "per_class_f1": per_f1_t,
                "nll": nll_t,
                "ece": ece_t,
            },
        }
        (args.output_dir / "reliabilitymetrics.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")

        calibration = {
            "epoch": int(start_epoch) - 1,
            "mode": cal_mode,
            "global_temperature": float(t_star),
            "temperature_vector": t_vec,
            "note": "Calibration fitted on eval split (evaluate-only mode).",
        }
        (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        return 0

    t0 = time.time()

    # Preserve prior training history on resume.
    # Previously, resuming from checkpoint_last.pt would reset history=[] and overwrite history.json,
    # making it look like earlier epochs were lost (even though checkpoints were still present).
    history: List[Dict[str, object]] = []
    history_path = args.output_dir / "history.json"
    if history_path.exists():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                # Avoid duplicating epochs if resuming after an interrupted write.
                keep: List[Dict[str, object]] = []
                for rec in loaded:
                    if not isinstance(rec, dict):
                        continue
                    ep = rec.get("epoch")
                    try:
                        ep_i = int(ep)  # type: ignore[arg-type]
                    except Exception:
                        continue
                    if ep_i < int(start_epoch):
                        keep.append(rec)
                history = keep
        except Exception:
            # Best-effort only; do not block training if history is unreadable.
            history = []
    if global_step == 0 and int(start_epoch) > 0:
        global_step = int(start_epoch) * int(steps_per_epoch)

    last_eval: Optional[Dict[str, object]] = None

    for epoch in range(start_epoch, int(args.max_epochs)):
        epoch_t0 = time.time()
        warmup_plain = epoch < int(args.plain_logits_warmup_epochs)
        m_epoch = margin_for_epoch(
            epoch,
            warmup_epochs_plain_logits=int(args.plain_logits_warmup_epochs),
            ramp_start=int(args.margin_ramp_start),
            ramp_end=int(args.margin_ramp_end),
            m_max=float(args.arcface_m),
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        for bi, batch in enumerate(train_loader):
            if args.max_train_batches and (bi + 1) > int(args.max_train_batches):
                break

            x, y, _src = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr_now = lr_for_step(
                global_step,
                total_steps=int(total_steps),
                base_lr=float(args.lr),
                warmup_steps=int(warmup_steps),
            )
            lr_now = lr_for_step_min(
                global_step,
                total_steps=int(total_steps),
                base_lr=float(args.lr),
                warmup_steps=int(warmup_steps),
                min_lr=float(args.min_lr),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            with autocast(autocast_device, enabled=use_amp):
                if warmup_plain:
                    logits = model.forward_logits(x)
                else:
                    logits = model.forward_arcface(x, y, margin=m_epoch)
                loss = F.cross_entropy(logits, y, weight=class_w)
                loss = loss / max(1, int(args.accum_steps))

            scaler.scale(loss).backward()

            if ((bi + 1) % int(args.accum_steps)) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu())
            seen += 1
            global_step += 1

        train_loss = running_loss / max(1, seen)

        eval_every = max(1, int(args.eval_every))
        do_eval = (epoch == start_epoch) or (eval_every == 1) or ((epoch % eval_every) == 0) or (epoch == int(args.max_epochs) - 1)

        if do_eval:
            # Eval (raw)
            val_logits, val_y = evaluate(
                model,
                val_loader,
                device=device,
                use_amp=use_amp,
                max_batches=args.max_val_batches,
                temperature=1.0,
                warmup_plain_logits=warmup_plain,
                margin=m_epoch,
            )
            acc, macro_f1, per_f1, nll, ece = metrics_from_logits(val_logits, val_y, num_classes=len(CANONICAL_7))

            # Temperature scaling on val
            cal_mode = str(args.temperature_scaling)
            t_star = 1.0
            t_vec: Optional[List[float]] = None
            val_logits_t = val_logits

            if cal_mode == "global":
                if args.fixed_temperature is not None:
                    t_star = float(args.fixed_temperature)
                else:
                    t_star = fit_temperature(val_logits, val_y, init_t=1.2)
                val_logits_t = val_logits / float(t_star)
            elif cal_mode == "vector":
                t_vec = fit_temperature_vector(val_logits, val_y, num_classes=len(CANONICAL_7), init_t=1.2)
                t_tensor = torch.tensor(t_vec, dtype=val_logits.dtype).view(1, -1)
                val_logits_t = val_logits / t_tensor
            elif cal_mode == "none":
                pass

            acc_t, macro_f1_t, per_f1_t, nll_t, ece_t = metrics_from_logits(
                val_logits_t, val_y, num_classes=len(CANONICAL_7)
            )

            last_eval = {
                "val": {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "per_class_f1": per_f1,
                    "nll": nll,
                    "ece": ece,
                },
                "calibration": {
                    "mode": cal_mode,
                    "temperature": float(t_star),
                    "temperature_vector": t_vec,
                    "val_scaled": {
                        "accuracy": acc_t,
                        "macro_f1": macro_f1_t,
                        "per_class_f1": per_f1_t,
                        "nll": nll_t,
                        "ece": ece_t,
                    },
                },
            }
        else:
            # Skip eval to save time; reuse last eval snapshot if available.
            if last_eval is None:
                # Safety: first epoch always evaluates.
                raise RuntimeError("Internal error: last_eval is None while do_eval is False")
            acc = float(last_eval["val"]["accuracy"])  # type: ignore[index]
            macro_f1 = float(last_eval["val"]["macro_f1"])  # type: ignore[index]
            per_f1 = dict(last_eval["val"]["per_class_f1"])  # type: ignore[index]
            nll = float(last_eval["val"]["nll"])  # type: ignore[index]
            ece = float(last_eval["val"]["ece"])  # type: ignore[index]
            t_star = float(last_eval["calibration"].get("temperature", 1.0))  # type: ignore[index]
            t_vec = last_eval["calibration"].get("temperature_vector")  # type: ignore[index]
            acc_t = float(last_eval["calibration"]["val_scaled"]["accuracy"])  # type: ignore[index]
            macro_f1_t = float(last_eval["calibration"]["val_scaled"]["macro_f1"])  # type: ignore[index]
            per_f1_t = dict(last_eval["calibration"]["val_scaled"]["per_class_f1"])  # type: ignore[index]
            nll_t = float(last_eval["calibration"]["val_scaled"]["nll"])  # type: ignore[index]
            ece_t = float(last_eval["calibration"]["val_scaled"]["ece"])  # type: ignore[index]

        epoch_sec = time.time() - epoch_t0
        total_sec = time.time() - t0

        epoch_rec = {
            "epoch": epoch,
            "train": {
                "loss": train_loss,
                "warmup_plain_logits": warmup_plain,
                "arcface_margin": m_epoch,
            },
            "val": {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "per_class_f1": per_f1,
                "nll": nll,
                "ece": ece,
            },
            "calibration": {
                "mode": str(args.temperature_scaling),
                "temperature": float(t_star),
                "temperature_vector": t_vec,
                "val_scaled": {
                    "accuracy": acc_t,
                    "macro_f1": macro_f1_t,
                    "per_class_f1": per_f1_t,
                    "nll": nll_t,
                    "ece": ece_t,
                },
            },
            "lr": float(optimizer.param_groups[0]["lr"]),
            "timing": {"epoch_sec": epoch_sec, "total_sec": total_sec},
            "eval": {"ran": bool(do_eval), "every": int(eval_every)},
        }
        history.append(epoch_rec)

        # Save artifacts each epoch (small, but helps reproducibility)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        reliability = {
            "epoch": epoch,
            "raw": {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_f1, "nll": nll, "ece": ece},
            "temperature_scaled": {
                "mode": str(args.temperature_scaling),
                "global_temperature": float(t_star),
                "temperature_vector": t_vec,
                "accuracy": acc_t,
                "macro_f1": macro_f1_t,
                "per_class_f1": per_f1_t,
                "nll": nll_t,
                "ece": ece_t,
            },
        }
        (args.output_dir / "reliabilitymetrics.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")

        calibration = {
            "epoch": epoch,
            "mode": str(args.temperature_scaling),
            "global_temperature": float(t_star),
            "temperature_vector": t_vec,
            "note": "Calibration fitted on validation NLL (global) or vector-NLL (vector).",
        }
        (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

        # Checkpointing
        _save_checkpoint(args.output_dir / "checkpoint_last.pt", epoch=epoch, best_macro_f1=best_macro_f1, best_epoch=best_epoch)
        if args.checkpoint_every > 0 and ((epoch + 1) % int(args.checkpoint_every) == 0):
            _save_checkpoint(
                args.output_dir / f"checkpoint_epoch{epoch:03d}.pt",
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                best_epoch=best_epoch,
            )

        # Best model tracking (only meaningful when eval ran)
        if do_eval and (macro_f1 > best_macro_f1):
            best_macro_f1 = float(macro_f1)
            best_epoch = int(epoch)
            _save_checkpoint(args.output_dir / "best.pt", epoch=epoch, best_macro_f1=best_macro_f1, best_epoch=best_epoch)
            if not args.skip_onnx_during_train:
                try:
                    _export_onnx(args.output_dir / "best.onnx")
                except Exception as e:
                    print("WARN: ONNX export failed for best model:", e)

        # Always keep a last.onnx for convenience (optionally skipped during training)
        if not args.skip_onnx_during_train:
            try:
                _export_onnx(args.output_dir / "last.onnx")
            except Exception as e:
                print("WARN: ONNX export failed for last model:", e)

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {acc:.4f} | val_macroF1 {macro_f1:.4f} | ece {ece:.4f} | T* {t_star:.3f} | img {args.image_size} | epoch_sec {epoch_sec:.1f} | total_sec {total_sec:.1f}"
        )

        if args.smoke:
            break

    # Ensure ONNX artifacts exist at the end (even if skipped during training).
    try:
        _export_onnx(args.output_dir / "last.onnx")
    except Exception as e:
        print("WARN: Final ONNX export failed for last model:", e)

    best_ckpt = args.output_dir / "best.pt"
    if best_ckpt.exists():
        try:
            _load_model_from_checkpoint(best_ckpt)
            _export_onnx(args.output_dir / "best.onnx")
        except Exception as e:
            print("WARN: Final ONNX export failed for best model:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
