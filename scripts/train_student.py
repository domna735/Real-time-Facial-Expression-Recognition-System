from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)

    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-val-batches", type=int, default=0)

    args = ap.parse_args()

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

    train_ds = ManifestImageDataset(train_rows, out_root=args.data_root, transform=train_tf, return_path=True)
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

    train_dl = DataLoader(train_ds, **train_dl_kwargs)
    val_dl = DataLoader(val_ds, **val_dl_kwargs)

    # Student model
    student = timm.create_model(str(args.model), pretrained=True, num_classes=len(CANONICAL_7)).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = GradScaler("cuda", enabled=use_amp)

    global_step = 0
    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = -1

    def save_ckpt(path: Path, *, epoch: int) -> None:
        ckpt = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": student.state_dict(),
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
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
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
        epoch_loss = 0.0
        t_epoch = time.time()

        for step, batch in enumerate(train_dl):
            x, y, _src, rel_path = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # LR schedule
            lr = lr_for_step(global_step, total_steps=total_steps, base_lr=float(args.lr), warmup_steps=warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast(autocast_device, enabled=use_amp):
                student_logits = student(x)
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

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu().item())
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
