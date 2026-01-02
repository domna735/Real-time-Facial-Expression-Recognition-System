from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import (
    CANONICAL_7,
    ManifestImageDataset,
    build_splits,
    default_transform,
    read_manifest,
)
from src.fer.utils.device import get_best_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test: train RN18 using classification_manifest.csv")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("Training_data_cleaned") / "classification_manifest.csv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("Training_data_cleaned"),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=224, help="Training input size (e.g., 224/384/448)")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--val-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.05, help="Used only for sources without explicit val")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = Path("outputs") / f"smoke_rn18_{args.manifest.stem}"

    torch.manual_seed(args.seed)

    device_info = get_best_device(prefer="cuda")
    device = device_info.device
    use_amp = device_info.backend == "cuda"

    rows = read_manifest(args.manifest)
    train_rows, val_rows, _test_rows = build_splits(
        rows,
        out_root=args.out_root,
        val_fraction_for_sources_without_val=args.val_fraction,
        seed=args.seed,
    )

    if len(train_rows) == 0 or len(val_rows) == 0:
        raise RuntimeError(f"Not enough data. train={len(train_rows)} val={len(val_rows)}")

    tfm = default_transform(image_size=args.image_size)
    train_ds = ManifestImageDataset(train_rows, out_root=args.out_root, transform=tfm)
    val_ds = ManifestImageDataset(val_rows, out_root=args.out_root, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CANONICAL_7))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if use_amp else "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model.train()
    step = 0
    running_loss = 0.0

    for batch in train_loader:
        x, y, _src = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(autocast_device, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().cpu())
        step += 1
        if step >= args.max_steps:
            break

    train_loss = running_loss / max(1, step)

    # Quick val
    model.eval()
    correct = 0
    total = 0
    val_step = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y, _src = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(autocast_device, enabled=use_amp):
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().cpu())
            total += int(y.numel())
            val_step += 1
            if val_step >= args.val_steps:
                break

    val_acc = correct / max(1, total)
    elapsed = time.time() - t0

    out = {
        "device": {"backend": device_info.backend, "detail": device_info.detail},
        "manifest": str(args.manifest),
        "out_root": str(args.out_root),
        "split_policy": {
            "use_existing_val": True,
            "val_fraction_for_sources_without_val": args.val_fraction,
            "seed": args.seed,
        },
        "sizes": {"train": len(train_rows), "val": len(val_rows)},
        "smoke": {
            "max_steps": args.max_steps,
            "val_steps": args.val_steps,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        },
    }

    out_path = args.output_dir / "smoke_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("device:", device_info.backend, "|", device_info.detail)
    print("train/val sizes:", len(train_rows), len(val_rows))
    print("train_loss:", train_loss)
    print("val_acc (partial):", val_acc)
    print("elapsed_sec:", elapsed)
    print("results:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
