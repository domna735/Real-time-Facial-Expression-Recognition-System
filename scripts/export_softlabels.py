from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import (  # noqa: E402
    CANONICAL_7,
    ManifestImageDataset,
    build_splits,
    read_manifest,
)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dynamic_import_train_teacher():
    import importlib

    # Ensure the module is importable by name for Windows DataLoader workers.
    return importlib.import_module("scripts.train_teacher")


def _load_teacher(ckpt_path: Path, *, device: torch.device):
    """Load teacher checkpoint saved by scripts/train_teacher.py.

    Returns: (model, meta)
    """
    mod = _dynamic_import_train_teacher()

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    args = dict(ckpt.get("args") or {})
    model_name = str(args.get("model") or "resnet18")
    embed_dim = int(args.get("embed_dim") or 256)
    arc_s = float(args.get("arcface_s") or 30.0)
    arc_m = float(args.get("arcface_m") or 0.35)

    model = mod.TeacherNet(
        model_name=model_name,
        num_classes=len(CANONICAL_7),
        embed_dim=embed_dim,
        arc_s=arc_s,
        arc_m=arc_m,
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model", {}), strict=True)
    model.eval()
    model.to(device)

    meta = {
        "ckpt": str(ckpt_path),
        "model": model_name,
        "embed_dim": embed_dim,
        "arcface_s": arc_s,
        "arcface_m": arc_m,
        "ckpt_epoch": ckpt.get("epoch"),
    }
    return model, meta


@torch.no_grad()
def _infer_logits(model, x: torch.Tensor) -> torch.Tensor:
    # Teacher inference uses ArcFace infer logits.
    return model.forward_infer(x)


def _pick_split(rows, split: str):
    split = (split or "train").lower()
    if split not in {"train", "val", "test", "all"}:
        raise SystemExit(f"Invalid --split: {split}")
    if split == "all":
        return rows
    return [r for r in rows if (r.split or "train").lower() == split]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export teacher softlabels (logits) aligned to a manifest split.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "Training_data_cleaned" / "classification_manifest_hq_train.csv",
        help="CSV manifest path.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "softlabels" / time.strftime("%Y%m%d_%H%M%S"),
        help="Output directory.",
    )
    ap.add_argument("--teacher-ckpt", type=Path, required=True, help="Teacher checkpoint (.pt) from scripts/train_teacher.py")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT, help="Root used to resolve relative image_path entries.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224, help="Input image size for transforms (must match teacher expectations).")
    ap.add_argument("--use-clahe", action="store_true", help="Apply CLAHE in preprocessing (match training if you used it).")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Storage dtype for logits.")

    args = ap.parse_args()

    out_dir: Path = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import teacher helpers (transforms) from train_teacher to stay consistent.
    train_teacher = _dynamic_import_train_teacher()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, model_meta = _load_teacher(args.teacher_ckpt, device=device)

    manifest_hash = _sha256_file(args.manifest)

    rows_all = read_manifest(args.manifest)
    train_rows, val_rows, test_rows = build_splits(rows_all, out_root=args.data_root)
    if args.split == "train":
        rows = train_rows
    elif args.split == "val":
        rows = val_rows
    elif args.split == "test":
        rows = test_rows
    else:
        rows = train_rows + val_rows + test_rows

    transform = train_teacher.build_transforms(
        image_size=int(args.image_size),
        train=False,
        use_clahe=bool(args.use_clahe),
        clahe_clip=float(args.clahe_clip),
        clahe_tile=int(args.clahe_tile),
    )

    ds = ManifestImageDataset(rows, out_root=args.data_root, transform=transform, return_path=True)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # Preallocate lists and save as NPZ at end.
    logits_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    path_list: List[str] = []
    src_list: List[str] = []

    t0 = time.time()
    for batch in dl:
        x, y, src, rel_path = batch
        x = x.to(device, non_blocking=True)
        logits = _infer_logits(model, x).detach().float().cpu()

        logits_list.append(logits)
        y_list.append(y.detach().cpu())
        # rel_path and src are lists/tuples of strings.
        path_list.extend([str(p) for p in rel_path])
        src_list.extend([str(s) for s in src])

    logits_all = torch.cat(logits_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    if args.dtype == "float16":
        logits_out = logits_all.to(torch.float16)
    else:
        logits_out = logits_all.to(torch.float32)

    # Write artifacts.
    (out_dir / "classorder.json").write_text(json.dumps(list(CANONICAL_7), indent=2), encoding="utf-8")
    (out_dir / "hash_manifest.json").write_text(
        json.dumps({"manifest": str(args.manifest), "sha256": manifest_hash}, indent=2),
        encoding="utf-8",
    )

    alignment = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "split": str(args.split),
        "rows": int(len(rows)),
        "image_size": int(args.image_size),
        "use_clahe": bool(args.use_clahe),
        "clahe_clip": float(args.clahe_clip),
        "clahe_tile": int(args.clahe_tile),
        "teacher": model_meta,
        "note": "Softlabels are aligned by row order of build_splits()+split selection (shuffle=False).",
    }
    (out_dir / "alignmentreport.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")

    # Save logits/labels and a sidecar index.
    import numpy as np

    np.savez_compressed(
        out_dir / "softlabels.npz",
        logits=logits_out.numpy(),
        y=y_all.numpy().astype("uint8"),
    )

    # Sidecar index (kept separate to avoid huge NPZ metadata overhead).
    with (out_dir / "softlabels_index.jsonl").open("w", encoding="utf-8") as f:
        for i, (p, s, yy) in enumerate(zip(path_list, src_list, y_all.tolist())):
            f.write(json.dumps({"i": int(i), "image_path": p, "source": s, "y": int(yy)}) + "\n")

    dt = time.time() - t0
    print(f"Exported {len(rows)} rows in {dt:.1f}s")
    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
