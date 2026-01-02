from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
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

    return importlib.import_module("scripts.train_teacher")


def _load_teacher(ckpt_path: Path, *, device: torch.device) -> Tuple[object, Dict[str, object], Dict[str, object]]:
    """Load teacher checkpoint saved by scripts/train_teacher.py.

    Returns: (model, meta, ckpt_args)
    """
    teacher_mod = _dynamic_import_train_teacher()

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    ckpt_args = dict(ckpt.get("args") or {})
    model_name = str(ckpt_args.get("model") or "resnet18")
    embed_dim = int(ckpt_args.get("embed_dim") or 256)
    arc_s = float(ckpt_args.get("arcface_s") or 30.0)
    arc_m = float(ckpt_args.get("arcface_m") or 0.35)

    model = teacher_mod.TeacherNet(
        model_name=model_name,
        num_classes=len(CANONICAL_7),
        embed_dim=embed_dim,
        arc_s=arc_s,
        arc_m=arc_m,
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model", {}), strict=True)
    model.eval().to(device)

    meta = {
        "ckpt": str(ckpt_path),
        "model": model_name,
        "embed_dim": embed_dim,
        "arcface_s": arc_s,
        "arcface_m": arc_m,
        "ckpt_epoch": ckpt.get("epoch"),
    }
    return model, meta, ckpt_args


@torch.no_grad()
def _infer_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model.forward_infer(x)


def _ensemble_logits(
    logits_list: List[torch.Tensor],
    weights: List[float],
    *,
    ensemble_space: str,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, str]:
    if ensemble_space == "logit":
        out = torch.zeros_like(logits_list[0].float())
        for l, w in zip(logits_list, weights):
            out = out + float(w) * l.float()
        return out, "avg_logits"

    # prob space: mix softmax probabilities, then store log(p)
    probs = None
    for l, w in zip(logits_list, weights):
        p = F.softmax(l.float(), dim=1)
        if probs is None:
            probs = float(w) * p
        else:
            probs = probs + float(w) * p
    assert probs is not None
    probs = torch.clamp(probs, min=eps)
    return torch.log(probs), "log_probs_from_prob_ensemble"


def _metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, object]:
    y = y.long()
    probs = F.softmax(logits, dim=1)

    pred = logits.argmax(dim=1)
    correct = (pred == y).to(torch.float32)
    acc = float(correct.mean().item())

    k = len(CANONICAL_7)
    idx = (y * k + pred).to(torch.int64)
    cm = torch.bincount(idx, minlength=k * k).reshape(k, k)

    per_f1: Dict[str, float] = {}
    f1s: List[float] = []
    for i in range(k):
        tp = float(cm[i, i].item())
        fp = float(cm[:, i].sum().item() - tp)
        fn = float(cm[i, :].sum().item() - tp)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        per_f1[CANONICAL_7[i]] = float(f1)
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / max(1, len(f1s)))
    nll = float(F.cross_entropy(logits, y).item())

    # ECE (15 bins)
    conf, _ = probs.max(dim=1)
    bins = torch.linspace(0, 1, 16, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(15):
        lo = bins[i]
        hi = bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            prop = in_bin.to(torch.float32).mean()
            acc_bin = correct[in_bin].mean()
            avg_conf = conf[in_bin].mean()
            ece = ece + prop * (avg_conf - acc_bin).abs()

    y_onehot = F.one_hot(y, num_classes=len(CANONICAL_7)).to(probs.dtype)
    brier = float(((probs - y_onehot) ** 2).sum(dim=1).mean().item())

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_f1,
        "nll": float(nll),
        "ece": float(ece.item()),
        "brier": float(brier),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export multi-teacher ensemble softlabels aligned to a manifest split, with metadata + metrics. "
            "Supports 2+ teachers (e.g., RN18+B3+CNXT)."
        )
    )
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
    ap.add_argument(
        "--teacher",
        type=Path,
        action="append",
        required=True,
        help="Teacher checkpoint (.pt). Repeat for multiple teachers.",
    )
    ap.add_argument(
        "--weight",
        type=float,
        action="append",
        required=True,
        help="Weight for corresponding --teacher. Repeat (same count as --teacher).",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "Training_data_cleaned",
        help=(
            "Root to resolve relative image_path entries. "
            "Most manifests in this repo store paths relative to Training_data_cleaned/."
        ),
    )
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--use-clahe", action="store_true")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument(
        "--ensemble-space",
        type=str,
        default="prob",
        choices=["prob", "logit"],
        help=(
            "How to ensemble teachers. "
            "'prob' mixes softmax probabilities then stores log(p). "
            "'logit' mixes raw logits directly."
        ),
    )
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Storage dtype")
    ap.add_argument(
        "--skip-path-verify",
        action="store_true",
        help=(
            "Skip the expensive pre-filter that checks every file exists on disk when building splits. "
            "Use this for very large manifests; missing files (if any) will error later during loading."
        ),
    )

    args = ap.parse_args()

    teachers = list(args.teacher or [])
    weights = [float(w) for w in (args.weight or [])]
    if len(teachers) != len(weights):
        raise SystemExit(f"Need same count for --teacher and --weight (got {len(teachers)} vs {len(weights)})")
    if len(teachers) < 2:
        raise SystemExit("Provide at least 2 teachers")

    s = float(sum(weights))
    if not (s > 0):
        raise SystemExit("Weights must sum to > 0")
    weights = [w / s for w in weights]

    out_dir: Path = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    teacher_mod = _dynamic_import_train_teacher()

    models: List[object] = []
    metas: List[Dict[str, object]] = []
    ckpt_args_list: List[Dict[str, object]] = []
    for t in teachers:
        model, meta, ckpt_args = _load_teacher(t, device=device)
        models.append(model)
        metas.append(meta)
        ckpt_args_list.append(ckpt_args)

    for t, ckpt_args in zip(teachers, ckpt_args_list):
        img = ckpt_args.get("image_size")
        if img is not None and int(img) != int(args.image_size):
            print(f"[WARN] teacher ckpt {t} image_size={img} but --image-size={args.image_size}")

    manifest_hash = _sha256_file(args.manifest)

    rows_all = read_manifest(args.manifest)
    train_rows, val_rows, test_rows = build_splits(
        rows_all,
        out_root=args.data_root,
        verify_paths=not bool(args.skip_path_verify),
    )
    if args.split == "train":
        rows = train_rows
    elif args.split == "val":
        rows = val_rows
    elif args.split == "test":
        rows = test_rows
    else:
        rows = train_rows + val_rows + test_rows

    if not rows:
        raise SystemExit(
            "No rows selected for export.\n"
            f"manifest={args.manifest}\n"
            f"split={args.split}\n"
        )

    print(f"Selected {len(rows)} rows for split='{args.split}'")
    print(f"Writing to: {out_dir}")
    if args.skip_path_verify:
        print("[INFO] --skip-path-verify enabled (missing files will error during loading)")

    transform = teacher_mod.build_transforms(
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

    logits_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    path_list: List[str] = []
    src_list: List[str] = []

    t0 = time.time()
    for batch in dl:
        x, y, src, rel_path = batch
        x = x.to(device, non_blocking=True)

        per_teacher_logits = [_infer_logits(m, x).detach() for m in models]
        logits, logits_kind = _ensemble_logits(per_teacher_logits, weights, ensemble_space=str(args.ensemble_space))
        logits_list.append(logits.cpu())

        y_list.append(y.detach().cpu())
        path_list.extend([str(p) for p in rel_path])
        src_list.extend([str(s_) for s_ in src])

    logits_all = torch.cat(logits_list, dim=0).float()
    y_all = torch.cat(y_list, dim=0)

    if args.dtype == "float16":
        logits_out = logits_all.to(torch.float16)
    else:
        logits_out = logits_all.to(torch.float32)

    # Artifacts (compatible with scripts/diagnose_alignment.py and student pipeline)
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
        "ensemble": {
            "weights": [float(w) for w in weights],
            "teachers": metas,
            "logits_kind": logits_kind,
            "ensemble_space": str(args.ensemble_space),
        },
        "note": "Softlabels are aligned by row order of build_splits()+split selection (shuffle=False).",
    }
    (out_dir / "alignmentreport.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")

    import numpy as np

    np.savez_compressed(
        out_dir / "softlabels.npz",
        logits=logits_out.numpy(),
        y=y_all.numpy().astype("uint8"),
    )

    with (out_dir / "softlabels_index.jsonl").open("w", encoding="utf-8") as f:
        for i, (p, s_, yy) in enumerate(zip(path_list, src_list, y_all.tolist())):
            f.write(json.dumps({"i": int(i), "image_path": p, "source": s_, "y": int(yy)}) + "\n")

    metrics = _metrics_from_logits(logits_all, y_all)
    (out_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"Exported ensemble logits for {len(rows)} rows in {dt:.1f}s")
    print("Teachers:")
    for w, m in zip(weights, metas):
        print(f"  w={w:.3f}  model={m.get('model')}  ckpt={m.get('ckpt')}")
    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
