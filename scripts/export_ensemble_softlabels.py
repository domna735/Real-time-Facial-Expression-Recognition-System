from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    resolve_image_path,
)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _warn_missing_rows(
    rows_all,
    *,
    out_root: Path,
    split: str,
) -> None:
    """Print a warning if the requested split contains many missing files on disk.

    This is a common source of confusingly low metrics: build_splits() drops missing
    images, so you may end up evaluating on a much smaller (and different) subset.
    """

    if split not in {"train", "val", "test"}:
        return

    raw = [r for r in rows_all if (r.split if r.split in {"train", "val", "test"} else "train") == split]
    if not raw:
        return

    missing_by_source: Dict[str, int] = {}
    missing = 0
    for r in raw:
        try:
            ok = resolve_image_path(out_root, r.image_path).exists()
        except Exception:
            ok = False
        if not ok:
            missing += 1
            missing_by_source[r.source] = int(missing_by_source.get(r.source, 0) + 1)

    if missing <= 0:
        return

    # Only warn if this is meaningfully large; otherwise it can be too noisy.
    frac = missing / max(1, len(raw))
    if missing < 50 and frac < 0.01:
        return

    top = sorted(missing_by_source.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_str = ", ".join([f"{k}={v}" for k, v in top])
    print(
        f"[WARN] {missing}/{len(raw)} rows in split='{split}' are missing on disk and will be dropped. "
        f"Top missing sources: {top_str}"
    )


def _dynamic_import_train_teacher():
    # Windows DataLoader workers spawn new interpreters.
    # Custom transforms must be importable by module name for pickling.
    import importlib

    return importlib.import_module("scripts.train_teacher")


def _load_teacher(ckpt_path: Path, *, device: torch.device):
    """Load teacher checkpoint saved by scripts/train_teacher.py.

    Returns: (model, meta, ckpt_args)
    """
    train_teacher = _dynamic_import_train_teacher()

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    args = dict(ckpt.get("args") or {})
    model_name = str(args.get("model") or "resnet18")
    embed_dim = int(args.get("embed_dim") or 256)
    arc_s = float(args.get("arcface_s") or 30.0)
    arc_m = float(args.get("arcface_m") or 0.35)

    model = train_teacher.TeacherNet(
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
    return model, meta, args


@torch.no_grad()
def _infer_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model.forward_infer(x)


def _ensemble_logits_from_probs(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    wa: float,
    wb: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    # Ensemble at probability level, then convert back to logits by log(p).
    # This makes softmax(logits_out) == probs_ensemble (up to fp error).
    pa = F.softmax(logits_a.float(), dim=1)
    pb = F.softmax(logits_b.float(), dim=1)
    p = wa * pa + wb * pb
    p = torch.clamp(p, min=eps)
    return torch.log(p)


def _ensemble_logits_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    wa: float,
    wb: float,
) -> torch.Tensor:
    # Simple weighted average in logit space.
    # This preserves the interpretation “teacher_logits / T” used in KD.
    return wa * logits_a.float() + wb * logits_b.float()


def _metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, object]:
    # Reuse the same definitions as scripts/compute_reliability.py (kept local to avoid import issues).
    y = y.long()
    probs = F.softmax(logits, dim=1)

    pred = logits.argmax(dim=1)
    correct = (pred == y).to(torch.float32)
    acc = float(correct.mean().item())

    # Confusion matrix
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

    # ECE
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

    # Brier
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
        description="Export ensemble softlabels (RN18+B3 etc.) aligned to a manifest split, with metadata + metrics."
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
    ap.add_argument("--teacher-a", type=Path, required=True, help="Teacher A checkpoint (.pt)")
    ap.add_argument("--teacher-b", type=Path, required=True, help="Teacher B checkpoint (.pt)")
    ap.add_argument("--weight-a", type=float, required=True)
    ap.add_argument("--weight-b", type=float, required=True)
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
            "How to ensemble two teachers. "
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

    wa = float(args.weight_a)
    wb = float(args.weight_b)
    s = wa + wb
    if not (s > 0):
        raise SystemExit("Weights must sum to > 0")
    wa /= s
    wb /= s

    out_dir: Path = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    teacher_mod = _dynamic_import_train_teacher()

    model_a, meta_a, ckpt_args_a = _load_teacher(args.teacher_a, device=device)
    model_b, meta_b, ckpt_args_b = _load_teacher(args.teacher_b, device=device)

    # Consistency checks
    img_a = ckpt_args_a.get("image_size")
    img_b = ckpt_args_b.get("image_size")
    if img_a is not None and int(img_a) != int(args.image_size):
        print(f"[WARN] teacher-a ckpt image_size={img_a} but --image-size={args.image_size}")
    if img_b is not None and int(img_b) != int(args.image_size):
        print(f"[WARN] teacher-b ckpt image_size={img_b} but --image-size={args.image_size}")

    manifest_hash = _sha256_file(args.manifest)

    rows_all = read_manifest(args.manifest)
    _warn_missing_rows(rows_all, out_root=args.data_root, split=str(args.split))
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
            f"split={args.split}\n\n"
            "Hint: try --split all, or use the eval-only manifest "
            "(Training_data_cleaned/classification_manifest_eval_only.csv) which contains val/test rows."
        )

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

        la = _infer_logits(model_a, x).detach()
        lb = _infer_logits(model_b, x).detach()

        if str(args.ensemble_space) == "logit":
            logits = _ensemble_logits_from_logits(la, lb, wa=wa, wb=wb).cpu()
            logits_kind = "avg_logits"
        else:
            # Ensemble on probabilities by default (more stable across architectures).
            logits = _ensemble_logits_from_probs(la, lb, wa=wa, wb=wb).cpu()
            logits_kind = "log_probs_from_prob_ensemble"

        logits_list.append(logits)
        y_list.append(y.detach().cpu())
        path_list.extend([str(p) for p in rel_path])
        src_list.extend([str(s_) for s_ in src])

    logits_all = torch.cat(logits_list, dim=0).float()
    y_all = torch.cat(y_list, dim=0)

    if args.dtype == "float16":
        logits_out = logits_all.to(torch.float16)
    else:
        logits_out = logits_all.to(torch.float32)

    # Write artifacts (compatible with scripts/diagnose_alignment.py and student pipeline).
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
            "weight_a": wa,
            "weight_b": wb,
            "teacher_a": meta_a,
            "teacher_b": meta_b,
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

    # Quick metrics on this split (raw; if you want temperature scaling, run scripts/compute_reliability.py)
    metrics = _metrics_from_logits(logits_all, y_all)
    (out_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"Exported ensemble logits for {len(rows)} rows in {dt:.1f}s")
    print(f"w_a={wa:.3f} ({meta_a.get('model')})  w_b={wb:.3f} ({meta_b.get('model')})")
    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
