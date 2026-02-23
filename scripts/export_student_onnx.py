from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


def _load_ckpt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _export(
    *,
    checkpoint: Path,
    out_onnx: Path,
    opset: int,
    dynamic_batch: bool,
) -> dict:
    ckpt = _load_ckpt(checkpoint)
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}

    model_name = str(ckpt_args.get("model") or "mobilenetv3_large_100")
    image_size = int(ckpt_args.get("image_size") or 224)

    try:
        import timm  # type: ignore
    except Exception as e:
        raise RuntimeError("timm is required to export student to ONNX") from e

    model = timm.create_model(model_name, pretrained=False, num_classes=len(CANONICAL_7))
    model.load_state_dict(ckpt.get("model", {}), strict=True)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    # Prefer the legacy exporter when available to avoid requiring extra deps.
    try:
        import inspect

        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    torch.onnx.export(model, dummy, str(out_onnx), **export_kwargs)

    # Optional verification if onnx is installed.
    ok = False
    err: Optional[str] = None
    try:
        import onnx  # type: ignore

        m = onnx.load(str(out_onnx))
        onnx.checker.check_model(m)
        ok = True
    except Exception as e:  # pragma: no cover
        err = str(e)

    meta = {
        "checkpoint": str(checkpoint),
        "model": model_name,
        "image_size": image_size,
        "onnx_path": str(out_onnx),
        "opset": int(opset),
        "dynamic_batch": bool(dynamic_batch),
        "onnx_check_ok": bool(ok),
        "onnx_check_error": err,
    }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a student checkpoint (.pt) to ONNX for CPU-friendly deployment.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--meta-out", type=Path, default=None, help="Optional JSON file to write export metadata")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    meta = _export(
        checkpoint=args.checkpoint,
        out_onnx=args.out,
        opset=int(args.opset),
        dynamic_batch=bool(args.dynamic_batch),
    )

    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
