from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Realtime inference runner for ArcFace-trained FER checkpoints. "
            "This is a thin wrapper around demo/realtime_demo.py (YuNet/DNN/Haar + smoothing + manual labeling)."
        )
    )
    ap.add_argument("--model-ckpt", type=Path, required=True, help="Teacher checkpoint (.pt)")
    ap.add_argument("--source", type=str, default="webcam", help="'webcam' or a video file path")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--detector", type=str, default="yunet", choices=["yunet", "dnn", "haar"])
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "demo" / "outputs")
    args = ap.parse_args()

    # Delegate to demo/realtime_demo.py to avoid duplicating UI logic.
    demo_path = REPO_ROOT / "demo" / "realtime_demo.py"
    if not demo_path.exists():
        raise SystemExit(f"Missing demo script: {demo_path}")

    import runpy

    sys.argv = [
        str(demo_path),
        "--source",
        str(args.source),
        "--camera-index",
        str(args.camera_index),
        "--detector",
        str(args.detector),
        "--model-ckpt",
        str(args.model_ckpt),
        "--output-dir",
        str(args.output_dir),
    ]

    runpy.run_path(str(demo_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
