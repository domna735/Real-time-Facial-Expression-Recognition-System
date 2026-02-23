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
    ap.add_argument(
        "--model-kind",
        type=str,
        choices=["teacher", "student"],
        default="teacher",
        help="Checkpoint type to load.",
    )
    ap.add_argument(
        "--model-ckpt",
        type=Path,
        default=None,
        help=(
            "Checkpoint (.pt). If omitted: uses default teacher, or auto-picks best student from outputs/students/."
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "dml"],
        help="Device preference for model inference.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature scaling for student probabilities. If omitted uses run calibration.json when available.",
    )
    ap.add_argument(
        "--temperature-json",
        type=Path,
        default=None,
        help="Optional calibration JSON containing global_temperature.",
    )
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
        "--device",
        str(args.device),
        "--source",
        str(args.source),
        "--camera-index",
        str(args.camera_index),
        "--detector",
        str(args.detector),
        "--model-kind",
        str(args.model_kind),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.model_ckpt is not None:
        sys.argv += ["--model-ckpt", str(args.model_ckpt)]
    if args.temperature is not None:
        sys.argv += ["--temperature", str(args.temperature)]
    if args.temperature_json is not None:
        sys.argv += ["--temperature-json", str(args.temperature_json)]

    runpy.run_path(str(demo_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
