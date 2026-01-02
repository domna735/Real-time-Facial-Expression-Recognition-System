from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RN18_RUN_DIR = (
    REPO_ROOT / "outputs" / "teachers" / "RN18_resnet18_seed1337_stageA_img224"
)
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7  # noqa: E402


@dataclass
class Params:
    ema_alpha: float = 0.70
    hysteresis_delta: float = 0.08
    vote_window: int = 15
    vote_min_count: int = 8
    show_emo_ratio: bool = True


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_probably_text_pointer(data: bytes) -> bool:
    # Common failure modes when downloading from GitHub:
    # - Git LFS pointer file (tiny text) instead of actual binary
    # - HTML error page (rate limit / auth / 404)
    prefix = data[:200].lstrip()
    if not prefix:
        return True
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return True
    if prefix.startswith(b"<"):
        # crude but effective for HTML
        return True
    return False


def _download_one(url: str, dst: Path) -> None:
    import urllib.request

    _ensure_parent(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    tmp.replace(dst)


def _download_with_validation(path: Path, urls: List[str], *, min_bytes: int = 1024 * 200) -> None:
    # If the file exists but is clearly not a real binary model, re-download.
    if path.exists():
        try:
            head = path.read_bytes()[:512]
            if path.stat().st_size >= min_bytes and not _is_probably_text_pointer(head):
                return
        except Exception:
            pass

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            print(f"Downloading model: {url}")
            _download_one(url, path)
            head = path.read_bytes()[:512]
            size = path.stat().st_size
            if size < min_bytes or _is_probably_text_pointer(head):
                raise RuntimeError(
                    f"Downloaded file does not look like a real binary model (size={size} bytes)."
                )
            return
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Required file missing/invalid and all downloads failed:\n"
        f"  path: {path}\n"
        f"  tried URLs:\n    " + "\n    ".join(urls) + "\n"
        f"  last error: {last_err}"
    )


def _load_teacher_from_checkpoint(ckpt_path: Path):
    import torch

    # Dynamic import of scripts/train_teacher.py without requiring scripts/ to be a package.
    import importlib.util
    import sys as _sys

    train_py = REPO_ROOT / "scripts" / "train_teacher.py"
    spec = importlib.util.spec_from_file_location("train_teacher", str(train_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {train_py}")
    mod = importlib.util.module_from_spec(spec)
    # IMPORTANT: Ensure the module is registered before exec_module.
    # Python's dataclasses (3.11+) expects sys.modules[__module__] to exist
    # while decorating @dataclass classes.
    _sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}

    model_name = str(ckpt_args.get("model", "resnet18"))
    embed_dim = int(ckpt_args.get("embed_dim", 512))
    image_size = int(ckpt_args.get("image_size", 224))

    # Build model with pretrained=False (we're loading weights).
    model = mod.TeacherNet(
        model_name=model_name,
        num_classes=len(CANONICAL_7),
        embed_dim=embed_dim,
        arc_s=float(ckpt_args.get("arcface_s", 30.0)),
        arc_m=float(ckpt_args.get("arcface_m", 0.35)),
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model", {}), strict=True)
    model.eval()

    device_info = mod.get_best_device(prefer="cuda")
    device = device_info.device
    model = model.to(device)

    # Build eval transform aligned to training.
    from PIL import Image
    from torchvision import transforms as T

    resize = int(round(image_size * 1.15))
    tfm = T.Compose(
        [
            T.Resize((resize, resize)),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def infer(face_bgr) -> Tuple[List[float], List[float]]:
        # Returns (logits, probs)
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_t = model.forward_infer(x)
            probs_t = torch.softmax(logits_t, dim=1)
        logits = logits_t.squeeze(0).detach().float().cpu().tolist()
        probs = probs_t.squeeze(0).detach().float().cpu().tolist()
        return logits, probs

    meta = {
        "ckpt": str(ckpt_path),
        "model": model_name,
        "image_size": image_size,
        "device": str(device),
    }
    return infer, meta


class FaceDetector:
    def __init__(self, method: str, *, model_dir: Path) -> None:
        self.method = method
        self.model_dir = model_dir

        self._yunet = None
        self._dnn = None
        self._haar = None

        if method == "yunet":
            yunet_path = model_dir / "face_detection_yunet_2023mar.onnx"
            # NOTE: opencv_zoo stores models via Git LFS. raw.githubusercontent.com may
            # return a tiny Git LFS *pointer* file instead of the binary model.
            # We validate size/content and retry alternate GitHub URLs.
            _download_with_validation(
                yunet_path,
                urls=[
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=1",
                    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                ],
                min_bytes=1024 * 200,
            )
            try:
                # input size is set per-frame via setInputSize
                self._yunet = cv2.FaceDetectorYN_create(str(yunet_path), "", (320, 320))
            except cv2.error as e:
                size = yunet_path.stat().st_size if yunet_path.exists() else -1
                raise RuntimeError(
                    "Failed to load YuNet ONNX model with OpenCV DNN.\n"
                    f"Model path: {yunet_path} (size={size} bytes)\n"
                    f"OpenCV error: {e}\n\n"
                    "Workarounds:\n"
                    "- Use `--detector dnn` (Caffe SSD) or `--detector haar` for realtime webcam.\n"
                    "- Or install a different OpenCV build that supports this ONNX."
                )

        elif method == "dnn":
            proto = model_dir / "deploy.prototxt"
            weights = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            _download_with_validation(
                proto,
                urls=[
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                ],
                min_bytes=1024,
            )
            _download_with_validation(
                weights,
                urls=[
                    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                ],
                min_bytes=1024 * 100,
            )
            self._dnn = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

        elif method == "haar":
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade))
        else:
            raise ValueError(f"Unknown detector: {method}")

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]

        if self.method == "yunet":
            assert self._yunet is not None
            self._yunet.setInputSize((w, h))
            _ok, faces = self._yunet.detect(frame_bgr)
            out: List[Tuple[int, int, int, int]] = []
            if faces is None:
                return out
            for f in faces:
                x, y, bw, bh = f[:4]
                out.append((int(x), int(y), int(bw), int(bh)))
            return out

        if self.method == "dnn":
            assert self._dnn is not None
            blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self._dnn.setInput(blob)
            det = self._dnn.forward()
            out: List[Tuple[int, int, int, int]] = []
            for i in range(det.shape[2]):
                conf = float(det[0, 0, i, 2])
                if conf < 0.5:
                    continue
                x1 = int(det[0, 0, i, 3] * w)
                y1 = int(det[0, 0, i, 4] * h)
                x2 = int(det[0, 0, i, 5] * w)
                y2 = int(det[0, 0, i, 6] * h)
                out.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
            return out

        assert self._haar is not None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _largest_face(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def _crop_with_margin(frame_bgr, box: Tuple[int, int, int, int], margin: float = 0.15):
    x, y, w, h = box
    H, W = frame_bgr.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    return frame_bgr[y1:y2, x1:x2]


def _draw_label_bar(frame_bgr, *, manual_label_idx: Optional[int]) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    bar_h = 60
    y1 = h - bar_h
    y2 = h

    seg_w = max(1, w // len(CANONICAL_7))
    for i, name in enumerate(CANONICAL_7):
        x1 = i * seg_w
        x2 = w if i == len(CANONICAL_7) - 1 else (i + 1) * seg_w
        is_sel = manual_label_idx == i
        color = (60, 200, 60) if is_sel else (40, 40, 40)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)
        cv2.putText(
            frame_bgr,
            f"{i+1}:{name}",
            (x1 + 8, y1 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return (0, y1, w, bar_h)


def _apply_hysteresis(
    probs: List[float], current_idx: Optional[int], delta: float
) -> Optional[int]:
    if not probs:
        return current_idx

    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    if current_idx is None:
        return top_idx

    if top_idx == current_idx:
        return current_idx

    # Switch only if the new top is sufficiently better than the current.
    if probs[top_idx] >= probs[current_idx] + float(delta):
        return top_idx
    return current_idx


def _vote_smooth(labels: Deque[int], *, window: int, min_count: int) -> Optional[int]:
    if window <= 1:
        return labels[-1] if labels else None
    if not labels:
        return None
    c = Counter(labels)
    lab, cnt = c.most_common(1)[0]
    return int(lab) if cnt >= int(min_count) else None


def _write_per_class_correctness_summary(
    *,
    per_frame_csv: Path,
    out_csv: Path,
    classes: List[str],
) -> None:
    # Summarize per-class correctness using frames that have a manual label.
    # This is intentionally simple: accuracy = correct / frames_with_manual_and_pred.
    if not per_frame_csv.exists():
        return

    by_class = {
        name: {
            "frames_with_manual": 0,
            "frames_with_manual_and_pred": 0,
            "correct_frames": 0,
        }
        for name in classes
    }
    overall = {
        "frames_with_manual": 0,
        "frames_with_manual_and_pred": 0,
        "correct_frames": 0,
    }

    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual = (row.get("manual_label") or "").strip()
            if not manual:
                continue
            if manual not in by_class:
                continue

            pred = (row.get("pred_label") or "").strip()

            by_class[manual]["frames_with_manual"] += 1
            overall["frames_with_manual"] += 1

            # Only count correctness when the predictor is stable and returns a real label.
            if pred and pred != "(unstable)":
                by_class[manual]["frames_with_manual_and_pred"] += 1
                overall["frames_with_manual_and_pred"] += 1
                if pred == manual:
                    by_class[manual]["correct_frames"] += 1
                    overall["correct_frames"] += 1

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "frames_with_manual",
                "frames_with_manual_and_pred",
                "correct_frames",
                "accuracy",
            ],
        )
        w.writeheader()
        for name in classes:
            den = int(by_class[name]["frames_with_manual_and_pred"])
            num = int(by_class[name]["correct_frames"])
            acc = (num / den) if den > 0 else ""
            w.writerow(
                {
                    "label": name,
                    "frames_with_manual": int(by_class[name]["frames_with_manual"]),
                    "frames_with_manual_and_pred": den,
                    "correct_frames": num,
                    "accuracy": f"{acc:.6f}" if isinstance(acc, float) else "",
                }
            )

        den = int(overall["frames_with_manual_and_pred"])
        num = int(overall["correct_frames"])
        acc = (num / den) if den > 0 else ""
        w.writerow(
            {
                "label": "__overall__",
                "frames_with_manual": int(overall["frames_with_manual"]),
                "frames_with_manual_and_pred": den,
                "correct_frames": num,
                "accuracy": f"{acc:.6f}" if isinstance(acc, float) else "",
            }
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time FER demo with manual labeling + tunable smoothing.")
    ap.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="'webcam' or a video file path.",
    )
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument(
        "--detector",
        type=str,
        choices=["yunet", "dnn", "haar"],
        default="yunet",
        help="Face detector method.",
    )
    ap.add_argument(
        "--model-ckpt",
        type=Path,
        default=DEFAULT_RN18_RUN_DIR / "best.pt",
        help="Teacher checkpoint (.pt) for inference.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "demo" / "outputs" / time.strftime("%Y%m%d_%H%M%S"),
        help="Output directory for CSV artifacts.",
    )

    args = ap.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    infer, model_meta = _load_teacher_from_checkpoint(args.model_ckpt)

    model_dir = REPO_ROOT / "demo" / "models"
    detector = FaceDetector(args.detector, model_dir=model_dir)

    if args.source.lower() == "webcam":
        cap = cv2.VideoCapture(int(args.camera_index))
        input_name = f"webcam:{args.camera_index}"
    else:
        cap = cv2.VideoCapture(str(args.source))
        input_name = str(args.source)

    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {args.source}")

    params = Params()

    # Smoothing state
    ema_probs: Optional[List[float]] = None
    hyster_idx: Optional[int] = None
    votes: Deque[int] = deque(maxlen=params.vote_window)

    # Manual labeling state
    manual_idx: Optional[int] = None
    events: List[Dict[str, object]] = []
    current_event: Optional[Dict[str, object]] = None

    # CSV writers
    frames_csv = out_dir / "per_frame.csv"
    events_csv = out_dir / "events.csv"
    summary_csv = out_dir / "demoresultssummary.csv"
    per_class_csv = out_dir / "per_class_correctness.csv"
    thresholds_json = out_dir / "thresholds.json"

    with frames_csv.open("w", newline="", encoding="utf-8") as f_frames:
        w_frames = csv.DictWriter(
            f_frames,
            fieldnames=[
                "frame_index",
                "time_sec",
                "manual_label",
                "pred_label",
                *[f"prob_{name}" for name in CANONICAL_7],
                "detector",
                "model",
                "ckpt",
            ],
        )
        w_frames.writeheader()

        win = "FER Demo"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        bar_rect = (0, 0, 0, 0)

        def _set_manual(idx: Optional[int], frame_index: int, tsec: float) -> None:
            nonlocal manual_idx, current_event
            if idx == manual_idx:
                return

            # Close previous event.
            if current_event is not None:
                current_event["end_frame"] = frame_index - 1
                current_event["end_time_sec"] = tsec
                events.append(current_event)
                current_event = None

            manual_idx = idx

            # Open new event.
            if manual_idx is not None:
                current_event = {
                    "label": CANONICAL_7[manual_idx],
                    "start_frame": frame_index,
                    "start_time_sec": tsec,
                    "end_frame": frame_index,
                    "end_time_sec": tsec,
                    "source": "manual",
                }

        def _on_mouse(event, x, y, _flags, _param):
            nonlocal bar_rect
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            bx, by, bw, bh = bar_rect
            if not (bx <= x < bx + bw and by <= y < by + bh):
                return
            seg_w = max(1, bw // len(CANONICAL_7))
            idx = min(len(CANONICAL_7) - 1, x // seg_w)
            # frame_index/time_sec are captured in the main loop; we update via a sentinel.
            _on_mouse.pending = idx  # type: ignore[attr-defined]

        _on_mouse.pending = None  # type: ignore[attr-defined]
        cv2.setMouseCallback(win, _on_mouse)

        frame_index = 0
        t_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            tsec = now - t_start

            # Apply any pending click label.
            pending = getattr(_on_mouse, "pending")
            if pending is not None:
                _set_manual(int(pending), frame_index, tsec)
                _on_mouse.pending = None  # type: ignore[attr-defined]

            faces = detector.detect(frame)
            face_box = _largest_face(faces)

            probs = [0.0] * len(CANONICAL_7)
            pred_idx: Optional[int] = None

            if face_box is not None:
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 220), 2)
                crop = _crop_with_margin(frame, face_box)
                _logits, probs = infer(crop)

                # EMA
                if ema_probs is None:
                    ema_probs = list(probs)
                else:
                    a = float(params.ema_alpha)
                    ema_probs = [a * e + (1.0 - a) * p for e, p in zip(ema_probs, probs)]

                # Hysteresis
                hyster_idx = _apply_hysteresis(ema_probs, hyster_idx, params.hysteresis_delta)

                # Vote window
                votes = deque(votes, maxlen=params.vote_window)
                if hyster_idx is not None:
                    votes.append(int(hyster_idx))
                pred_idx = _vote_smooth(votes, window=params.vote_window, min_count=params.vote_min_count)

            pred_label = CANONICAL_7[pred_idx] if pred_idx is not None else "(unstable)"
            manual_label = CANONICAL_7[manual_idx] if manual_idx is not None else ""

            # UI overlays
            cv2.putText(
                frame,
                f"src={input_name} det={args.detector} pred={pred_label} manual={manual_label}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"EMA a={params.ema_alpha:.2f} hyst d={params.hysteresis_delta:.2f} vote={params.vote_window}/{params.vote_min_count} (o=overlay)",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            if params.show_emo_ratio and probs:
                base_y = 85
                for i, name in enumerate(CANONICAL_7):
                    cv2.putText(
                        frame,
                        f"{name[:3]}:{probs[i]:.2f}",
                        (10 + (i % 4) * 115, base_y + (i // 4) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

            bar_rect = _draw_label_bar(frame, manual_label_idx=manual_idx)

            cv2.imshow(win, frame)

            # Log per-frame
            row = {
                "frame_index": frame_index,
                "time_sec": f"{tsec:.6f}",
                "manual_label": manual_label,
                "pred_label": pred_label,
                **{f"prob_{name}": f"{float(probs[i]):.6f}" for i, name in enumerate(CANONICAL_7)},
                "detector": args.detector,
                "model": model_meta.get("model"),
                "ckpt": model_meta.get("ckpt"),
            }
            w_frames.writerow(row)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                params.show_emo_ratio = not params.show_emo_ratio
            if key == ord("c"):
                _set_manual(None, frame_index, tsec)

            # Manual labeling keys 1..7
            if ord("1") <= key <= ord("7"):
                idx = int(chr(key)) - 1
                _set_manual(idx, frame_index, tsec)

            # Parameter hotkeys
            if key == ord("["):
                params.ema_alpha = max(0.0, params.ema_alpha - 0.05)
            if key == ord("]"):
                params.ema_alpha = min(0.99, params.ema_alpha + 0.05)
            if key == ord("-"):
                params.hysteresis_delta = max(0.0, params.hysteresis_delta - 0.02)
            if key == ord("="):
                params.hysteresis_delta = min(0.5, params.hysteresis_delta + 0.02)
            if key == ord("v"):
                params.vote_window = max(1, params.vote_window - 2)
            if key == ord("b"):
                params.vote_window = min(101, params.vote_window + 2)
            if key == ord("n"):
                params.vote_min_count = max(1, params.vote_min_count - 1)
            if key == ord("m"):
                params.vote_min_count = min(params.vote_window, params.vote_min_count + 1)

            frame_index += 1

    # Finalize last event
    t_end = time.time() - t_start
    if current_event is not None:
        current_event["end_frame"] = max(0, frame_index - 1)
        current_event["end_time_sec"] = float(t_end)
        events.append(current_event)

    # Write events CSV
    with events_csv.open("w", newline="", encoding="utf-8") as f_events:
        w = csv.DictWriter(
            f_events,
            fieldnames=["label", "start_frame", "end_frame", "start_time_sec", "end_time_sec", "source"],
        )
        w.writeheader()
        for e in events:
            w.writerow(e)

    # thresholds.json (optional but helpful)
    thresholds_json.write_text(
        json.dumps(
            {
                "ema_alpha": params.ema_alpha,
                "hysteresis_delta": params.hysteresis_delta,
                "vote_window": params.vote_window,
                "vote_min_count": params.vote_min_count,
                "show_emo_ratio": params.show_emo_ratio,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # demoresultssummary.csv
    with summary_csv.open("w", newline="", encoding="utf-8") as f_sum:
        w = csv.DictWriter(
            f_sum,
            fieldnames=[
                "time",
                "input",
                "detector",
                "ckpt",
                "model",
                "frames_logged",
                "events_logged",
                "ema_alpha",
                "hysteresis_delta",
                "vote_window",
                "vote_min_count",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_name,
                "detector": args.detector,
                "ckpt": model_meta.get("ckpt"),
                "model": model_meta.get("model"),
                "frames_logged": int(frame_index),
                "events_logged": int(len(events)),
                "ema_alpha": params.ema_alpha,
                "hysteresis_delta": params.hysteresis_delta,
                "vote_window": params.vote_window,
                "vote_min_count": params.vote_min_count,
            }
        )

    # per_class_correctness.csv (for easy comparison of 7 expressions)
    _write_per_class_correctness_summary(
        per_frame_csv=frames_csv,
        out_csv=per_class_csv,
        classes=list(CANONICAL_7),
    )

    cap.release()
    cv2.destroyAllWindows()

    print("\nWrote artifacts:")
    print(f"- {frames_csv}")
    print(f"- {events_csv}")
    print(f"- {summary_csv}")
    print(f"- {per_class_csv}")
    print(f"- {thresholds_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
