from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


CANONICAL_7: Tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
LABEL_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_7)}


@dataclass(frozen=True)
class ManifestRow:
    image_path: str  # posix-like path relative to out_root
    label: str
    split: str  # train|val|test
    source: str
    # Optional ExpW-style bbox metadata (and similar). When present, the loader can crop faces on-the-fly.
    confidence: Optional[float] = None
    orig_image: Optional[str] = None
    face_id: Optional[int] = None
    bbox_top: Optional[int] = None
    bbox_left: Optional[int] = None
    bbox_right: Optional[int] = None
    bbox_bottom: Optional[int] = None


def resolve_image_path(out_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return (out_root / p).resolve()


def _clamp_bbox(
    top: int,
    left: int,
    right: int,
    bottom: int,
    *,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w))
    bottom = max(0, min(bottom, h))
    if right <= left:
        right = min(w, left + 1)
    if bottom <= top:
        bottom = min(h, top + 1)
    return top, left, right, bottom


def _stable_seed(seed: int, source: str) -> int:
    # Deterministic per-source seed without importing hashlib (keep it simple & stable).
    acc = seed
    for ch in source:
        acc = (acc * 131 + ord(ch)) & 0xFFFFFFFF
    return acc


def read_manifest(csv_path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            def _opt_int(key: str) -> Optional[int]:
                v = (r.get(key) or "").strip()
                if not v:
                    return None
                try:
                    return int(float(v))
                except Exception:
                    return None

            def _opt_float(key: str) -> Optional[float]:
                v = (r.get(key) or "").strip()
                if not v:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            rows.append(
                ManifestRow(
                    image_path=(r.get("image_path") or "").strip(),
                    label=(r.get("label") or "").strip(),
                    split=(r.get("split") or "").strip().lower(),
                    source=(r.get("source") or "").strip(),
                    confidence=_opt_float("confidence"),
                    orig_image=(r.get("orig_image") or "").strip() or None,
                    face_id=_opt_int("face_id"),
                    bbox_top=_opt_int("bbox_top"),
                    bbox_left=_opt_int("bbox_left"),
                    bbox_right=_opt_int("bbox_right"),
                    bbox_bottom=_opt_int("bbox_bottom"),
                )
            )
    return rows


def build_splits(
    manifest_rows: Sequence[ManifestRow],
    *,
    out_root: Path,
    val_fraction_for_sources_without_val: float = 0.05,
    seed: int = 1337,
    verify_paths: bool = True,
) -> Tuple[List[ManifestRow], List[ManifestRow], List[ManifestRow]]:
    """Split policy:

    - Use existing `split=='val'` rows as validation.
    - For sources that have *no* explicit val rows: carve `val_fraction_for_sources_without_val` from that source's train rows.
    - Test rows are returned separately (not used by default).

    Returns: (train_rows, val_rows, test_rows)
    """

    # Filter out any obviously invalid rows early.
    filtered: List[ManifestRow] = []
    for r in manifest_rows:
        if not r.image_path or r.label not in LABEL_TO_INDEX:
            continue
        # Normalize split
        split = r.split if r.split in {"train", "val", "test"} else "train"
        filtered.append(
            ManifestRow(
                image_path=r.image_path,
                label=r.label,
                split=split,
                source=r.source,
                confidence=r.confidence,
                orig_image=r.orig_image,
                face_id=r.face_id,
                bbox_top=r.bbox_top,
                bbox_left=r.bbox_left,
                bbox_right=r.bbox_right,
                bbox_bottom=r.bbox_bottom,
            )
        )

    by_source: Dict[str, List[ManifestRow]] = {}
    for r in filtered:
        by_source.setdefault(r.source, []).append(r)

    train_out: List[ManifestRow] = []
    val_out: List[ManifestRow] = []
    test_out: List[ManifestRow] = []

    for source, rows in by_source.items():
        src_train = [r for r in rows if r.split == "train"]
        src_val = [r for r in rows if r.split == "val"]
        src_test = [r for r in rows if r.split == "test"]

        if src_val:
            train_out.extend(src_train)
            val_out.extend(src_val)
        else:
            if val_fraction_for_sources_without_val > 0 and src_train:
                rng = random.Random(_stable_seed(seed, source))
                # Stratified-ish by label: sample per label.
                by_label: Dict[str, List[ManifestRow]] = {}
                for r in src_train:
                    by_label.setdefault(r.label, []).append(r)

                src_val_new: List[ManifestRow] = []
                src_train_new: List[ManifestRow] = []
                for label, items in by_label.items():
                    items = items[:]  # copy
                    rng.shuffle(items)
                    k = max(1, int(round(len(items) * val_fraction_for_sources_without_val))) if len(items) > 20 else 0
                    src_val_new.extend(items[:k])
                    src_train_new.extend(items[k:])

                train_out.extend(src_train_new)
                val_out.extend(src_val_new)
            else:
                train_out.extend(src_train)

        test_out.extend(src_test)

    if verify_paths:
        # Ensure paths exist (drop missing).
        def _exists(r: ManifestRow) -> bool:
            return resolve_image_path(out_root, r.image_path).exists()

        train_out = [r for r in train_out if _exists(r)]
        val_out = [r for r in val_out if _exists(r)]
        test_out = [r for r in test_out if _exists(r)]

    return train_out, val_out, test_out


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        out_root: Path,
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.out_root = out_root
        self.transform = transform if transform is not None else default_transform()
        self.return_path = bool(return_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_path = resolve_image_path(self.out_root, r.image_path)
        # Always RGB
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if (
                r.bbox_top is not None
                and r.bbox_left is not None
                and r.bbox_right is not None
                and r.bbox_bottom is not None
            ):
                w, h = im.size
                top, left, right, bottom = _clamp_bbox(
                    r.bbox_top,
                    r.bbox_left,
                    r.bbox_right,
                    r.bbox_bottom,
                    w=w,
                    h=h,
                )
                im = im.crop((left, top, right, bottom))
            im = self.transform(im)

        y = LABEL_TO_INDEX[r.label]
        if self.return_path:
            return im, y, r.source, r.image_path
        return im, y, r.source


def default_transform(image_size: int = 224) -> T.Compose:
    # Keep it simple and standard for RN18.
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
