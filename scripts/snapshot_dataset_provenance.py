from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FileStat:
    rel_path: str
    size: int


def _iter_files(root: Path, *, include_all: bool) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if include_all:
            yield p
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def _stable_filelist_hash(items: List[FileStat]) -> str:
    # Stable fingerprint without reading file contents.
    # Hash is over sorted lines: <relpath>\t<size>\n
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.rel_path):
        h.update(it.rel_path.encode("utf-8"))
        h.update(b"\t")
        h.update(str(int(it.size)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _norm_rel(root: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _summarize(root: Path, *, include_all: bool, sample_n: int) -> Dict[str, object]:
    ext_counts: Dict[str, int] = {}
    total_bytes = 0
    items: List[FileStat] = []

    for p in _iter_files(root, include_all=include_all):
        try:
            st = p.stat()
        except OSError:
            continue
        rel = _norm_rel(root, p)
        size = int(st.st_size)
        total_bytes += size
        items.append(FileStat(rel_path=rel, size=size))

        ext = p.suffix.lower() or "<no_ext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Build a small deterministic sample list.
    sample = [it.rel_path for it in sorted(items, key=lambda x: x.rel_path)[: max(0, int(sample_n))]]

    return {
        "root": _norm_rel(REPO_ROOT, root) if root.is_absolute() else str(root.as_posix()),
        "include_all_files": bool(include_all),
        "image_exts": sorted(IMAGE_EXTS),
        "file_count": int(len(items)),
        "total_bytes": int(total_bytes),
        "ext_counts": dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "stable_filelist_sha256": _stable_filelist_hash(items),
        "sample_paths": sample,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Snapshot dataset folder provenance (counts + stable SHA256 fingerprint over the file list). "
            "This helps document exactly what local dataset copy was evaluated, especially for Kaggle-packaged datasets."
        )
    )
    ap.add_argument("--root", type=Path, required=True, help="Dataset folder root")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/provenance/dataset_snapshot__<folder>__<YYYYMMDD>.json)",
    )
    ap.add_argument("--include-all-files", action="store_true", help="Include non-image files in the snapshot")
    ap.add_argument("--sample-n", type=int, default=25, help="Number of sample relative paths to record")

    args = ap.parse_args()

    root = args.root
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    stamp = time.strftime("%Y%m%d")
    if args.out is None:
        out = REPO_ROOT / "outputs" / "provenance" / f"dataset_snapshot__{root.name}__{stamp}.json"
    else:
        out = args.out
        if not out.is_absolute():
            out = (REPO_ROOT / out).resolve()

    out.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {"os": os.name},
        "dataset": _summarize(root, include_all=bool(args.include_all_files), sample_n=int(args.sample_n)),
    }

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
