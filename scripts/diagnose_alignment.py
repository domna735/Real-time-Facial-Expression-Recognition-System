from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.data.manifest_dataset import CANONICAL_7, read_manifest  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose alignment between manifests, softlabels, and checkpoints.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--softlabels-dir", type=Path, required=True, help="Directory produced by scripts/export_softlabels.py")
    ap.add_argument("--require-classorder", action="store_true")
    args = ap.parse_args()

    problems: List[str] = []

    # 1) Manifest sanity
    rows = read_manifest(args.manifest)
    if not rows:
        problems.append(f"manifest is empty: {args.manifest}")

    # 2) Softlabels artifacts
    align_path = args.softlabels_dir / "alignmentreport.json"
    classorder_path = args.softlabels_dir / "classorder.json"
    hash_manifest_path = args.softlabels_dir / "hash_manifest.json"
    npz_path = args.softlabels_dir / "softlabels.npz"
    index_path = args.softlabels_dir / "softlabels_index.jsonl"

    for p in [align_path, classorder_path, hash_manifest_path, npz_path, index_path]:
        if not p.exists():
            problems.append(f"missing: {p}")

    if align_path.exists():
        align = _read_json(align_path)
        if str(align.get("manifest")) != str(args.manifest):
            problems.append("alignmentreport.manifest does not match --manifest")

    if classorder_path.exists():
        co = _read_json(classorder_path)
        if isinstance(co, list):
            if co != list(CANONICAL_7):
                problems.append(f"classorder.json mismatch: expected {list(CANONICAL_7)}")
        else:
            problems.append("classorder.json is not a list")
    elif args.require_classorder:
        problems.append("--require-classorder but classorder.json missing")

    # 3) Basic shape check
    if npz_path.exists():
        import numpy as np

        data = np.load(npz_path)
        logits = data["logits"]
        y = data["y"]
        if logits.ndim != 2 or logits.shape[1] != len(CANONICAL_7):
            problems.append(f"softlabels logits shape invalid: {logits.shape}")
        if y.ndim != 1 or y.shape[0] != logits.shape[0]:
            problems.append(f"softlabels y shape invalid: y={y.shape} logits={logits.shape}")

    if problems:
        for m in problems:
            print(f"[FAIL] {m}")
        _fail(f"Alignment diagnosis failed with {len(problems)} issue(s).")

    print("[OK] alignment artifacts look consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
