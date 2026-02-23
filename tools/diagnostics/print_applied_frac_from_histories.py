from __future__ import annotations

import json
from pathlib import Path


def read_history(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"history.json is not a list: {path}")
    return data


def series(hist, key1: str, key2: str):
    out = []
    for e in hist:
        if not isinstance(e, dict):
            out.append(None)
            continue
        block = e.get(key1)
        if not isinstance(block, dict):
            out.append(None)
            continue
        out.append(block.get(key2))
    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    runs = {
        "nl_topk0p05_w0p1": repo
        / "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953/history.json",
        "negl_ent0p3": repo
        / "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203/history.json",
        "negl_ent0p5": repo
        / "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949/history.json",
        "synergy_ent0p4_topk0p05": repo
        / "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602/history.json",
    }

    for name, p in runs.items():
        print(f"\n=== {name}")
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        hist = read_history(p)
        nl_frac = series(hist, "nl", "applied_frac")
        negl_frac = series(hist, "negl", "applied_frac")
        print("nl.applied_frac  :", nl_frac)
        print("negl.applied_frac:", negl_frac)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
