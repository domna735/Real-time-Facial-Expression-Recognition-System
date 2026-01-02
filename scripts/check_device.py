from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fer.utils.device import get_best_device


def main() -> None:
    info = get_best_device("auto")
    print(f"backend: {info.backend}")
    print(f"device:  {info.device}")
    print(f"detail:  {info.detail}")


if __name__ == "__main__":
    main()
