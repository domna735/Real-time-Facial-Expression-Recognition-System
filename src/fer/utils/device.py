from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    backend: str  # "cuda" | "dml" | "cpu"
    device: torch.device
    detail: str


def get_best_device(prefer: str = "auto") -> DeviceInfo:
    prefer = (prefer or "auto").lower()

    if prefer in {"cuda", "auto"} and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return DeviceInfo(backend="cuda", device=torch.device("cuda"), detail=name)

    if prefer in {"dml", "auto"}:
        try:
            import torch_directml  # type: ignore

            dml = torch_directml.device()
            return DeviceInfo(backend="dml", device=dml, detail=str(dml))
        except Exception:
            pass

    return DeviceInfo(backend="cpu", device=torch.device("cpu"), detail="cpu")
