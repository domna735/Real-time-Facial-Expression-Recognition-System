from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fer.nl.memory import AssociativeMemory, apply_memory_gate  # noqa: E402
from src.fer.utils.grad_accum import GradAccumState, scale_loss_for_accum  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test for NL scaffolding (memory + grad accumulation).")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    torch.manual_seed(1337)

    # Tiny model
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 7)).to(device)
    mem = AssociativeMemory(hidden_dim=int(args.hidden_dim), layers=int(args.layers), input_dim=4).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(mem.parameters()), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = GradAccumState(accum_steps=max(1, int(args.accum)))

    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    for step in range(int(args.steps)):
        x = torch.randn(32, 64, device=device)
        y = torch.randint(0, 7, (32,), device=device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss = scale_loss_for_accum(loss, accum_steps=state.accum_steps)

        scaler.scale(loss).backward()

        # Apply memory gate to ONE parameter's grad as a minimal sanity check.
        # (We don't implement a full meta-optimizer here.)
        p0 = next(model.parameters())
        if p0.grad is not None:
            gated = apply_memory_gate(grad=p0.grad, param=p0, memory=mem, step=step, total_steps=int(args.steps))
            p0.grad.copy_(gated)

        if state.should_step():
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        state = state.next()

    dt = time.time() - t0
    print(f"[OK] NL smoke done: steps={args.steps} accum={args.accum} amp={use_amp} device={device} time_sec={dt:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
