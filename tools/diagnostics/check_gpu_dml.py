import torch

try:
    import torch_directml
except Exception as e:
    raise SystemExit(f"torch_directml import failed: {e}")

dml = torch_directml.device()

print("torch version:", torch.__version__)
print("dml device:", dml)

# Smoke test: matmul on DirectML device
x = torch.randn(512, 512, device=dml)
y = x @ x

print("ok:", y.shape, y.device)
print("mean:", float(y.mean().cpu()))
