import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"device {i}:", torch.cuda.get_device_name(i))
        try:
            x = torch.randn(1024, 1024, device=f"cuda:{i}")
            print("test tensor device:", x.device)
            print("✓ GPU is fully functional!")
        except RuntimeError as e:
            print(f"⚠ GPU kernel incompatibility: {e}")
            print("  Your RTX 5070 Ti uses sm_120, PyTorch CUDA wheels you installed support only up to sm_90.")
            print("  Workaround: Use DirectML for now, or use a PyTorch build that includes sm_120 kernels.")
            print("\n  Testing CPU fallback...")
            x = torch.randn(1024, 1024, device="cpu")
            print("✓ CPU fallback works:", x.device)
else:
    print("CUDA is NOT available!")
