
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")

try:
    print(f"is_autocast_enabled(): {torch.is_autocast_enabled()}")
except Exception as e:
    print(f"is_autocast_enabled() failed: {e}")

try:
    print(f"is_autocast_enabled('cuda'): {torch.is_autocast_enabled('cuda')}")
except Exception as e:
    print(f"is_autocast_enabled('cuda') failed: {e}")
