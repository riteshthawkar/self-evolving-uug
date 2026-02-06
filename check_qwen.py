
import transformers
print(f"Transformers version: {transformers.__version__}")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("SUCCESS: Qwen2_5_VLForConditionalGeneration imported")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"FAILURE: {e}")
