import requests
from PIL import Image
import sys, os
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, parent_dir)

import torch
from transformers import AutoProcessor, AutoTokenizer
from vargpt_qwen_v1_1.modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
from vargpt_qwen_v1_1.prepare_vargpt_v1_1 import prepare_vargpt_qwen2vl_v1_1 
from vargpt_qwen_v1_1.processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor
from patching_utils.patching import patching

model_id = "VARGPT-family/VARGPT-v1.1"

prepare_vargpt_qwen2vl_v1_1(model_id)

model = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True, 
).to(0)

patching(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = VARGPTQwen2VLProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please explain the meme in detail."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
image_file = "./assets/llava_bench_demo.png"
print(prompt)

raw_image = Image.open(image_file)
inputs = processor(images=[raw_image], text=prompt, return_tensors='pt').to(0, torch.float32)

output = model.generate(
    **inputs, 
    max_new_tokens=2048, 
    do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))
