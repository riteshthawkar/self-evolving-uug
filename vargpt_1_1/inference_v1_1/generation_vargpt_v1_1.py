# https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
import requests
from PIL import Image
import torch
import sys, os
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, parent_dir)

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

# some instruction examples:
# Can you depict a scene of A power metalalbum cover featuring a fantasy-style illustration witha white falcon.
# Imagine a scene of a mesmerizing image showcasing heart shaped flowers, known as Lamprocapnos, in vibrant colors and intricate details, surrounded by a magical atmosphere created by the soft bokeh and fairy lights.
# Please design a drawing of a butterfly on a flower.
# Please create a painting of a black weasel is standing in the grass.
# Can you generate a rendered photo of a rabbit sitting in the grass.
# I need a designed image of a cute gray puppy is running on the green grass, with colorful flowers on the grass.."

 


conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Can you depict a scene of A power metalalbum cover featuring a fantasy-style illustration witha white falcon."},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)

inputs = processor(text=prompt, return_tensors='pt').to(0, torch.float32)
model._IMAGE_GEN_PATH = "output.png"
output = model.generate(
    **inputs, 
    max_new_tokens=4096, 
    do_sample=False)

print(processor.decode(output[0][:-1], skip_special_tokens=True))
