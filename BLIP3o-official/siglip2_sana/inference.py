from pipeline_reconstruction import ReconstructionPipeline
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipImageProcessor
from PIL import Image
import numpy as np
from PIL import Image
import torch
import sys
import os
import random


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def img_process(images, processor, image_aspect_ratio):
    if image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images
                
                
decoder_path = sys.argv[1]
device_1 = 0



model = ReconstructionPipeline.from_pretrained(decoder_path).to(device="cuda:1").to(dtype=torch.float32)
gen_image_processor = SiglipImageProcessor.from_pretrained("google/siglip2-so400m-patch16-512")


img_path = 'fig.jpg'
images = Image.open(img_path)

x_source = img_process(
    images,
    gen_image_processor,
    "square",
).squeeze(0).to(device="cuda:1")

x_source = [x_source]
samples = model.sample_images_autoencoder(x_source=x_source)
samples[0].save('reconstruction.png')

