import os
import sys
import random
import numpy as np
import torch
from PIL import Image
import gradio as gr
from diffusers import DiffusionPipeline
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import get_model_name_from_path
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Constants
MAX_SEED = 10000

model_path = sys.argv[1]
diffusion_path = model_path + "/diffusion-decoder"


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt_list: list[str]) -> str:
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt_list[0])
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def make_prompt(text: str) -> list[str]:
    raw = f"Please generate image based on the following caption: {text}"
    return [add_template([raw])]

def randomize_seed_fn(seed: int, randomize: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize else seed

def generate_image(prompt: str, seed: int, guidance_scale: float, randomize: bool) -> list[Image.Image]:
    seed = randomize_seed_fn(seed, randomize)
    set_global_seed(seed)
    formatted = make_prompt(prompt)
    images = []
    for _ in range(4):
        out = pipe(formatted, guidance_scale=guidance_scale)
        images.append(out.image)
    return images

def process_image(prompt: str, img: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    text_prompt_for_qwen = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt_for_qwen],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda:0')
    generated_ids = multi_model.generate(**inputs, max_new_tokens=1024)
    input_token_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text

# Initialize model + pipeline
disable_torch_init()
model_path = os.path.expanduser(sys.argv[1])
tokenizer, multi_model, _ = load_pretrained_model(model_path)

pipe = DiffusionPipeline.from_pretrained(
    diffusion_path,
    custom_pipeline="pipeline_llava_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multi_model,
    tokenizer=tokenizer,
    safety_checker=None
)
pipe.vae.to('cuda:0')
pipe.unet.to('cuda:0')

# Gradio UI
with gr.Blocks(title="BLIP3-o") as demo:
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(label="Input Image (optional)", type="pil")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want...",
                lines=1
            )
            seed_slider = gr.Slider(
                label="Seed",
                minimum=0, maximum=int(MAX_SEED),
                step=1, value=42
            )
            randomize_checkbox = gr.Checkbox(
                label="Randomize seed", value=False
            )
            guidance_slider = gr.Slider(
                label="Guidance Scale",
                minimum=1.0, maximum=30.0,
                step=0.5, value=3.0
            )
            run_btn    = gr.Button("Run")
            clean_btn  = gr.Button("Clean All")


            text_only = [
                [None, "A cute cat."],
                [None, "A young woman with freckles wearing a straw hat, standing in a golden wheat field."],
                [None, "A group of friends having a picnic in the park."]
            ]

            image_plus_text = [
                [f"animal-compare.png", "Are these two pictures showing the same kind of animal?"],
                [f"funny_image.jpeg", "Why is this image funny?"],
            ]

            all_examples = text_only + image_plus_text

            gr.Examples(
                examples=all_examples,
                inputs=[image_input, prompt_input],
                cache_examples=False,
                label="Try a sample (image generation (text input) or image understanding (image + text))"
            )



        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Generated Images", columns=4)
            output_text    = gr.Textbox(label="Generated Text", visible=False)

    def run_all(img, prompt, seed, guidance, randomize):
        if img is not None:
            txt = process_image(prompt, img)
            return (
                gr.update(value=[], visible=False),
                gr.update(value=txt, visible=True)
            )
        else:
            imgs = generate_image(prompt, seed, guidance, randomize)
            return (
                gr.update(value=imgs, visible=True),
                gr.update(value="", visible=False)
            )

    def clean_all():
        return (
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=42),
            gr.update(value=False),
            gr.update(value=3.0),
            gr.update(value=[], visible=False),
            gr.update(value="", visible=False)
        )

    # Chain seed randomization → run_all when clicking “Run”
    run_btn.click(
        fn=randomize_seed_fn,
        inputs=[seed_slider, randomize_checkbox],
        outputs=seed_slider
    ).then(
        fn=run_all,
        inputs=[image_input, prompt_input, seed_slider, guidance_slider, randomize_checkbox],
        outputs=[output_gallery, output_text]
    )

    # Bind Enter on the prompt textbox to the same chain
    prompt_input.submit(
        fn=randomize_seed_fn,
        inputs=[seed_slider, randomize_checkbox],
        outputs=seed_slider
    ).then(
        fn=run_all,
        inputs=[image_input, prompt_input, seed_slider, guidance_slider, randomize_checkbox],
        outputs=[output_gallery, output_text]
    )

    # Clean all inputs/outputs
    clean_btn.click(
        fn=clean_all,
        inputs=[],
        outputs=[image_input, prompt_input, seed_slider,
                 randomize_checkbox, guidance_slider,
                 output_gallery, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)
