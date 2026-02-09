from typing import Optional, Union, List
from typing import Tuple

import torch
from PIL import Image
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from reconstruct import Reconstruct


class ReconstructionPipeline(Reconstruct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        caption: Optional[str] = "",
        image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt: Optional[str] = "",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        output_type: Optional[str] = "pil",
        device: Optional[Union[str, torch.device]] = "cuda",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        samples = self.sample_images(
            x_source=image,
            caption=caption,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )
        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)

