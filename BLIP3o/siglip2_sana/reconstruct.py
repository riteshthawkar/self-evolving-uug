from copy import deepcopy
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, LCMScheduler, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms import InterpolationMode, v2
from transformers import AutoModelForDepthEstimation
from transformers import PreTrainedModel, SiglipImageProcessor, CLIPImageProcessor, AutoImageProcessor
import PIL
from encoder import EncoderConfig, Encoder
from sana import SanaConfig, Sana
from torchvision.transforms import v2
from trainer_utils import ProcessorWrapper
from tqdm import tqdm


class ReconstructConfig(
    EncoderConfig,
    SanaConfig,
):

    def __init__(
        self,
        encoder_id: str = "google/siglip2-so400m-patch16-512",
        diffusion_model: str = "sana",
        vae_id: str = "stabilityai/sdxl-vae",
        input_size: int = 32,
        noise_scheduler_id: str = "facebook/DiT-XL-2-256",
        scheduler_id: str = "facebook/DiT-XL-2-256",
        num_pooled_tokens: int = -1,
        **kwargs,
    ):

        SanaConfig.__init__(self, **kwargs)


        for key, value in kwargs.items():
            setattr(self, key, value)
        self.encoder_id = encoder_id
        self.diffusion_model = diffusion_model
        self.vae_id = vae_id
        self.input_size = input_size
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self.num_pooled_tokens = num_pooled_tokens



class Reconstruct(PreTrainedModel):
    config_class = ReconstructConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

  
        self.encoder = Encoder(EncoderConfig(**config.to_dict()))
        config.latent_embedding_size = self.encoder.model.config.hidden_size
        self.processor = SiglipImageProcessor.from_pretrained(config.encoder_id)
        self.source_image_size = min(self.processor.size["height"], self.processor.size["width"])

        self.source_transform = v2.Compose(
            [
                v2.Resize(self.source_image_size),
                v2.CenterCrop(self.source_image_size),
                ProcessorWrapper(self.processor),
            ]
        )



        self.transformer = Sana(SanaConfig(**config.to_dict()))
        self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.noise_scheduler_id, subfolder="scheduler")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(config.scheduler_id, subfolder="scheduler")


    def get_source_transform(self):
        return self.source_transform


    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.vae is not None:
            latents = latents / self.vae.config.scaling_factor
            if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            samples = self.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def sample_images_autoencoder(
        self,
        x_source=None,
        pred_latent=None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        source_transform = self.get_source_transform()

        bsz = len(x_source)
        x_source_null = [source_transform(PIL.Image.new("RGB", (img.shape[1], img.shape[2]))).to(device=device, dtype=dtype) for img in x_source]
        x_source = torch.stack(x_source_null + x_source, dim=0)

        latent_size = self.config.input_size
        latent_channels = self.config.in_channels

        latents = randn_tensor(
            shape=(bsz * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )


        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        z_latents_input = self.encoder(x_source, num_pooled_tokens=self.config.num_pooled_tokens)

        if not pred_latent is None:
            z_latents_input[1] = pred_latent
        for t in tqdm(self.scheduler.timesteps, desc="Sampling images", disable=not enable_progress_bar):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = latent_model_input.to(z_latents_input.dtype)
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.transformer(
                x=latent_model_input,
                z_latents=z_latents_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latents.device),
            )

            # learned sigma
            if self.config.learn_sigma:
                noise_pred = torch.split(noise_pred, latent_channels, dim=1)[0]

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latents(latents.to(self.vae.dtype) if self.vae is not None else latents, return_tensor=return_tensor)
        return samples


