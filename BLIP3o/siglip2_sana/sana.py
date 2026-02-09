from diffusers import SanaTransformer2DModel
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from diffusers.models.normalization import RMSNorm
import math


class SanaConfig(PretrainedConfig):
    model_type = "Sana"

    def __init__(
        self,
        unet_id: str = "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self._gradient_checkpointing = _gradient_checkpointing


class Sana(PreTrainedModel):
    config_class = SanaConfig

    def __init__(
        self,
        config: SanaConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.transformer = SanaTransformer2DModel.from_pretrained(config.unet_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.connector_in_dim = 1152
        self.connector_out_dim = self.transformer.config.caption_channels
        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))
        self.connector = nn.Sequential(
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        if self._gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()


    def forward(self, x, timestep, z_latents, **kwargs):
        model_pred = self.transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=self.connector(z_latents),
            encoder_attention_mask=None,
        ).sample
        return model_pred

