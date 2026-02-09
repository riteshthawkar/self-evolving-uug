from typing import List

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
    CLIPVisionModel,
    CLIPVisionConfig,
    CLIPImageProcessor,
    AutoImageProcessor,
    AutoModel,
)
import torch.nn.functional as F


class EncoderConfig(PretrainedConfig):
    model_type = "encoder"

    def __init__(
        self,
        encoder_id: str = "google/siglip-so400m-patch14-384",
        from_scratch: bool = False,
        pooler_output: bool = True,
        as_latents: bool = False,
        input_size: int = 64,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.encoder_id = encoder_id
        self.from_scratch = from_scratch
        self.pooler_output = pooler_output
        self.as_latents = as_latents
        self.input_size = input_size
        self._gradient_checkpointing = _gradient_checkpointing


class Encoder(PreTrainedModel):
    config_class = EncoderConfig

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.as_latents = config.as_latents
        self.input_size = config.input_size
        self.pooler_output = config.pooler_output if not self.as_latents else False

        if "siglip2" in config.encoder_id:
            self.processor = SiglipImageProcessor.from_pretrained(config.encoder_id)
            self.model = SiglipVisionModel.from_pretrained(config.encoder_id, attn_implementation="sdpa")
            if not self.pooler_output:
                self.model.vision_model.encoder.layers[-1].requires_grad_(False)
                self.model.vision_model.post_layernorm.requires_grad_(False)
                self.model.vision_model.head.requires_grad_(False)

        if config._gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def process(self, image):
        if not isinstance(image, List):
            image = [image]
        return self.processor(image, return_tensors="pt")["pixel_values"]

    def forward(self, x=None, num_pooled_tokens: int = -1):
        if x is not None:
            output = self.model(x, output_attentions=False, output_hidden_states=True)
        else:
            output = None

        if self.pooler_output:
            return output.pooler_output if output is not None else None
        else:
            num_pooled_tokens = num_pooled_tokens if not self.as_latents else -1
            original_last_hidden_state = output.hidden_states[-2] if output is not None else None
            if self.as_latents:
                # down sample the last hidden state's channels 1152 to 4 using average pooling
                last_hidden_state = F.adaptive_avg_pool1d(original_last_hidden_state, 16)
                size = int(last_hidden_state.size(1) ** 0.5)
                last_hidden_state = last_hidden_state.view(-1, size, size, last_hidden_state.size(-1))
                last_hidden_state = last_hidden_state.permute(0, 3, 1, 2)
                last_hidden_state = F.interpolate(last_hidden_state, (self.input_size, self.input_size), mode="bilinear", align_corners=False)

            elif num_pooled_tokens > 0 and original_last_hidden_state is not None:
                last_hidden_state = original_last_hidden_state
                size = int(last_hidden_state.size(1) ** 0.5)
                if isinstance(self.model, CLIPVisionModel):
                    cls_token, last_hidden_state = last_hidden_state[:, 0, None], last_hidden_state[:, 1:]
                else:
                    cls_token = None
                last_hidden_state = last_hidden_state.view(-1, size, size, last_hidden_state.size(-1))
                last_hidden_state = (
                    nn.functional.adaptive_avg_pool2d(last_hidden_state.permute(0, 3, 1, 2), int(num_pooled_tokens**0.5))
                    .permute(0, 2, 3, 1)
                    .view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))
                )
                last_hidden_state = torch.cat([cls_token, last_hidden_state], dim=1) if cls_token is not None else last_hidden_state

            else:
                last_hidden_state = original_last_hidden_state

            return last_hidden_state

