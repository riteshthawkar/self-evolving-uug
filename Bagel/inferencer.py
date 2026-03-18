# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.runtime_precision import autocast_context
from modeling.bagel.qwen2_navit import NaiveCache


def _extract_assistant_segment(decoded_text: str) -> str:
    text = str(decoded_text or "")
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.rsplit("<|im_start|>", 1)[-1]
    return text



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def _vae_device_dtype(self):
        for p in self.vae_model.parameters():
            return p.device, p.dtype
        return torch.device("cpu"), torch.float32

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.environ.get(name, str(default))).strip()
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _resize_for_understanding(self, image: Image.Image) -> Image.Image:
        """Cap proposer/solver rollout images before the ViT context update.

        The self-evolving understanding path only needs the ViT branch, so using
        the larger VAE-oriented resize policy here wastes memory and can OOM on
        multi-GPU runs. Reuse the policy-update edge budget by default.
        """
        max_edge = self._env_int(
            "BAGEL_UNDERSTANDING_MAX_VIT_EDGE",
            self._env_int("BAGEL_POLICY_MAX_VIT_EDGE", 448),
        )
        if max_edge <= 0:
            return image

        width, height = image.size
        longest = max(int(width), int(height))
        if longest <= max_edge:
            return image

        scale = float(max_edge) / float(longest)
        new_width = max(1, int(round(float(width) * scale)))
        new_height = max(1, int(round(float(height) * scale)))
        return image.resize((new_width, new_height), resample=Image.BICUBIC)
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        latent0 = unpacked_latent[0]

        # Release large KV-cache references before VAE decode to reduce peak memory.
        if isinstance(gen_context, dict):
            gen_context["past_key_values"] = None
        if isinstance(cfg_text_precontext, dict):
            cfg_text_precontext["past_key_values"] = None
        if isinstance(cfg_img_precontext, dict):
            cfg_img_precontext["past_key_values"] = None

        del unpacked_latent
        del past_key_values, cfg_text_past_key_values, cfg_img_past_key_values
        del generation_input, generation_input_cfg_text, generation_input_cfg_img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = self.decode_image(latent0, image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        vae_device, vae_dtype = self._vae_device_dtype()
        if latent.device != vae_device or latent.dtype != vae_dtype:
            latent = latent.to(device=vae_device, dtype=vae_dtype)

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        token_ids = unpacked_latent[:, 0].detach().to("cpu").tolist()
        output = self.tokenizer.decode(token_ids)
        return _extract_assistant_segment(output)
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        text_top_p=1.0,
        text_top_k=0,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:
        model_device = next(self.model.parameters()).device

        def _run_once(*, use_autocast: bool) -> List[Union[str, Image.Image]]:
            output_list = []
            gen_context = self.init_gen_context()
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)
            target_image_shapes = image_shapes

            with autocast_context(model_device, enabled=use_autocast):
                if think:
                    if understanding_output:
                        system_prompt = VLM_THINK_SYSTEM_PROMPT
                    else:
                        system_prompt = GEN_THINK_SYSTEM_PROMPT
                    gen_context = self.update_context_text(system_prompt, gen_context)
                    cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

                for input_term in input_lists:
                    if isinstance(input_term, str):
                        cfg_text_context = deepcopy(gen_context)
                        gen_context = self.update_context_text(input_term, gen_context)
                        cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                    elif isinstance(input_term, Image.Image):
                        rgb_image = pil_img2rgb(input_term)
                        if understanding_output:
                            input_term = self._resize_for_understanding(rgb_image)
                            gen_context = self.update_context_image(input_term, gen_context, vae=False, vit=True)
                        else:
                            input_term = self.vae_transform.resize_transform(rgb_image)
                            gen_context = self.update_context_image(input_term, gen_context, vae=True, vit=True)
                            target_image_shapes = input_term.size[::-1]
                            cfg_text_context = deepcopy(gen_context)

                    else:
                        raise ValueError(f"Unsupported input type: {type(input_term)}")

                if understanding_output:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        top_p=text_top_p,
                        top_k=text_top_k,
                        max_length=max_think_token_n,
                    )
                    output_list.append(gen_text)

                else:
                    if think:
                        gen_text = self.gen_text(
                            gen_context,
                            do_sample=do_sample,
                            temperature=text_temperature,
                            top_p=text_top_p,
                            top_k=text_top_k,
                            max_length=max_think_token_n,
                        )
                        gen_context = self.update_context_text(gen_text, gen_context)
                        output_list.append(gen_text)

                    img = self.gen_image(
                        target_image_shapes,
                        gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                        enable_taylorseer=enable_taylorseer,
                    )
                    output_list.append(img)

            return output_list

        try:
            return _run_once(use_autocast=True)
        except RuntimeError as exc:
            msg = str(exc).lower()
            hipblas_like = ("hipblas" in msg) and ("invalid_value" in msg or "heuristic" in msg)
            dtype_mismatch = ("mat1 and mat2 must have the same dtype" in msg) or ("expected scalar type" in msg)
            if not (hipblas_like or dtype_mismatch):
                raise
            print("[inferencer] Runtime precision failure detected; retrying once with autocast disabled.")
            return _run_once(use_autocast=False)
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
