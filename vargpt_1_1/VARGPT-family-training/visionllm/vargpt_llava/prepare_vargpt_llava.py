from .configuration_vargpt_llava import VARGPTLlavaConfig
from .processing_vargpt_llava import VARGPTLlavaProcessor
from .modeling_vargpt_llava import VARGPTLlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, CLIPVisionConfig, CLIPVisionModel, AutoTokenizer, AutoImageProcessor, CLIPImageProcessor, AutoConfig
from transformers import AutoProcessor, LlavaProcessor, GenerationConfig, LlavaForConditionalGeneration
import torch
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

cfg={
  "ignore_index": -100,
  "image_token_index": 32000,
  "model_type": "vargpt_llava",
  "pad_token_id": 32001,
  "projector_hidden_act": "gelu",
  "text_config": {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "rms_norm_eps": 1e-05,
    "torch_dtype": "float16",
    "vocab_size": 32064
  },
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.36.0.dev0",
  "vision_config": {
    "hidden_size": 1024,
    "image_size": 336,
    "intermediate_size": 4096,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "vocab_size": 32000
  },
  "hidden_size": 4096,
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "default",
  "vocab_size": 32064
}


vision_model = "openai/clip-vit-large-patch14-336"
llava_model_id = "llava-hf/llava-1.5-7b-hf" 


vargpt_save_path = "VARGPT_LLaVA-7B" 


def check_file_exists(directory, filename):
    import os
    file_path = os.path.join(directory, filename)
    return os.path.isfile(file_path)


def prepare_vargpt_llava(save_path=vargpt_save_path, prepared_modules=["model", "tokenizer", "processor", "image_processor"], device=None):


    from llamafactory.data.template import _register_template, StringFormatter, EmptyFormatter, get_mm_plugin

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    existsed = False
    if check_file_exists(save_path, "config.json"):
        existsed = True

    if existsed:
        vargpt_llava_config = VARGPTLlavaConfig.from_pretrained(save_path)
    else:
        vargpt_llava_config = VARGPTLlavaConfig(**cfg)


    tokenizer = AutoTokenizer.from_pretrained(llava_model_id)
    special_tokens_dict = {
        'additional_special_tokens': tokenizer.additional_special_tokens + ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']  # 你想添加的特殊 token
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    generation_config = GenerationConfig.from_pretrained(llava_model_id)
    generation_config.special_tokens = {
        "image_gen_start": "<|image_gen_start|>",
        "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
        "image_gen_end": "<|image_gen_end|>",
        "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
        "image_gen_pad": "<|image_gen_pad|>",
        "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
    }
    generation_config.allowed_special_tokens = ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']
    
    image_process = CLIPImageProcessor.from_pretrained(vision_model)
    process = VARGPTLlavaProcessor(image_processor=image_process, tokenizer=tokenizer)

    if not existsed:
        vargpt_llava_config.train_from_scratch = False
        vargpt_llava_config.torch_dtype = torch.bfloat16  # 明确设置 dtype
        vargpt_llava_config.special_tokens = {
            "image_gen_start": "<|image_gen_start|>",
            "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
            "image_gen_end": "<|image_gen_end|>",
            "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
            "image_gen_pad": "<|image_gen_pad|>",
            "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
            
        }   
        model = VARGPTLlavaForConditionalGeneration._from_config(vargpt_llava_config).to(
            device=device,
            dtype=torch.bfloat16)
        print(f"New model embedding size before resize: {model.get_input_embeddings().weight.shape[0]}")

        print(f"Original tokenizer size before adding tokens: {len(AutoTokenizer.from_pretrained(llava_model_id))}")
        original_model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        print(f"Original model embedding size: {original_model.get_input_embeddings().weight.shape[0]}")
        print(f"New tokenizer size after adding tokens: {len(tokenizer)}")
        print(f"Number of added tokens: {num_added_tokens}")
        
        model.load_state_dict(original_model.state_dict(), strict=False)
        vae_ckpt = '/VAR/vae_ch160v4096z32.pth'
        model.vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        var_ckpt = "/VAR/var_d30.pth"
        ckpt = torch.load(var_ckpt, map_location='cpu')
        new_state_dict = {}
        for key, value in ckpt.items():
            if key in model.vargpt_gen.state_dict():
                if model.vargpt_gen.state_dict()[key].shape == value.shape:
                    new_state_dict[key] = value
                else:
                    print(f"跳过参数 {key} 因为形状不匹配: checkpoint形状 {value.shape} vs 模型形状 {model.vargpt_gen.state_dict()[key].shape}")
        model.vargpt_gen.load_state_dict(new_state_dict, strict=False)

        vargpt_llava_config = model.config
        vargpt_llava_config.special_tokens = {
            "image_gen_start": "<|image_gen_start|>",
            "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
            "image_gen_end": "<|image_gen_end|>",
            "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
            "image_gen_pad": "<|image_gen_pad|>",
            "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
        }
        print(f"New model embedding size after loading weights: {model.get_input_embeddings().weight.shape[0]}")
        
        
    vargpt_llava_config.architectures = [VARGPTLlavaForConditionalGeneration.__name__]
    vargpt_llava_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    vargpt_llava_config.padding_side = tokenizer.padding_side

    if not existsed:
        vargpt_llava_config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        generation_config.save_pretrained(save_path)
        image_process.save_pretrained(save_path)
        process.save_pretrained(save_path)
        model.save_pretrained(save_path, torch_dtype=torch.bfloat16)

    # register into hugginface
    AutoConfig.register(vargpt_llava_config.model_type, VARGPTLlavaConfig)
    AutoModelForVision2Seq.register(VARGPTLlavaConfig, VARGPTLlavaForConditionalGeneration)
    AutoProcessor.register(VARGPTLlavaConfig, processor_class=VARGPTLlavaProcessor)

    if_verify = False

    _register_template(
        name="vargpt_llava",
        format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
        default_system=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        mm_plugin=get_mm_plugin(name="vargpt_llava", image_token="<image>", image_gen_token = "<|image_gen_pad|>", image_gen_token_num=680),
    )

  
if __name__ == "__main__":
    prepare_vargpt_llava()