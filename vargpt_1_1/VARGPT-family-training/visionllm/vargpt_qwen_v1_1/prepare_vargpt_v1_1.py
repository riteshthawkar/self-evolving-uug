from .configuration_vargpt_qwen2_vl import VARGPTQwen2VLConfig
from .processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor
from .image_processing_qwen2_vl import VARGPTQwen2VLImageProcessor
from .modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, CLIPVisionConfig, CLIPVisionModel, AutoTokenizer, AutoImageProcessor, CLIPImageProcessor, AutoConfig
from transformers import AutoProcessor, Qwen2TokenizerFast, LlavaProcessor, GenerationConfig
import torch
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin
import os

cfg= {
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "vision_token_id": 151654,
  "image_token_id": 151655,
  "video_token_id": 151656,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "vargpt_qwen2_vl",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.41.2",
  "use_cache": True,
  "use_sliding_window": False,
  "vision_config": {
    "depth": 32,
    "embed_dim": 1280,
    "mlp_ratio": 4,
    "num_heads": 16,
    "in_chans": 3,
    "hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2
  },
  "rope_scaling": {
    "type": "mrope",
    "mrope_section": [
      16,
      24,
      24
    ]
  },
  "vocab_size": 152064
}
qwen2vl_model_id = os.environ.get("VARGPT_QWEN2VL_MODEL_ID", "VARGPT-family/VARGPT-v1.1")
vargpt_save_path = "VARGPT-v1.1" 


def check_file_exists(directory, filename):
    import os
    file_path = os.path.join(directory, filename)
    return os.path.isfile(file_path)


def prepare_vargpt_qwen2vl_v1_1(
    save_path=vargpt_save_path,
    prepared_modules=["model", "tokenizer", "processor", "image_processor"],
    device=None,
    base_model_id=None,
):


    from llamafactory.data.template import _register_template, StringFormatter, EmptyFormatter, get_mm_plugin

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    existsed = check_file_exists(save_path, "config.json")
    source_model_id = base_model_id or os.environ.get("VARGPT_QWEN2VL_MODEL_ID", qwen2vl_model_id)

    if existsed:
        vargpt_qwen2vl_config = VARGPTQwen2VLConfig.from_pretrained(save_path)
        source_for_assets = save_path
    else:
        source_for_assets = source_model_id
        try:
            vargpt_qwen2vl_config = VARGPTQwen2VLConfig.from_pretrained(source_for_assets)
        except Exception:
            vargpt_qwen2vl_config = VARGPTQwen2VLConfig(**cfg)


    try:
        tokenizer = AutoTokenizer.from_pretrained(source_for_assets)
    except Exception:
        if source_for_assets == save_path:
            source_for_assets = source_model_id
            existsed = False
            tokenizer = AutoTokenizer.from_pretrained(source_for_assets)
        else:
            raise
    special_tokens_dict = {
        'additional_special_tokens': tokenizer.additional_special_tokens + ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']  # 你想添加的特殊 token
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    try:
        generation_config = GenerationConfig.from_pretrained(source_for_assets)
    except Exception:
        generation_config = GenerationConfig()
    generation_config.special_tokens = {
        "image_gen_start": "<|image_gen_start|>",
        "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
        "image_gen_end": "<|image_gen_end|>",
        "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
        "image_gen_pad": "<|image_gen_pad|>",
        "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
    }
    generation_config.allowed_special_tokens = ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']
    
    try:
        image_process = VARGPTQwen2VLImageProcessor.from_pretrained(source_for_assets)
    except Exception:
        if source_for_assets == save_path:
            source_for_assets = source_model_id
            existsed = False
            image_process = VARGPTQwen2VLImageProcessor.from_pretrained(source_for_assets)
        else:
            raise

    process = VARGPTQwen2VLProcessor(image_processor=image_process, tokenizer=tokenizer)

    model = None
    prepare_from_scratch = os.environ.get("VARGPT_PREPARE_FROM_SCRATCH", "0").lower() in ("1", "true", "yes")
    if not existsed and prepare_from_scratch:
        vargpt_qwen2vl_config.train_from_scratch = False
        vargpt_qwen2vl_config.torch_dtype = torch.bfloat16  # 明确设置 dtype
        vargpt_qwen2vl_config.special_tokens = {
            "image_gen_start": "<|image_gen_start|>",
            "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
            "image_gen_end": "<|image_gen_end|>",
            "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
            "image_gen_pad": "<|image_gen_pad|>",
            "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
            
        }   
        model = VARGPTQwen2VLForConditionalGeneration._from_config(vargpt_qwen2vl_config).to(
            device=device,
            dtype=torch.bfloat16)
        print(f"New model embedding size before resize: {model.get_input_embeddings().weight.shape[0]}")

        print(f"Original tokenizer size before adding tokens: {len(AutoTokenizer.from_pretrained(source_for_assets))}")
        original_model = AutoModelForVision2Seq.from_pretrained(
            source_for_assets,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        print(f"Original model embedding size: {original_model.get_input_embeddings().weight.shape[0]}")
        print(f"New tokenizer size after adding tokens: {len(tokenizer)}")
        print(f"Number of added tokens: {num_added_tokens}")
        
        model.load_state_dict(original_model.state_dict(), strict=False)
        model.vae_local.quantize = torch.nn.Identity()

        var_ckpt = "./weights/infinity_2b_reg.pth"
        # model.vargpt_gen.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
        ckpt = torch.load(var_ckpt, map_location='cpu')
        new_state_dict = {}
        for key, value in ckpt.items():
            if key in model.vargpt_gen.state_dict():
                if model.vargpt_gen.state_dict()[key].shape == value.shape:
                    new_state_dict[key] = value
                else:
                    print(f"跳过参数 {key} 因为形状不匹配: checkpoint形状 {value.shape} vs 模型形状 {model.vargpt_gen.state_dict()[key].shape}")
        model.vargpt_gen.load_state_dict(new_state_dict, strict=False)

        vargpt_qwen2vl_config = model.config
        vargpt_qwen2vl_config.special_tokens = {
            "image_gen_start": "<|image_gen_start|>",
            "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
            "image_gen_end": "<|image_gen_end|>",
            "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
            "image_gen_pad": "<|image_gen_pad|>",
            "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
        }
        print(f"New model embedding size after loading weights: {model.get_input_embeddings().weight.shape[0]}")
        

    vargpt_qwen2vl_config.architectures = [VARGPTQwen2VLForConditionalGeneration.__name__]
    vargpt_qwen2vl_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    vargpt_qwen2vl_config.padding_side = tokenizer.padding_side

    if not existsed:
        vargpt_qwen2vl_config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        generation_config.save_pretrained(save_path)
        image_process.save_pretrained(save_path)
        process.save_pretrained(save_path)
        if model is not None:
            model.save_pretrained(save_path, torch_dtype=torch.bfloat16)

    # register into hugginface
    AutoConfig.register(vargpt_qwen2vl_config.model_type, VARGPTQwen2VLConfig)
    AutoModelForVision2Seq.register(VARGPTQwen2VLConfig, VARGPTQwen2VLForConditionalGeneration)
    AutoImageProcessor.register(VARGPTQwen2VLConfig, image_processor_class=VARGPTQwen2VLImageProcessor)
    AutoProcessor.register(VARGPTQwen2VLConfig, processor_class=VARGPTQwen2VLProcessor)

    if_verify = False
    _register_template(
        name="vargpt_qwen2_vl",
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_separator=EmptyFormatter(slots=["\n"]),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True,
        replace_jinja_template=False,
        mm_plugin=get_mm_plugin(name="vargpt_qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>", 
            image_gen_token = "<|image_gen_pad|>", 
            image_gen_token_num=2560), # 256*256: 640 tokens 512*512: 2560tokens
    )
    
if __name__ == "__main__":
    prepare_vargpt_qwen2vl_v1_1()
