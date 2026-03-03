from .configuration_vargpt_qwen2_vl import VARGPTQwen2VLConfig
from .processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor
from .image_processing_qwen2_vl import VARGPTQwen2VLImageProcessor
from .modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, CLIPVisionConfig, CLIPVisionModel, AutoTokenizer, AutoImageProcessor, CLIPImageProcessor, AutoConfig
from transformers import AutoProcessor, Qwen2TokenizerFast, LlavaProcessor, GenerationConfig
import torch
import os
from pathlib import Path
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

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
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_LOCAL_TOKENIZER = _THIS_DIR / "Qwen2-VL-tokenizer"
_DEFAULT_QWEN2VL_SOURCE = (
    str(_DEFAULT_LOCAL_TOKENIZER) if _DEFAULT_LOCAL_TOKENIZER.exists() else "Qwen/Qwen2-VL-7B-Instruct"
)
qwen2vl_model_id = os.environ.get("VARGPT_QWEN2VL_SOURCE", _DEFAULT_QWEN2VL_SOURCE)
vargpt_save_path = os.environ.get("VARGPT_PREPARED_SAVE_PATH", "VARGPT-v1.1")


def check_file_exists(directory, filename):
    import os
    file_path = os.path.join(directory, filename)
    return os.path.isfile(file_path)


def _resolve_source_path(source: str, fallback: str) -> str:
    """Resolve local/relative source path; fallback for invalid dot-relative paths."""
    source = str(source).strip()
    if not source:
        return fallback
    p = Path(source)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    if source.startswith("./") or source.startswith("../"):
        rel = (_THIS_DIR / source).resolve()
        if rel.exists():
            return str(rel)
        # Dot-relative but missing: do not pass invalid HF repo id like "./foo".
        return fallback
    return source

def prepare_vargpt_qwen2vl_v1_1(
    save_path=vargpt_save_path,
    qwen2vl_source=qwen2vl_model_id,
    prepared_modules=["model", "tokenizer", "processor", "image_processor"],
    device=None,
):


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existsed = False
    if check_file_exists(save_path, "config.json"):
        existsed = True
        vargpt_qwen2vl_config = VARGPTQwen2VLConfig.from_pretrained(save_path)
    else:
        # If save_path is a HF/local model id with config, treat it as existing.
        try:
            vargpt_qwen2vl_config = VARGPTQwen2VLConfig.from_pretrained(save_path)
            existsed = True
        except Exception:
            vargpt_qwen2vl_config = VARGPTQwen2VLConfig(**cfg)

    base_source = _resolve_source_path(str(qwen2vl_source), _DEFAULT_QWEN2VL_SOURCE)
    if existsed:
        tokenizer_source = _resolve_source_path(str(save_path), base_source)
    else:
        tokenizer_source = base_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    special_tokens_dict = {
        'additional_special_tokens': tokenizer.additional_special_tokens + ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']  # 你想添加的特殊 token
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    generation_config = GenerationConfig.from_pretrained(tokenizer_source)
    generation_config.special_tokens = {
        "image_gen_start": "<|image_gen_start|>",
        "image_gen_start_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_start|>'),
        "image_gen_end": "<|image_gen_end|>",
        "image_gen_end_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_end|>'),
        "image_gen_pad": "<|image_gen_pad|>",
        "image_gen_pad_token_id": tokenizer.convert_tokens_to_ids('<|image_gen_pad|>')
    }
    generation_config.allowed_special_tokens = ['<|image_gen_start|>', '<|image_gen_end|>', '<|image_gen_pad|>']

    image_process = VARGPTQwen2VLImageProcessor.from_pretrained(tokenizer_source)

    process = VARGPTQwen2VLProcessor(image_processor=image_process, tokenizer=tokenizer)

    if not existsed:
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

        print(f"Original tokenizer size before adding tokens: {len(AutoTokenizer.from_pretrained(tokenizer_source))}")
        original_model = AutoModelForVision2Seq.from_pretrained(
            tokenizer_source,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        print(f"Original model embedding size: {original_model.get_input_embeddings().weight.shape[0]}")
        print(f"New tokenizer size after adding tokens: {len(tokenizer)}")
        print(f"Number of added tokens: {num_added_tokens}")
        
        model.load_state_dict(original_model.state_dict(), strict=False)

        model.vae_local.quantize = torch.nn.Identity()

        var_ckpt = _resolve_source_path("./weights/infinity_2b_reg.pth", "./weights/infinity_2b_reg.pth")
        if not os.path.isfile(var_ckpt):
            raise FileNotFoundError(
                f"Missing VARGPT VAR checkpoint: {var_ckpt}. "
                "Set VARGPT_QWEN2VL_SOURCE to a prepared model path or ensure weights are available."
            )
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
        model.save_pretrained(save_path, torch_dtype=torch.bfloat16)

    # register into hugginface
    AutoConfig.register(vargpt_qwen2vl_config.model_type, VARGPTQwen2VLConfig)
    AutoModelForVision2Seq.register(VARGPTQwen2VLConfig, VARGPTQwen2VLForConditionalGeneration)
    AutoImageProcessor.register(VARGPTQwen2VLConfig, image_processor_class=VARGPTQwen2VLImageProcessor)
    AutoProcessor.register(VARGPTQwen2VLConfig, processor_class=VARGPTQwen2VLProcessor)


if __name__ == "__main__":
    prepare_vargpt_qwen2vl_v1_1()
