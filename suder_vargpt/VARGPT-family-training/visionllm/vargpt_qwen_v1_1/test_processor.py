
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
try:
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_processor(processor, config, tokenizer, model_args)
except Exception as e:
    logger.debug(f"Processor was not found: {e}.")
    processor = None

# Avoid load tokenizer, see:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
if processor is not None and "Processor" not in processor.__class__.__name__:
    processor = None
    