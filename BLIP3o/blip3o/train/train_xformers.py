from blip3o.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from blip3o.train.train import train

if __name__ == "__main__":
    train()
