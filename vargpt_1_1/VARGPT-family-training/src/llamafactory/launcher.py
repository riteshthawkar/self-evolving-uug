# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


def _patch_broken_flash_attn_on_rocm() -> None:
    r"""
    Prevents crash loops when a partial/incompatible flash-attn package is present.

    On ROCm clusters this common failure mode appears as:
      ModuleNotFoundError: No module named 'flash_attn_2_cuda'
    during transformers imports.
    """
    # Allow users to opt out if they explicitly want to debug flash-attn.
    if os.environ.get("LLAMAFACTORY_DISABLE_FLASH_ATTN_GUARD", "0").lower() in ("1", "true", "yes"):
        return

    try:
        import torch  # type: ignore
    except Exception:
        return

    # Guard is mainly needed on ROCm stacks where CUDA flash-attn wheels are often installed by mistake.
    if not getattr(torch.version, "hip", None):
        return

    broken_flash = False
    try:
        # If this import fails, installed flash-attn is unusable for this runtime.
        import flash_attn_2_cuda  # type: ignore # noqa: F401
    except Exception:
        broken_flash = True

    if not broken_flash:
        return

    os.environ.setdefault("HF_FLASH_ATTN_2_ENABLED", "0")

    # Monkey-patch Transformers availability checks before model modules are imported.
    try:
        import transformers.utils as t_utils  # type: ignore
        import transformers.utils.import_utils as t_import_utils  # type: ignore

        def _fa2_unavailable() -> bool:
            return False

        def _fa2_ver_unavailable(*args, **kwargs) -> bool:  # noqa: ANN002, ANN003
            return False

        t_import_utils.is_flash_attn_2_available = _fa2_unavailable
        t_import_utils.is_flash_attn_greater_or_equal_2_10 = _fa2_ver_unavailable
        t_utils.is_flash_attn_2_available = _fa2_unavailable
        t_utils.is_flash_attn_greater_or_equal_2_10 = _fa2_ver_unavailable
    except Exception:
        # Best-effort patch; downstream code still has other non-flash attention paths.
        pass


def launch():
    _patch_broken_flash_attn_on_rocm()
    from llamafactory.train.tuner import run_exp  # use absolute import

    run_exp()


if __name__ == "__main__":
    launch()
