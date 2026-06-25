# Copyright 2024 the LlamaFactory team.
#
# Compatibility alias for this checkout: the data package implementation lives
# in llamafactory.data_temp, while upstream-style imports expect llamafactory.data.

from importlib import import_module
from pathlib import Path
import sys

_data_temp = import_module("llamafactory.data_temp")

__path__ = [str(Path(__file__).resolve().parents[1] / "data_temp")]
__all__ = list(getattr(_data_temp, "__all__", []))

for _submodule in (
    "aligner",
    "collator",
    "data_utils",
    "find_gen_str",
    "find_token_sequence",
    "formatter",
    "loader",
    "mm_plugin",
    "parser",
    "preprocess",
    "template",
    "tool_utils",
):
    sys.modules[f"{__name__}.{_submodule}"] = import_module(f"llamafactory.data_temp.{_submodule}")

for _name in __all__:
    globals()[_name] = getattr(_data_temp, _name)
