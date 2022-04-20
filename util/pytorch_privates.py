import functools
import inspect
import torch

from typing import Any, Optional
from types import FunctionType


def _log_api_usage_once(obj: Any) -> None:
    """Copied from <https://github.com/pytorch/vision/blob/1ac6e8b91b980b052324f77828a5ef4a6715dd66/torchvision/utils.py#L525>"""

    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Copied from <https://github.com/pytorch/vision/blob/1ac6e8b91b980b052324f77828a5ef4a6715dd66/torchvision/models/_utils.py#L76>"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
