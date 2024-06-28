"""
Helpers for distributed training.
"""

import torch as th
def dev():
    if th.cuda.is_available():
        return th.device(f"cuda:{th.cuda.current_device()}")
    return th.device("cpu")