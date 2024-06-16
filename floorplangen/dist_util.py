"""
Helpers for distributed training.
"""

import torch as th
def dev():
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")