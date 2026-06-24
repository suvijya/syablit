"""Utility functions."""
import gc
import torch

def clear_mem():
    """Clear GPU/CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
