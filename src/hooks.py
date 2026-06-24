"""Activation hooks for abliteration."""
import torch
from typing import Dict, List, Callable

def direction_ablation_hook(refusal_direction):
    """Create a hook that ablates a refusal direction."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        proj = torch.einsum("...d,d->...", x, refusal_direction.to(x.device))
        return x - proj.unsqueeze(-1) * refusal_direction.unsqueeze(0)
    return hook_fn

def get_orthogonalized_matrix(matrix, direction):
    """Orthogonalize a matrix w.r.t. a direction."""
    proj = torch.matmul(matrix, direction.to(matrix.device))
    return matrix - torch.outer(proj, direction.to(matrix.device))

def get_activation_hook(activations, key, position, pre=True):
    """Create a hook to capture activations."""
    def hook(module, inp, output):
        if pre:
            x = inp[0]
        else:
            x = output
        if x.dim() > 2:
            x = x[0, position]
        activations.setdefault(key, []).append(x.detach().cpu())
    return hook
