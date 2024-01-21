import torch
from torch import nn

from typing import Tuple


class RMSnorm(nn.Module):
    def __init__(self, layer_shape: Tuple, eps=1e-8, bias: bool = False):
        super(RMSnorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(layer_shape))

    def forward(self, x: torch.Tensor):
        ff_rms = torch.linalg.norm(x, dim=(1, 2), keepdim=True) * x[0].numel() ** -0.5
        x = x / ff_rms
        # [: x.shape[1], :]: allows run in evalutation mode
        x = self.scale[: x.shape[1], :].unsqueeze(0) * x
        return x
