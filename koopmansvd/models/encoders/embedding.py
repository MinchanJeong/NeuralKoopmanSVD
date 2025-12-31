# koopmansvd/models/encoders/embedding.py
import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    """
    Embeds input x (assumed in [0, 1]) into sinusoidal features.
    Output dimension is 2 * input_dim.
    Maps x -> [sin(2*pi*x), cos(2*pi*x)]
    """

    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = input_dim * 2

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x is in [0, 1] or periodic
        scaled_x = 2 * np.pi * x
        return torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
