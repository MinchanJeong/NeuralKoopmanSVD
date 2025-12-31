import torch
import torch.nn as nn
from typing import List, Union

from koopmansvd.models.encoders.base import BaseEncoder


class MLPEncoder(BaseEncoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: Union[nn.Module, str] = "ReLU",  # Default is now Case-Sensitive
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self._output_dim = output_dim

        # 1. Resolve Activation (Strict Case Matching)
        if isinstance(activation, str):
            if hasattr(nn, activation):
                act_layer_cls = getattr(nn, activation)
            else:
                raise ValueError(
                    f"Invalid activation '{activation}'. Must match torch.nn class name exactly (e.g., 'ReLU', 'Tanh')."
                )
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            act_layer_cls = activation
        else:

            def act_layer_cls(activation):
                if isinstance(activation, nn.Module):
                    return activation
                else:
                    return activation()

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))

            layers.append(act_layer_cls())
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
