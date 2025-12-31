# koopmansvd/models/encoders/cnn.py
import torch
import torch.nn as nn
from typing import List
from .base import BaseEncoder


class CNNEncoder(BaseEncoder):
    """
    Paper Architecture Implementation.
    Structure: Conv5x5 -> MaxPool -> Conv5x5 -> MaxPool -> Flatten -> Linear
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 16,
        hidden_channels: List[int] = [16, 32],
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self._output_dim = output_dim

        # Layer 1: (B, 1, 28, 28) -> (B, 16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_channels[0], kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Layer 2: (B, 16, 14, 14) -> (B, 32, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels[0],
                hidden_channels[1],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Flatten Dimension Calculation
        # 32 channels * 7 * 7 = 1568
        self.flatten_dim = hidden_channels[1] * 7 * 7

        # Final Projection: (B, 1568) -> (B, output_dim)
        self.fc = nn.Linear(self.flatten_dim, output_dim)

        # Init weights (Orthogonal init matches legacy notebook)
        nn.init.orthogonal_(self.fc.weight)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, 1, 28, 28)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return self.fc(x)
