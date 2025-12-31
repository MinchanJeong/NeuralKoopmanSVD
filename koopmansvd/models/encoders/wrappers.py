# koopmansvd/models/encoders/wrappers.py
import torch
import numpy as np
from koopmansvd.models.encoders.base import BaseEncoder


class CenteringWrapper(BaseEncoder):
    """
    Prepends a constant '1' feature for centering.
    Output dim increases by 1.
    """

    def __init__(self, encoder: BaseEncoder):
        super().__init__()
        self.encoder = encoder

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim + 1

    def forward(self, x) -> torch.Tensor:
        feats = self.encoder(x)
        ones = torch.ones(feats.size(0), 1, device=feats.device, dtype=feats.dtype)
        return torch.cat([ones, feats], dim=1)


class BatchL2NormalizationWrapper(BaseEncoder):
    """
    Applies Batch L2 Normalization (used in NeuralEF/LoRA).
    """

    def __init__(self, encoder: BaseEncoder, momentum: float = 0.9):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum

        dim = encoder.output_dim
        self.register_buffer("running_norm", torch.ones(1, dim))
        self.register_buffer("running_norm_sq", torch.ones(1, dim))

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    def forward(self, x) -> torch.Tensor:
        feats = self.encoder(x)  # (B, D)

        if self.training:
            # Calculate batch L2 norm per feature
            batch_norm = feats.norm(dim=0, keepdim=True) / np.sqrt(feats.size(0))

            # Update using squared values (Unbiased/RMS logic)
            current_sq = batch_norm.detach() ** 2
            self.running_norm_sq = (
                self.momentum * self.running_norm_sq + (1 - self.momentum) * current_sq
            )

            return feats / (batch_norm + 1e-8)
        else:
            # Take sqrt of the running squared average
            running_norm = torch.sqrt(self.running_norm_sq)
            return feats / (running_norm + 1e-8)
