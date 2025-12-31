# koopmansvd/models/encoders/base.py
import torch.nn as nn
import torch
from abc import abstractmethod


class BaseEncoder(nn.Module):
    """
    Abstract base class for all encoders.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns the dimension of the output feature vector."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Input data. Can be Tensor (MLP) or Dict (SchNet).
        Returns:
            Tensor of shape (Batch_Size, Output_Dim)
        """
        raise NotImplementedError
