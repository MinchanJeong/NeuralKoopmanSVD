# koopmansvd/data/base.py
from abc import ABC, abstractmethod
from typing import Any
from torch.utils.data import Dataset


class ShapeError(Exception):
    pass


class BaseContextDataset(Dataset, ABC):
    """
    Abstract base class for Koopman datasets.
    """

    def __init__(self, time_lag: int = 1):
        self.time_lag = time_lag

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Returns a data sample consisting of Input-Target pairs for Koopman dynamics.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
