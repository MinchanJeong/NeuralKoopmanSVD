# koopmansvd/data/context.py
import logging
import bisect
from typing import Any, Dict, Union, Sequence
import numpy as np
import torch
from koopmansvd.data.base import BaseContextDataset

logger = logging.getLogger(__name__)


class TensorContextDataset(BaseContextDataset):
    """
    Memory-efficient Dataset for generic tensor trajectories.

    Supports both in-memory arrays (NumPy, Tensor) and lazy-loading objects
    (numpy.memmap, zarr, xarray, h5py) to handle large-scale datasets like ERA5
    without immediate OOM errors.
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, Sequence[Any]],
        observables: Dict[str, Sequence[Any]] = None,
        time_lag: int = 1,
        backend: str = "torch",
    ):
        """
        Args:
            data: Trajectory data. Can be a single array/tensor or a list of them.
                  Supports lazy-loading objects (must implement __len__ and __getitem__).
            observables: Optional dictionary of auxiliary observables (not fully implemented).
            time_lag: Time delay (tau) between input and target.
            backend: Target format for __getitem__ output ('torch' or 'numpy').
        """
        super().__init__(time_lag)

        # 1. Standardize Input to List of Trajectories
        # NOTE: We do NOT convert data to backend here to preserve lazy-loading capabilities.
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, torch.Tensor) and data.is_cuda:
                logger.warning(
                    "TensorContextDataset received a CUDA tensor. Moving to CPU.\n"
                    "Reason: CUDA tensors cannot be shared across DataLoader worker processes (num_workers > 0).\n"
                    "For optimal performance with large datasets, please use 'koopmansvd.data.io.load_trajectory_data' "
                    "to load data as memory-mapped files or lazy objects (Zarr/Xarray), instead of pre-loading to GPU."
                )
                data = data.cpu()
            self.trajectories = [data]
        elif hasattr(data, "shape") and hasattr(data, "__getitem__"):
            # Handles xarray, zarr, h5py objects
            self.trajectories = [data]
        elif isinstance(data, Sequence) and not isinstance(data, str):
            self.trajectories = data
        else:
            raise TypeError("Data must be an array-like object or a sequence of them.")

        self.backend = backend
        self.observables = observables or {}

        # 2. Build Index Mapping
        self.window_span = 1 + time_lag

        self.traj_lengths = [len(t) for t in self.trajectories]
        self.valid_windows_per_traj = [
            max(0, length - self.window_span + 1) for length in self.traj_lengths
        ]

        # Accessing len() on mmap/zarr is efficient (reads metadata only)
        self.traj_lengths = [len(t) for t in self.trajectories]
        self.valid_windows_per_traj = [
            max(0, length - self.window_span + 1) for length in self.traj_lengths
        ]

        # Cumulative sum for fast indexing
        self.cumulative_indices = np.cumsum(self.valid_windows_per_traj)
        if len(self.cumulative_indices) > 0:
            self.total_windows = self.cumulative_indices[-1]
        else:
            self.total_windows = 0

    def _convert_window(self, window_data: Any) -> Union[torch.Tensor, np.ndarray]:
        """
        Converts a small sliced window into the target backend format.
        Triggers actual disk I/O for lazy objects.
        """
        # 1. Trigger computation for Lazy Objects
        if hasattr(window_data, "values"):  # xarray
            window_data = window_data.values
        elif hasattr(window_data, "compute"):  # dask
            window_data = window_data.compute()

        # 2. Convert to Target Backend
        if self.backend == "torch":
            if torch.is_tensor(window_data):
                return window_data.float()

            # Ensure data is in RAM and float32.
            # np.array(..., copy=True) handles read-only memmap buffers safely.
            arr = np.array(window_data, dtype=np.float32)
            return torch.from_numpy(arr)

        elif self.backend == "numpy":
            if isinstance(window_data, torch.Tensor):
                return window_data.detach().cpu().numpy()
            return np.asanyarray(window_data, dtype=np.float32)

        return window_data

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx: int) -> Union[torch.Tensor, np.ndarray]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of bounds.")

        # 1. Locate the specific trajectory
        traj_idx = bisect.bisect_right(self.cumulative_indices, idx)

        # 2. Calculate local start index within that trajectory
        if traj_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_indices[traj_idx - 1]

        # 3. Slice the window (Lazy Access)
        traj = self.trajectories[traj_idx]

        # Explicit int casting for compatibility with strict custom arrays
        start = int(local_idx)
        end = start + self.window_span
        step = self.time_lag

        # NOTE: This slicing triggers disk I/O for mmap/zarr objects.
        window_data = traj[start:end:step]

        # 4. Convert and return
        return self._convert_window(window_data)
