# koopmansvd/data/collate.py
import torch
from typing import List, Any, Tuple, Callable


def default_tensor_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)


class KoopmanCollate:
    """
    Handles temporal pairing for Koopman Dynamics.
    Delegates spatial batching to `base_collate_fn`.
    """

    def __init__(self, base_collate_fn: Callable = default_tensor_collate):
        self.base_collate_fn = base_collate_fn

    def __call__(self, batch: List[Any]) -> Tuple[Any, Any]:
        """
        Args:
            batch: List of [x_t, x_{t+tau}] pairs.
        Returns:
            (Batch_X, Batch_Y) where each is collated by base_collate_fn.
        """
        if not batch:
            return None, None

        # 1. Unzip the pairs
        # batch = [[x1, y1], [x2, y2], ...]
        # list_x = [x1, x2, ...]
        list_x = [sample[0] for sample in batch]
        list_y = [sample[1] for sample in batch]

        # 2. Apply specific collation strategy (Tensor Stack vs Graph Concat)
        batch_x = self.base_collate_fn(list_x)
        batch_y = self.base_collate_fn(list_y)

        return batch_x, batch_y
