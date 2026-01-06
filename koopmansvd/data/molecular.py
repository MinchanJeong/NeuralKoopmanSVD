# koopmansvd/data/molecular.py
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
from koopmansvd.data.base import BaseContextDataset

import ase.db
from schnetpack.data import ASEAtomsData
from schnetpack.transform import CastTo32, MatScipyNeighborList
import schnetpack.properties as properties

logger = logging.getLogger(__name__)


# --- Fix for ASE/SchNetPack Compatibility ---
class FixedASEAtomsData(ASEAtomsData):
    """
    Subclass of ASEAtomsData that fixes compatibility issues with newer ASE versions.

    Problem:
      Newer ASE enforces strict connection management for SQLite databases.
      Accessing 'metadata' requires an active connection (self.connection is not None).
      SchNetPack's ASEAtomsData assumes the connection is always open after init,
      but ASE now initializes it in a closed state or requires a context manager.

    Solution:
      Override the 'metadata' property (which is called during SchNetPack's __init__).
      If the connection is closed, manually trigger __enter__() to open it and keep it
      open, mimicking the legacy behavior SchNetPack expects.
    """

    @property
    def metadata(self):
        # Check if the internal ASE database connection exists
        if hasattr(self, "conn") and self.conn is not None:
            # If the actual sqlite connection is None, force it open
            if getattr(self.conn, "connection", None) is None:
                # __enter__() is the standard ASE method to establish the DB connection
                self.conn.__enter__()

        # Proceed to access the metadata property from the parent class
        return super().metadata

    def __del__(self):
        # Good practice to close the connection when the dataset is destroyed,
        # although Python/OS usually handles this.
        if hasattr(self, "conn") and self.conn is not None:
            if getattr(self.conn, "connection", None) is not None:
                self.conn.__exit__(None, None, None)


def schnet_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of molecular dictionaries into a single batch dictionary.

    Standard GNN Batching Strategy:
    Instead of stacking dimensions (B, N, 3), we concatenate all atoms along the
    first dimension (B*N, 3). This effectively creates one large graph containing
    disconnected components (molecules).

    Args:
        batch_list: List of dictionaries, each representing one molecule/frame.

    Returns:
        collated_batch: A single dictionary with concatenated tensors and updated indices.
    """
    if not batch_list:
        return {}

    # --- Step 1: Collect Atom Counts ---
    # We need to know how many atoms are in each sample to generate batch indices.
    n_atoms_list = []
    for i, sample in enumerate(batch_list):
        n = sample.get(properties.n_atoms)
        if n is None:
            # Infer n_atoms from shape of Z (atomic numbers) or R (positions)
            if properties.Z in sample and isinstance(
                sample[properties.Z], torch.Tensor
            ):
                n = sample[properties.Z].shape[0]
            elif properties.R in sample and isinstance(
                sample[properties.R], torch.Tensor
            ):
                n = sample[properties.R].shape[0]
            else:
                raise KeyError(
                    f"'{properties.n_atoms}' key missing in batch item {i} and cannot infer."
                )
        elif isinstance(n, torch.Tensor):
            n = int(n.item())
        n_atoms_list.append(int(n))

    n_atoms_tensor = torch.tensor(n_atoms_list, dtype=torch.long)

    # Compute offsets for indices (neighbor lists)
    # If sample 0 has 10 atoms, indices for sample 1 must be shifted by 10.
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), n_atoms_tensor[:-1]]), dim=0)

    collated_batch = {}

    # --- Step 2: Generate Batch Indices (_idx_m) ---
    # This tensor maps every atom to its corresponding batch index [0, 0, ..., 1, 1, ...].
    # Crucial for aggregation (pooling) operations later in the model.

    # Determine device from the first tensor found in the batch
    device = "cpu"
    for v in batch_list[0].values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break

    batch_idx = []
    for i, n in enumerate(n_atoms_list):
        batch_idx.append(torch.full((n,), i, dtype=torch.long, device=device))
    collated_batch[properties.idx_m] = torch.cat(batch_idx, dim=0)

    # --- Step 3: Process and Collate All Properties ---
    keys = set().union(*(d.keys() for d in batch_list))

    # Define keys that are strictly atom-wise and must be concatenated (N_total, ...)
    atom_keys = {
        properties.R,
        properties.Z,
        properties.forces,
        properties.dipole_moment,
    }

    for key in keys:
        if key == properties.idx_m:
            continue  # Already handled

        values = [b.get(key) for b in batch_list]

        # Skip if all values for this key are None
        if all(v is None for v in values):
            continue

        first_valid = next((v for v in values if v is not None), None)
        if first_valid is None:
            continue

        # A. Handle Neighbor List Indices (Shift by offset)
        if key in [properties.idx_i, properties.idx_j]:
            shifted_values = []
            for i, v in enumerate(values):
                if v is None:
                    continue
                shifted_values.append(v + offsets[i])
            if shifted_values:
                collated_batch[key] = torch.cat(shifted_values, dim=0)

        # B. Handle PBC Offsets
        elif key == properties.offsets:
            valid_vals = [v for v in values if v is not None]
            if valid_vals:
                collated_batch[key] = torch.cat(valid_vals, dim=0)

        # C. Handle System-level Scalars (n_atoms)
        elif key == properties.n_atoms:
            collated_batch[key] = n_atoms_tensor.unsqueeze(1)  # (B, 1)

        # D. Handle Unit Cells (Stacking, not Concatenating)
        elif key == properties.cell:
            valid_cells = []
            for v in values:
                if v is None:
                    # Fallback for missing cell
                    valid_cells.append(
                        torch.zeros(
                            (1, 3, 3),
                            device=first_valid.device,
                            dtype=first_valid.dtype,
                        )
                    )
                elif v.ndim == 2:
                    valid_cells.append(v.unsqueeze(0))
                else:
                    valid_cells.append(v)
            collated_batch[key] = torch.cat(valid_cells, dim=0)  # Result: (B, 3, 3)

        # E. General Tensor Handling
        elif isinstance(first_valid, torch.Tensor):
            # Case E-1: Atom-wise properties -> Concatenate
            if key in atom_keys or (key.startswith("_") and "positions" in key):
                valid_vals = [v for v in values if v is not None]
                collated_batch[key] = torch.cat(valid_vals, dim=0)

            # Case E-2: Heuristic inference for atom-wise properties
            # If the first dim matches the number of atoms for that sample, assume it's atom-wise.
            elif all(
                v is not None and v.shape[0] == n for v, n in zip(values, n_atoms_list)
            ):
                collated_batch[key] = torch.cat(values, dim=0)

            # Case E-3: System-wise properties -> Stack (Create Batch Dim)
            else:
                try:
                    cleaned_values = []
                    for v in values:
                        if v is None:
                            cleaned_values.append(torch.zeros_like(first_valid))
                        else:
                            cleaned_values.append(v)
                    collated_batch[key] = torch.stack(cleaned_values, dim=0)
                except RuntimeError:
                    # Fallback to cat if stack fails (mismatched shapes)
                    valid_vals = [v for v in values if v is not None]
                    collated_batch[key] = torch.cat(valid_vals, dim=0)

        else:
            # F. Non-Tensor Data (Strings, Lists) -> Return as List
            collated_batch[key] = values

    return collated_batch


class MolecularContextDataset(BaseContextDataset):
    """
    Dataset for Molecular Dynamics data (SchNetPack compatible).
    Yields pairs of atomic graphs (Dict[str, Tensor]).
    """

    def __init__(
        self,
        db_path: str,
        cutoff: float = 6.0,
        time_lag: int = 1,
    ):
        super().__init__(time_lag)
        self.db_path = Path(db_path)
        self.cutoff = cutoff

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Open temporarily just to get the length, then close.
        # This prevents broken SQLite connections when forking processes.
        self._len = 0
        self.window_span = 1 + self.time_lag

        # Temporary connection for length calculation
        with ase.db.connect(str(self.db_path)) as conn:
            raw_len = len(conn)
        self._len = max(0, raw_len - self.window_span + 1)

        # Define transforms (to be used later)
        self.transforms = [CastTo32(), MatScipyNeighborList(cutoff=cutoff)]

        # This will hold the actual dataset instance per worker
        self.dataset = None

    def _get_dataset(self):
        """Lazy initialization of the dataset to ensure thread/process safety."""
        if self.dataset is None:
            self.dataset = FixedASEAtomsData(
                str(self.db_path), transforms=self.transforms
            )
        return self.dataset

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        """
        Returns a list of atomic graphs: [Structure_t, Structure_{t+tau}, ...]
        Length of list == 2
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of bounds.")

        # Ensure dataset is initialized for this process
        ds = self._get_dataset()

        window_items = []
        for i in range(2):
            # Calculate exact frame index
            frame_idx = idx + (i * self.time_lag)
            window_items.append(ds[frame_idx])

        return window_items
