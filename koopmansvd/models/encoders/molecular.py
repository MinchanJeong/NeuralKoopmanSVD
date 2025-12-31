# koopmansvd/models/encoders/molecular.py
import logging
import torch
import torch.nn as nn
from typing import Dict

# Try importing SchNetPack
try:
    import schnetpack.nn as snn
    import schnetpack.properties as properties
    from schnetpack.representation import SchNet
    from schnetpack.atomistic import PairwiseDistances

    _HAS_SCHNET = True
except ImportError:
    _HAS_SCHNET = False

from koopmansvd.models.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


class SchNetSystemEncoder(BaseEncoder):
    """
    Wraps SchNet to output system-wise features directly.
    Handles:
      1. Pairwise Distance Computation
      2. SchNet Representation (Atom-wise)
      3. Aggregation (Atom-wise -> System-wise)
      4. Final Projection
    """

    def __init__(
        self,
        n_atom_basis: int = 64,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff: float = 6.0,
        output_dim: int = 16,  # n_modes
        aggregation: str = "mean",  # 'mean' or 'sum'
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if not _HAS_SCHNET:
            raise ImportError("schnetpack is required for SchNetSystemEncoder.")

        self._output_dim = output_dim
        self.aggregation = aggregation
        self.cutoff = cutoff

        # 1. SchNet Components
        self.pwise_dist = PairwiseDistances()
        self.radial_basis = snn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        self.representation = SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=self.radial_basis,
            cutoff_fn=snn.CosineCutoff(cutoff),
        )

        # 2. Output Projection (Atom-wise projection before aggregation)
        # Note: Projecting before aggregation is standard in SchNetPack
        layers = [nn.Linear(n_atom_basis, output_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(output_dim, affine=False))
        self.output_layer = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Dictionary containing 'R', 'Z', 'idx_m', etc.
        Returns:
            System-wise features: (Batch_Size, Output_Dim)
        """
        # 1. Prepare Inputs (Distances)
        if properties.R not in inputs:
            raise KeyError("Input dict missing 'R' (positions).")

        if properties.idx_m not in inputs:
            raise KeyError("Input dict missing 'idx_m' (batch indices).")

        idx_m = inputs[properties.idx_m]  # (Total_Atoms,) - System index for each atom

        if idx_m.numel() == 0:
            logger.warning("Input contains no atoms. Returning zero features.")
            return torch.zeros(
                (0, self._output_dim),
                device=idx_m.device,
                dtype=inputs[properties.R].dtype,
            )

        batch_size = int(idx_m.max().item()) + 1 if idx_m.numel() > 0 else 0

        # Compute pairwise distances if not present
        # (SchNetPack's PairwiseDistances modifies dict in-place or returns new)
        inputs = self.pwise_dist(inputs)

        # Ensure offsets exist (for non-periodic)
        if properties.offsets not in inputs:
            n_pairs = inputs[properties.idx_i].shape[0]
            inputs[properties.offsets] = torch.zeros(
                (n_pairs, 3),
                device=inputs[properties.R].device,
                dtype=inputs[properties.R].dtype,
            )

        # 2. Atom-wise Representation
        # SchNet returns dict with 'scalar_representation'
        res = self.representation(inputs)
        atom_feats = res["scalar_representation"]  # (Total_Atoms, n_atom_basis)

        # 3. Project to Feature Space
        atom_feats = self.output_layer(atom_feats)  # (Total_Atoms, output_dim)

        # 4. Aggregate to System-wise Feature
        system_feats = self._aggregate(atom_feats, idx_m, batch_size)

        return system_feats

    def _aggregate(self, atom_feats, idx_m, batch_size):
        # Initialize output
        out = torch.zeros(
            (batch_size, self.output_dim),
            dtype=atom_feats.dtype,
            device=atom_feats.device,
        )

        # Scatter Add
        # idx_m needs to be broadcasted: (Total_Atoms, 1) -> (Total_Atoms, Feat_Dim)
        idx_m_expanded = idx_m.unsqueeze(1).expand(-1, self.output_dim)
        out.scatter_add_(0, idx_m_expanded, atom_feats)

        if self.aggregation == "mean":
            # Count atoms per system
            ones = torch.ones_like(idx_m, dtype=atom_feats.dtype)
            counts = torch.zeros(
                (batch_size,), dtype=atom_feats.dtype, device=atom_feats.device
            )
            counts.scatter_add_(0, idx_m, ones)

            # Avoid division by zero
            counts = counts.clamp(min=1).unsqueeze(1)
            out = out / counts

        return out
