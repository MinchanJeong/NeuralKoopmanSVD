# tests/test_components.py
import torch
import numpy as np
import sys
import types

from koopmansvd.data.collate import KoopmanCollate
from koopmansvd.models.metrics import directed_hausdorff_distance


# -------------------------------------------------------------------------
# 1. Environment Mocking
# -------------------------------------------------------------------------
class MockProperties:
    R = "_positions"
    Z = "_atomic_numbers"
    n_atoms = "_n_atoms"
    idx_m = "_idx_m"
    cell = "_cell"
    offsets = "_offsets"
    idx_i = "_idx_i"
    idx_j = "_idx_j"
    forces = "forces"


if "schnetpack" not in sys.modules:
    schnetpack = types.ModuleType("schnetpack")
    schnetpack.properties = MockProperties
    sys.modules["schnetpack"] = schnetpack
    sys.modules["schnetpack.properties"] = MockProperties
else:
    import schnetpack.properties


# -------------------------------------------------------------------------
# 2. Metric Tests
# -------------------------------------------------------------------------
def test_hausdorff_distance():
    """Test metric computation for complex eigenvalues."""
    # Case 1: Real numbers
    # Set A: {1, 2}, Set B: {1, 5}
    # d(A, B) = max( min(|1-1|, |1-5|), min(|2-1|, |2-5|) )
    #         = max( 0, 1 ) = 1
    a = np.array([1, 2])
    b = np.array([1, 5])
    dist = directed_hausdorff_distance(a, b)
    assert np.isclose(dist, 1.0)

    # Case 2: Complex plane
    # A: {i}, B: {0} -> dist = |i - 0| = 1.0
    a_c = np.array([1j])
    b_c = np.array([0])
    dist_c = directed_hausdorff_distance(a_c, b_c)
    assert np.isclose(dist_c, 1.0)


# -------------------------------------------------------------------------
# 3. Collation Tests (Data Pipeline)
# -------------------------------------------------------------------------
def test_koopman_collate_tensor():
    """Test standard tensor collation (Stacking)."""
    collate = KoopmanCollate()

    # Batch of 2 pairs. Each pair is (x_t, x_{t+1})
    # Dim = 10
    x1, y1 = torch.randn(10), torch.randn(10)
    x2, y2 = torch.randn(10), torch.randn(10)
    batch = [(x1, y1), (x2, y2)]

    collated_x, collated_y = collate(batch)

    # Must stack: Shape becomes (Batch=2, Dim=10)
    assert collated_x.shape == (2, 10)
    assert collated_y.shape == (2, 10)

    # Value check
    assert torch.equal(collated_x[0], x1)
    assert torch.equal(collated_y[1], y2)


def test_koopman_collate_molecular():
    """
    Test molecular graph collation (Concatenation + Indexing).
    Checks if schnet_collate_fn correctly creates large batch graphs.
    """
    # Import locally to avoid top-level import errors if schnetpack is missing
    from koopmansvd.data.molecular import schnet_collate_fn

    collate = KoopmanCollate(base_collate_fn=schnet_collate_fn)

    # Molecule 1: 2 atoms
    m1_x = {
        "_positions": torch.randn(2, 3),
        "_atomic_numbers": torch.tensor([1, 6]),
        "_n_atoms": torch.tensor([2]),
    }
    # Deepcopy for safe dummy data
    m1_y = {k: v.clone() for k, v in m1_x.items()}

    # Molecule 2: 3 atoms
    m2_x = {
        "_positions": torch.randn(3, 3),
        "_atomic_numbers": torch.tensor([1, 1, 8]),
        "_n_atoms": torch.tensor([3]),
    }
    m2_y = {k: v.clone() for k, v in m2_x.items()}

    batch = [(m1_x, m1_y), (m2_x, m2_y)]
    coll_x, coll_y = collate(batch)

    # Check 1: Feature Concatenation (Total atoms = 2 + 3 = 5)
    expected_atoms = 2 + 3
    assert coll_x["_positions"].shape == (expected_atoms, 3)
    assert coll_x["_atomic_numbers"].shape == (expected_atoms,)

    # Check 2: Batch Index Map (_idx_m)
    # Molecule 0 has 2 atoms -> indices [0, 0]
    # Molecule 1 has 3 atoms -> indices [1, 1, 1]
    expected_idx_m = torch.tensor([0, 0, 1, 1, 1])
    assert torch.equal(coll_x["_idx_m"].cpu(), expected_idx_m)

    # Check 3: System-wise property stacking
    # n_atoms should be stacked to (Batch=2, 1)
    assert torch.equal(coll_x["_n_atoms"].squeeze(), torch.tensor([2, 3]))
