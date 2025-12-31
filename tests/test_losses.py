import torch
from koopmansvd.models.losses import NestedLoRALoss


def test_lora_loss_shape():
    """Ensure loss returns a scalar."""
    batch_size = 32
    n_modes = 5

    loss_fn = NestedLoRALoss(n_modes=n_modes, nesting="jnt")

    f = torch.randn(batch_size, n_modes)
    g = torch.randn(batch_size, n_modes)

    loss = loss_fn(f, g)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_lora_perfect_correlation():
    """If f == g and they are orthonormal, loss should be minimal (-k)."""
    n_modes = 4
    batch_size = 100

    # Create orthonormal-ish data E[f^T f] = I
    # Base pattern has norm 1. To make E[f^T f] = I, we need scaling.
    # Current M = (1/B) * f^T f.
    # Pattern: [1, 0..], [0, 1..] repeats.
    # f^T f is diagonal with entries (B/n_modes).
    # M = (1/B) * (B/n_modes) * I = (1/n_modes) * I.
    # We want M = I. So we need to scale f by sqrt(n_modes).

    f = torch.eye(n_modes).repeat(batch_size // n_modes, 1)
    f *= n_modes**0.5  # Scale to make covariance Identity
    g = f.clone()

    loss_fn = NestedLoRALoss(n_modes=n_modes, nesting=None)
    loss = loss_fn(f, g)

    expected = -float(n_modes)
    # Allow some numerical error
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-1)


def test_masks_creation():
    """Verify Joint Nesting masks are created correctly."""
    n_modes = 3
    loss_fn = NestedLoRALoss(n_modes=n_modes, nesting="jnt")

    assert loss_fn.vec_mask is not None
    assert loss_fn.mat_mask is not None

    # vec_mask is registered with unsqueeze(0) for broadcasting (1, K)
    assert loss_fn.vec_mask.shape == (1, n_modes)
    assert loss_fn.mat_mask.shape == (n_modes, n_modes)
