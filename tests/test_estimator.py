import numpy as np
from koopmansvd.models.inference.estimator import KoopmanEstimator


def test_estimator_lifecycle():
    """Test partial_fit, finalize, and predict."""
    rank = 5
    batch_size = 10
    dim_x = 20

    estimator = KoopmanEstimator(rank=rank, use_cca=True)

    # Mock Data
    f_t = np.random.randn(batch_size, rank)
    g_t = np.random.randn(batch_size, rank)
    f_next = np.random.randn(batch_size, rank)
    g_next = np.random.randn(batch_size, rank)
    x_raw = np.random.randn(batch_size, dim_x)
    y_raw = np.random.randn(batch_size, dim_x)

    # 1. Partial Fit
    estimator.partial_fit(f_t, g_t, f_next, g_next, x_raw, y_raw)
    assert estimator._train_n == batch_size

    # 2. Finalize
    estimator.finalize()
    assert estimator._is_fitted

    # Check shape directly
    K_shape = estimator.operators["ali"]["fwd"]["K"].shape
    assert K_shape == (rank, rank)

    # 3. Predict
    # Forward prediction
    pred = estimator.predict(f_t, g_t, t=1, mode="aligned")

    # Predict returns reconstructed state in original space (dim_x), NOT latent space (rank)
    assert pred.shape == (batch_size, dim_x)
