# koopmansvd/models/metrics.py
import logging
import torch
from torchmetrics import Metric
import numpy as np

from koopmansvd.models.inference import linalg

logger = logging.getLogger(__name__)


class KoopmanScoreMetric(Metric):
    """
    Computes Koopman Operator approximation scores (VAMP-2 or VAMP-E) in a streaming fashion.

    This metric accumulates sufficient statistics (empirical second-moment matrices)
    over the entire dataset (or epoch) to ensure numerical stability before calculating the score.

    VAMP-2 vs. VAMP-E (Wu & Noé, 2019):
    - VAMP-2:
      Maximizes the sum of squared singular values of the whitened Koopman operator.
      Intuitively, this measures the amount of dynamic information captured by the model.

    - VAMP-E (VAMP-Error):
      Measures the approximation accuracy of the fixed operator $\hat{\mathcal{K}}$*
      (learned from training) when applied to the validation distribution.
      It corresponds to minimizing the Hilbert-Schmidt error: || \hat{\mathcal{K}} - \mathcal{K} ||_{HS(\mathcal{D}_{val})} $$

    Why VAMP-E is better for Validation:
      Directly calculating VAMP-2 on validation data implicitly **re-fits** a new
      operator to the validation set (optimistic bias). VAMP-E strictly evaluates
      how well the **frozen training model** generalizes to unseen data.

    Notation (matches Section 2.3.1 of the paper):
        M_f: \hat{M}_{\rho_0}[f] (Covariance of current state features)
        M_g: \hat{M}_{\rho_1}[g] (Covariance of future state features)
        T_fg: \hat{T}[f, g]     (Cross-covariance)
    """

    # DDP compatible
    full_state_update = True

    def __init__(self, feature_dim: int, schatten_norm: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.schatten_norm = schatten_norm

        # States for accumulation (DDP compatible)
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Second Moment Accumulators
        self.add_state(
            "accum_M_f",
            default=torch.zeros(feature_dim, feature_dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "accum_M_g",
            default=torch.zeros(feature_dim, feature_dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "accum_T_fg",
            default=torch.zeros(feature_dim, feature_dim),
            dist_reduce_fx="sum",
        )

        # Reference CCA components from Training set (for VAMP-E)
        self.ref_cca = None

    def update(self, f: torch.Tensor, g: torch.Tensor):
        """
        Accumulate statistics for second-moment matrices.
        Args:
            f: Feature of current state f(x). Shape: (Batch, Dim)
            g: Feature of future state g(x'). Shape: (Batch, Dim)
        """
        batch_size = f.shape[0]
        self.n_samples += batch_size

        # Accumulate Second Moments
        # M_f = \sum f^T f, etc.
        self.accum_M_f += f.T @ f
        self.accum_M_g += g.T @ g
        self.accum_T_fg += f.T @ g

    def set_ref_stats(self, cca_components: linalg.CCAComponents):
        """
        Sets reference CCA components derived from the Training set.
        Calling this enables VAMP-E calculation mode for validation.

        Args:
            cca_components: Precomputed CCAComponents containing whitening matrices and singular vectors.
        """
        self.ref_cca = cca_components

    def compute(self):
        """
        Computes the Koopman Score based on the available context.

        1. Validation Mode (VAMP-E): If `set_ref_stats` was called with training statistics.
           Computes the operator approximation error relative to the training operator.
           Formula: Derived from || K_train - K_val ||_HS^2

        2. Training/Self-Consistent Mode (VAMP-2): If no reference stats are present.
           Computes the standard VAMP-2 score assuming the current batch defines the operator.
           Formula: || M_f^{-1/2} T_{fg} M_g^{-1/2} ||_F^2

        Returns:
            torch.Tensor: The calculated score (scalar).
        """
        if self.n_samples < 2:
            return torch.tensor(0.0, device=self.accum_M_f.device)

        if self.ref_cca is not None:
            # --- [Mode 1] VAMP-E Calculation ---
            # Use numpy-based linear algebra from linalg.py for consistency with reference components
            M_f = (self.accum_M_f / self.n_samples).cpu().numpy()
            M_g = (self.accum_M_g / self.n_samples).cpu().numpy()
            T_fg = (self.accum_T_fg / self.n_samples).cpu().numpy()

            scores = linalg.compute_vamp_scores(M_f, T_fg, M_g, self.ref_cca)

            # Return VAMP-E (VAMP-Error score)
            return torch.tensor(scores["vampe"], device=self.accum_M_f.device)
        else:
            # --- [Mode 2] VAMP-2 Calculation (Self-Consistent) ---
            logger.debug(
                "Reference CCA stats not set. Defaulting to self-consistent VAMP-2 score."
            )

            M_f = self.accum_M_f / self.n_samples
            M_g = self.accum_M_g / self.n_samples
            T_fg = self.accum_T_fg / self.n_samples

            # Calculate VAMP-2 via Least Squares for numerical stability
            # We want Trace( M_f^{-1} T_{fg} M_g^{-1} T_{fg}^T )
            # 1. Solve M_f * X = T_{fg}  ->  X = M_f^{-1} T_{fg}
            try:
                X = torch.linalg.lstsq(M_f, T_fg).solution
                # 2. Solve M_g * Y = T_{fg}^T  ->  Y = M_g^{-1} T_{fg}^T
                Y = torch.linalg.lstsq(M_g, T_fg.T).solution
                return torch.trace(X @ Y)
            except RuntimeError as e:
                # Fallback for extremely ill-conditioned matrices during early training
                logger.warning(
                    f"VAMP-2 calculation failed due to singular matrix: {e}. Returning 0.0"
                )
                return torch.tensor(0.0, device=self.accum_M_f.device)


def directed_hausdorff_distance(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """
    Computes the directed Hausdorff distance from set_a to set_b in the complex plane.
    d(A, B) = max_{a in A} min_{b in B} |a - b|

    Args:
        set_a: Array of complex/real numbers (Estimated Eigenvalues).
        set_b: Array of complex/real numbers (Ground Truth Eigenvalues).
    """
    if len(set_a) == 0:
        return np.inf
    if len(set_b) == 0:
        return np.inf

    # Calculate pairwise distances matrix: (len(a), len(b))
    dists = np.abs(set_a[:, None] - set_b[None, :])

    # For each point in A, find distance to nearest point in B
    min_dists = np.min(dists, axis=1)

    # The directed Hausdorff distance is the maximum of these minimum distances
    return float(np.max(min_dists))
