# koopmansvd/models/metrics.py
import logging
import torch
from torchmetrics import Metric
import numpy as np
from typing import Dict, Union
from koopmansvd.models.inference import linalg

logger = logging.getLogger(__name__)


class KoopmanScoreMetric(Metric):
    r"""
    Computes VAMP-2 and VAMP-E scores using test moments and training CCA components.

    Definitions (following Table 1 of Jeong et al., 2025):
        Let f_tilde, g_tilde be the whitened basis functions on the Test set (using Training whiteners).
            f_tilde = (M[f]_train)^{-1/2} @ f_test
            g_tilde = (M[g]_train)^{-1/2} @ g_test

        Let U, V be the singular vectors and S (Sigma) be the singular values from Training.
            T[f_tilde, g_tilde]_train = U @ S @ V.T

        We define the 'Projected Test Moments' (Unscaled correlations in training subspace):
            C_00 = U.T @ E[f_tilde @ f_tilde.T]_test @ U  = U.T @ (W_f @ M00 @ W_f) @ U
            C_01 = U.T @ E[f_tilde @ g_tilde.T]_test @ V  = U.T @ (W_f @ M01 @ W_g) @ V
            C_11 = V.T @ E[g_tilde @ g_tilde.T]_test @ V  = V.T @ (W_g @ M11 @ W_g) @ V

        We define the 'Aligned Singular Functions' and their Moments on the Test set:
            phi_tilde(x)  = S^{1/2} @ U.T @ f_tilde(x)
            psi_tilde(x') = S^{1/2} @ V.T @ g_tilde(x')

            M_phi  = E[phi_tilde @ phi_tilde.T]  = S^{1/2} @ C_00 @ S^{1/2}
            M_psi  = E[psi_tilde @ psi_tilde.T]  = S^{1/2} @ C_11 @ S^{1/2}
            T_stat = E[phi_tilde @ psi_tilde.T]  = S^{1/2} @ C_01 @ S^{1/2}

    Metrics:
        1. (Validation) VAMP-E (VAMP-Error): Evaluates the trained Operator.
           Measures the approximation error (Hilbert-Schmidt norm) of the *fixed* training operator
           against the validation dynamics. It evaluates the difference between true and
           trained Koopman operator on test data, therefore both the learned directions (U, V)
           and the learned singular values (S).

           Score = 2 * tr(T_stat) - tr(M_phi @ M_psi)
                 = 2 * tr(S @ C_01) - tr(S @ C_00 @ S @ C_11)  (using cyclic property of trace)

        2. (Validation) Projected VAMP-2 (R2 Score): Evaluates the trained Subspace.
           Measures the quality of the *subspace* spanned by the learned features (U, V).

           Due to the whitening (normalization) step in VAMP-2 calculation, this metric is
           *invariant* to the scaling of the features. Therefore, even if the learned
           singular values (S) from training do not match the validation timescales,
           this score can still be high if the learned singular vectors (U, V) correctly
           span the dominant dynamics. It effectively measures the correlation capacity
           of the learned subspace.

           Intuition: High R2 but Low VAMP-E implies the model captured the correct
           physical shapes (reaction coordinates) but estimated incorrect timescales.

           Score =  || M_phi^{-1/2} @ T_stat @ M_psi^{-1/2} ||_F^2
                 =  || C_00^{-1/2} @ C_01 @ C_11^{-1/2} ||_F^2

        3. (Training) Naive VAMP-2:
           Condition: No reference statistics set (ref_cca is None).
           Computes the standard VAMP-2 score using only the current training epoch statistics.
           Score = || (M[f]_train)^{-1/2} @ E[f @ g.T]_train @ (M[g]_train)^{-1/2} ||_F^2
           Measures the maximum potential correlation the feature extractor can capture
           on the current training data,
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

    def compute(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            - Training Mode: torch.Tensor (Naive VAMP-2 scalar)
            - Validation Mode: Dict {'val/vamp_e': ..., 'val/proj_vamp_2': ...}
        """
        if self.n_samples < 2:
            return torch.tensor(0.0, device=self.accum_M_f.device)

        if self.ref_cca is not None:
            # --- [Mode 1] VAMP-E, Projected VAMP-2 Calculation ---
            # Use numpy-based linear algebra from linalg.py for consistency with reference components
            M_f = (self.accum_M_f / self.n_samples).cpu().numpy()
            M_g = (self.accum_M_g / self.n_samples).cpu().numpy()
            T_fg = (self.accum_T_fg / self.n_samples).cpu().numpy()

            scores = linalg.compute_vamp_scores(M_f, T_fg, M_g, self.ref_cca)

            return {
                "vamp_e": torch.tensor(scores["vampe"], device=self.accum_M_f.device),
                "vamp_2": torch.tensor(scores["vamp2"], device=self.accum_M_f.device),
            }
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
