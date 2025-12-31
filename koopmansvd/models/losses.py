# koopmansvd/models/losses.py

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

__all__ = ["VAMPLoss", "DPLoss", "NestedLoRALoss"]

# --- Math Helpers ---


def _sqrtmh(A: torch.Tensor) -> torch.Tensor:
    """Computes the square root of a Symmetric/Hermitian positive definite matrix.
    Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_.
    Zeroes out small eigenvalues for numerical stability.

    Args:
        A: Input matrix (..., N, N).

    Returns:
        Square root of A.
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def _covariance(
    X: torch.Tensor, Y: Optional[torch.Tensor] = None, center: bool = False
) -> torch.Tensor:
    """Computes the empirical covariance matrix.

    Args:
        X: Tensor of shape (Batch, DimX).
        Y: Optional Tensor of shape (Batch, DimY). If None, computes Cov(X, X).
        center: If True, centers the data (subtracts mean) before computing.

    Returns:
        Covariance matrix of shape (DimX, DimY) (or (DimX, DimX)).
    """
    assert X.ndim == 2
    n_samples = X.shape[0]
    cov_norm = torch.rsqrt(torch.tensor(n_samples, device=X.device, dtype=X.dtype))

    if Y is None:
        _X = cov_norm * X
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _X)
    else:
        assert Y.ndim == 2
        _X = cov_norm * X
        _Y = cov_norm * Y
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
            _Y = _Y - _Y.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _Y)


def _log_fro_metric_deformation_loss(cov: torch.Tensor) -> torch.Tensor:
    """Logarithmic + Frobenius metric deformation loss (Kostic et al. 2023).

    Defined as Tr(C^2 - C - ln(C)).

    Args:
        cov: Symmetric positive-definite matrix.
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss


def _compute_second_moment(f: torch.Tensor, seq_nesting: bool = False) -> torch.Tensor:
    """Computes the second moment matrix M = 1/N * f^T f.

    If seq_nesting is True, applies partial stop-gradients to enforce sequential dependency:
      - Lower Triangle (i > j): <f_i, sg(f_j)> (Current optimizes against fixed past)
      - Upper Triangle (i < j): <sg(f_i), f_j> (Future optimizes against fixed current)
      - Diagonal: <f_i, f_i>

    Args:
        f: Tensor of shape (Batch, Modes).
        seq_nesting: If True, applies sequential nesting gradient blocking.

    Returns:
        Tensor of shape (Modes, Modes).
    """
    batch_size = f.shape[0]

    if not seq_nesting:
        return (f.T @ f) / batch_size
    else:
        # Lower: i > j
        M_lower = torch.tril(f.T @ f.detach(), diagonal=-1)
        # Upper: i < j
        M_upper = torch.triu(f.detach().T @ f, diagonal=1)
        # Diagonal
        M_diag = torch.diag((f * f).sum(dim=0))

        return (M_lower + M_upper + M_diag) / batch_size


def _kostic_regularization(
    M: torch.Tensor, h_mean: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the metric distortion regularization (Kostic et al.).

    Loss = ||M - I||_F^2 + 2 * ||mean(h)||^2 (if h_mean provided)

    Args:
        M: Covariance/Moment matrix (Modes, Modes).
        h_mean: Optional mean vector of features (Modes,).
    """
    k = M.shape[0]
    eye = torch.eye(k, device=M.device)
    reg = ((M - eye) ** 2).sum()

    if h_mean is not None:
        reg = reg + 2 * (h_mean**2).sum()
    return reg


# --- Score Functions ---


def _vamp_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    schatten_norm: int = 2,
    center_covariances: bool = True,
) -> torch.Tensor:
    """Computes the VAMP score (Wu & Noe, 2019)."""
    cov_X = _covariance(X, center=center_covariances)
    cov_Y = _covariance(Y, center=center_covariances)
    cov_XY = _covariance(X, Y, center=center_covariances)

    if schatten_norm == 2:
        # VAMP-2: Trace( (C_XX \ C_XY) @ (C_YY \ C_XY^T) )
        # Implemented stably via least squares
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        return torch.trace(M_X @ M_Y)

    elif schatten_norm == 1:
        # VAMP-1: Nuclear Norm
        sqrt_cov_X = _sqrtmh(cov_X)
        sqrt_cov_Y = _sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_XY,
                torch.linalg.pinv(sqrt_cov_Y, hermitian=True),
            ]
        )
        return torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def _deepprojection_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    relaxed: bool = True,
    metric_deformation: float = 1.0,
    center_covariances: bool = True,
) -> torch.Tensor:
    """Computes the Deep Projection score (Kostic et al. 2023)."""
    cov_X = _covariance(X, center=center_covariances)
    cov_Y = _covariance(Y, center=center_covariances)
    cov_XY = _covariance(X, Y, center=center_covariances)

    # Regularization terms
    R_X = _log_fro_metric_deformation_loss(cov_X)
    R_Y = _log_fro_metric_deformation_loss(cov_Y)

    if relaxed:
        # Relaxed DP Objective: ||C_XY||_F^2 / (||C_XX||_2 * ||C_YY||_2)
        numerator = torch.linalg.matrix_norm(cov_XY, ord="fro") ** 2
        denominator = torch.linalg.matrix_norm(cov_X, ord=2) * torch.linalg.matrix_norm(
            cov_Y, ord=2
        )
        S = numerator / (denominator + 1e-8)
    else:
        # Full DP Objective (Equivalent to VAMP-2)
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        S = torch.trace(M_X @ M_Y)

    return S - 0.5 * metric_deformation * (R_X + R_Y)


# --- Loss Classes ---


class VAMPLoss(nn.Module):
    """VAMP Loss (Wu & Noe, 2019)."""

    def __init__(
        self, schatten_norm: int = 2, center_covariances: bool = True, **kwargs
    ):
        super().__init__()
        self.schatten_norm = schatten_norm
        self.center_covariances = center_covariances

    def forward(
        self, f: torch.Tensor, g: torch.Tensor, svals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return -_vamp_score(f, g, self.schatten_norm, self.center_covariances)


class DPLoss(nn.Module):
    """Deep Projection (DP) Loss (Kostic et al., 2023)."""

    def __init__(
        self,
        relaxed: bool = True,
        metric_deformation: float = 1.0,
        center_covariances: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.relaxed = relaxed
        self.metric_deformation = metric_deformation
        self.center_covariances = center_covariances

    def forward(
        self, f: torch.Tensor, g: torch.Tensor, svals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Returns negative score because we minimize loss
        return -_deepprojection_score(
            f,
            g,
            relaxed=self.relaxed,
            metric_deformation=self.metric_deformation,
            center_covariances=self.center_covariances,
        )


class NestedLoRALoss(nn.Module):
    """
    Computes the Nested Low-Rank Approximation (LoRA) objective.

    This loss minimizes the low-rank approximation error of the Koopman operator
    without requiring matrix inversions or singular value decompositions.
    It corresponds to Equation (4) in the paper, optionally extended with
    nesting techniques (Section 3.1).

    The objective is defined as:
        L(f, g) = -2 * Tr(T[f, g]) + Tr(M_rho0[f] @ M_rho1[g])

    Args:
        n_modes (int): The rank k of the approximation (number of singular values).
        nesting (str, optional): The nesting strategy to learn ordered singular functions.
            - 'jnt' (Joint): Optimizes all modes simultaneously with a mask (Eq. 5 in NeuralSVD).
            - 'seq' (Sequential): Optimizes modes iteratively via stop-gradient.
            - None: Standard LoRA without ordering guarantees. Defaults to 'jnt'.
        reg_weight (float): Weight for Kostic regularization (Eq. 2 in paper). Defaults to 0.0.

    References:
        Jeong et al., "Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems", NeurIPS 2025.
        Section 3.1: Learning (Eq. 4).
    """

    def __init__(
        self, n_modes: int, nesting: str = "jnt", reg_weight: float = 0.0, **kwargs
    ):
        super().__init__()
        self.nesting = nesting
        self.reg_weight = reg_weight
        self.n_modes = n_modes

        if nesting == "jnt":
            if n_modes is None:
                raise ValueError("n_modes must be specified for joint nesting.")
            vec_mask, mat_mask = self._create_joint_masks(n_modes)
            self.register_buffer("vec_mask", vec_mask.unsqueeze(0))
            self.register_buffer("mat_mask", mat_mask)
        else:
            self.vec_mask, self.mat_mask = None, None

    def _create_joint_masks(self, n_modes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = np.ones(n_modes) / n_modes
        vec_mask_np = np.cumsum(weights[::-1])[::-1].copy()
        vec_tensor = torch.tensor(vec_mask_np, dtype=torch.float32)
        mat_tensor = torch.minimum(vec_tensor.unsqueeze(1), vec_tensor.unsqueeze(0))
        return vec_tensor, mat_tensor

    def forward(
        self, f: torch.Tensor, g: torch.Tensor, svals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if svals is not None:
            scale = svals.unsqueeze(0).sqrt()
            f = f * scale
            g = g * scale

        # 1. Correlation Term
        if self.nesting == "jnt":
            corr = -2 * (self.vec_mask * f * g).mean(dim=0).sum()
        else:
            corr = -2 * (f * g).mean(dim=0).sum()

        # 2. Metric Term
        is_seq = self.nesting == "seq"
        m_f = _compute_second_moment(f, seq_nesting=is_seq)
        m_g = _compute_second_moment(g, seq_nesting=is_seq)

        if self.nesting == "jnt":
            metric = (self.mat_mask * m_f * m_g).sum()
        else:
            metric = (m_f * m_g).sum()

        loss = corr + metric

        # 3. Regularization (Optional)
        if self.reg_weight > 0:
            loss += self.reg_weight * (
                _kostic_regularization(m_f, f.mean(0))
                + _kostic_regularization(m_g, g.mean(0))
            )
        return loss
