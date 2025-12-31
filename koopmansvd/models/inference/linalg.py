# koopmansvd/models/inference/linalg.py
import numpy as np
from scipy.linalg import schur, svd
from numpy.linalg import pinv
from typing import Tuple, Dict, NamedTuple


class OperatorStats(NamedTuple):
    """
    Container for empirical second-moment matrices defined in Section 2.3.1 of the paper.
    Notation matches:
        M_f: M_{rho_0}[f] (Eq. 2.3.1)
        M_g: M_{rho_1}[g] (Eq. 2.3.1)
        T_fg: T[f, g] (Eq. 2.3.1)
    """

    M_f: np.ndarray  # \hat{M}_{\rho_0}[f]
    M_g: np.ndarray  # \hat{M}_{\rho_1}[g]
    T_fg: np.ndarray  # \hat{T}[f, g]


class CCAComponents(NamedTuple):
    """
    Components from the CCA procedure described in Section 3.2.1.
    """

    align_f: np.ndarray  # Alignment matrix for f (approx. Sigma^{1/2} U^T M_f^{-1/2})
    align_g: np.ndarray  # Alignment matrix for g (approx. Sigma^{1/2} V^T M_g^{-1/2})
    sigma: np.ndarray  # Singular values (Sigma)
    U: np.ndarray  # Left singular vectors of whitened T
    V: np.ndarray  # Right singular vectors of whitened T
    M_f_inv_sqrt: np.ndarray  # (M_{\rho_0}[f])^{-1/2}
    M_g_inv_sqrt: np.ndarray  # (M_{\rho_1}[g])^{-1/2}


def compute_moments(f: np.ndarray, g: np.ndarray) -> OperatorStats:
    """
    Computes empirical second-moment matrices required for Koopman operator estimation.
    See Section 2.3.1.

    Args:
        f: Feature of current state x, f(x). Shape: (N, Rank)
        g: Feature of future state x', g(x'). Shape: (N, Rank)
    Returns:
        OperatorStats containing \hat{M}_f, \hat{M}_g, and \hat{T}_{fg}.
    """
    N = f.shape[0]

    # Compute empirical second moments (Eq. 2.3.1)
    # \hat{M}_{\rho_0}[f] = 1/N \sum f(x)f(x)^T
    M_f = (f.T @ f) / N

    # \hat{M}_{\rho_1}[g] = 1/N \sum g(x')g(x')^T
    M_g = (g.T @ g) / N

    # \hat{T}[f, g] = 1/N \sum f(x)g(x')^T
    T_fg = (f.T @ g) / N

    return OperatorStats(M_f, M_g, T_fg)


def perform_cca(stats: OperatorStats, epsilon: float = 1e-6) -> CCAComponents:
    """
    Performs Canonical Correlation Analysis (CCA) as described in 'Approach 1: CCA + LoRA' (Section 3.2.1).
    This procedure whitens the basis and corrects them via SVD of the joint moment matrix.

    See Eq. (6) and surrounding text.
    """

    def _inv_sqrt(M):
        # Use eigh for stability with symmetric matrices (M is positive semi-definite)
        w, v = np.linalg.eigh(M)
        # Clip small eigenvalues for numerical stability
        w = np.maximum(w, epsilon)
        inv_sqrt = v @ np.diag(w**-0.5) @ v.T
        return inv_sqrt

    # 1. Compute Inverse Square Roots for Whitening
    # (M_{\rho_0}[f])^{-1/2} and (M_{\rho_1}[g])^{-1/2}
    M_f_inv_sqrt = _inv_sqrt(stats.M_f)
    M_g_inv_sqrt = _inv_sqrt(stats.M_g)

    # 2. Form the whitened joint second moment matrix (Section 3.2.1)
    # T[\tilde{f}, \tilde{g}] = (M_f)^{-1/2} T[f, g] (M_g)^{-1/2}
    T_tilde = M_f_inv_sqrt @ stats.T_fg @ M_g_inv_sqrt

    # 3. Perform SVD on the whitened matrix
    # T[\tilde{f}, \tilde{g}] = U \Sigma V^T
    U, sigma, Vh = svd(T_tilde)
    V = Vh.T  # numpy returns V^H, we denote V as columns

    # 4. Compute Alignment Matrices (Eq. 6)
    # The aligned singular functions are \tilde{\phi}(x) = \Sigma^{1/2} U^T \tilde{f}(x)
    # So the transformation matrix A_f applied to f(x) is: A_f = \Sigma^{1/2} U^T (M_f)^{-1/2}
    sigma_sqrt = np.sqrt(sigma)[:, None]

    align_f = (sigma_sqrt * U.T) @ M_f_inv_sqrt
    align_g = (sigma_sqrt * V.T) @ M_g_inv_sqrt

    return CCAComponents(
        align_f=align_f,
        align_g=align_g,
        sigma=sigma,
        U=U,
        V=V,
        M_f_inv_sqrt=M_f_inv_sqrt,
        M_g_inv_sqrt=M_g_inv_sqrt,
    )


def compute_edmd_operator(
    M_b: np.ndarray, T_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the finite-dimensional approximation of the Koopman Operator via EDMD (Section 3.2.2).
    K^{ols}_b = (M_{\rho_0}[b])^+ T[b] (Eq. 9).

    Args:
        M_b: Second moment of basis b, M_{\rho_0}[b] (or M_f / M_g)
        T_b: Joint moment T[b] (or T[f, f] / T[f, g] depending on usage)

    Returns:
        K: The approximate Koopman matrix
        U, Q: Real Schur decomposition components (K = Q U Q^T) for stable power iteration.
    """
    # K = (M_b)^+ @ T_b
    M_b_inv = pinv(M_b)
    K = M_b_inv @ T_b

    # Real Schur Decomposition for stability (K = Q @ U @ Q.T)
    U, Q = schur(K, output="real")
    return K, U, Q


def compute_vamp_scores(
    M00: np.ndarray, M01: np.ndarray, M11: np.ndarray, cca: CCAComponents
) -> Dict[str, float]:
    """
    Computes VAMP-2 and VAMP-E scores using test moments and training CCA components.

    Args:
        M00: Test moment M_{\rho_0}[f]
        M01: Test moment T[f, g]
        M11: Test moment M_{\rho_1}[g]
        cca: CCA components learned from training data
    """
    # 1. Project Test Moments to Whitened Training Space
    # \tilde{M} = (M_f^{train})^{-1/2} M^{test} (M_g^{train})^{-1/2}
    M00_w = cca.M_f_inv_sqrt @ M00 @ cca.M_f_inv_sqrt
    M01_w = cca.M_f_inv_sqrt @ M01 @ cca.M_g_inv_sqrt
    M11_w = cca.M_g_inv_sqrt @ M11 @ cca.M_g_inv_sqrt

    # 2. Project onto top-k singular vectors from training
    # C_cca corresponds to projecting moments onto the subspace spanned by U and V
    # C00_cca = U^T \tilde{M}_{00} U
    C00_cca = cca.U.T @ M00_w @ cca.U
    C01_cca = cca.U.T @ M01_w @ cca.V
    C11_cca = cca.V.T @ M11_w @ cca.V

    # --- VAMP-2 Calculation ---
    # Score = || C00^{-1/2} C01 C11^{-1/2} ||_F^2
    def _safe_inv_sqrt(M):
        w, v = np.linalg.eigh(M)
        w = np.maximum(w, 1e-10)
        return v @ np.diag(w**-0.5) @ v.T

    inv_sqrt_00 = _safe_inv_sqrt(C00_cca)
    inv_sqrt_11 = _safe_inv_sqrt(C11_cca)

    M_vamp = inv_sqrt_00 @ C01_cca @ inv_sqrt_11
    vamp2 = np.sum(np.linalg.svd(M_vamp, compute_uv=False) ** 2)

    # --- VAMP-E Calculation ---
    # Related to ||K - \hat{K}||_{HS}^2 (Appendix B.1)
    # S contains the singular values (Sigma) learned during training
    S = np.diag(cca.sigma)

    term1 = 2 * np.trace(S @ C01_cca)
    term2 = np.trace(S @ C00_cca @ S @ C11_cca)

    vampe = term1 - term2

    return {"vamp2": float(vamp2), "vampe": float(vampe)}


def predict_linear_map(
    x0: np.ndarray, K: np.ndarray, power: int, projection: np.ndarray
) -> np.ndarray:
    """
    Predicts x_t via the linear approximation:
    E[g(x_t)] \approx K^t E[g(x_0)]

    Used for Eq. (7) or Eq. (9) predictions.
    """
    if power == 0:
        return x0

    K_pow = np.linalg.matrix_power(K, power)
    return x0 @ K_pow @ projection
