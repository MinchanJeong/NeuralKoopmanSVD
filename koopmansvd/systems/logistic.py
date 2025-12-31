# koopmansvd/datasets/logistic.py
import numpy as np
import scipy.integrate
import scipy.special
from scipy.stats.sampling import NumericalInversePolynomial
from typing import Optional

from koopmansvd.systems.base import DiscreteTimeDynamics


class CosineDistribution:
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)

    def pdf(self, x):
        return self.C_N * ((np.cos(np.pi * x)) ** self.N)


class LogisticMap(DiscreteTimeDynamics):
    """
    Noisy Logistic Map dynamics.
    x_{t+1} = 4 * x_t * (1 - x_t) + xi_t (mod 1)
    """

    def __init__(self, r: float = 4.0, N: int = 20, rng_seed: Optional[int] = None):
        self.r = r
        self.N = N
        self.rng = np.random.default_rng(rng_seed)

        # Noise generator setup
        if N % 2 != 0:
            raise ValueError("N must be even for Cosine noise.")

        self._noise_rng = NumericalInversePolynomial(
            CosineDistribution(N),
            domain=(-0.5, 0.5),
            mode=0,
            random_state=self.rng,
        )

        # Precompute Ground Truth Eigenvalues (Transfer Operator)
        self._compute_spectrum()

    def _step(self, x: np.ndarray) -> np.ndarray:
        # Logistic Map Step
        y = self.r * x * (1 - x)
        noise = self._noise_rng.rvs(x.shape)
        return np.mod(y + noise, 1.0)

    def _compute_spectrum(self):
        """Computes analytic eigenvalues of the transfer operator."""
        # Matrix elements of the operator in the noise feature basis
        K = np.array(
            [
                [self._koopman_el(i, j) for j in range(self.N + 1)]
                for i in range(self.N + 1)
            ]
        )
        self._eigvals = scipy.linalg.eigvals(K)
        # Sort by magnitude
        idx = np.argsort(np.abs(self._eigvals))[::-1]
        self._eigvals = self._eigvals[idx]

    def _noise_feature(self, x, i):
        """Basis function beta_i(x)"""
        normalization_cst = np.pi / scipy.special.beta(self.N // 2 + 0.5, 0.5)
        return (
            ((np.sin(np.pi * x)) ** (self.N - i))
            * ((np.cos(np.pi * x)) ** i)
            * np.sqrt(scipy.special.binom(self.N, i) * normalization_cst)
        )

    def _noise_feature_composed(self, x, i):
        """alpha_i(x) = beta_i(T(x))"""
        y = self.r * x * (1 - x)
        return self._noise_feature(y, i)

    def _koopman_el(self, i, j):
        """Integral for matrix element <beta_i, alpha_j>"""

        def func(x):
            return self._noise_feature(x, i) * self._noise_feature_composed(x, j)

        return scipy.integrate.quad(func, 0, 1)[0]

    def eig(self):
        """Returns the ground truth eigenvalues."""
        return self._eigvals
