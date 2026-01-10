import numpy as np
from typing import Optional, Dict
from koopmansvd.models.inference import linalg


class KoopmanEstimator:
    """
    Estimates the Koopman Operator spectral properties from learned features.

    This class implements the inference procedures described in Section 3.2 of the paper.
    It accumulates sufficient statistics (covariance matrices) from the entire dataset
    to perform stable spectral analysis.

    Supports two inference approaches:
    1. **CCA + LoRA (Approach 1)**: Performs Canonical Correlation Analysis on the
       learned features to retrieve whitening and alignment matrices.
       Recommended for singular value estimation (Table 1).
    2. **EDMD (Approach 2)**: Computes the Koopman matrix via Extended Dynamic
       Mode Decomposition using the learned basis functions.
       Recommended for propagating dynamics (Eq. 9).

    Attributes:
        stats (dict): Accumulated empirical second-moment matrices (M_f, M_g, T_fg).
        operators (dict): Computed Koopman matrices (K_fwd, K_bwd) for 'raw' and 'aligned' bases.

    Notations:
        - rho0: Distribution of current states (x_t)
        - rho1: Distribution of future states (x_{t+1})
        - f: Encoder (Left singular functions)
        - g: Lagged Encoder (Right singular functions)

    TODO: Consider refactoring operator finalization logic into separate
        builder classes (e.g., CCABuilder, EDMDBuilder) if more
        inference methods are added in the future.
    """

    # We recommend using float64 for numerical stability
    NUMPY_FLOAT_DTYPE: np.dtype = np.float64

    def __init__(self, rank: Optional[int] = None, use_cca: bool = True):
        self.rank = rank
        self.use_cca = use_cca
        self._is_fitted = False
        self._train_n = 0

        # --- Training Accumulators (Full Batch Stats) ---
        self._train_n = 0
        self._train_accum = {
            # Self-Covariances
            "M_f_rho0": None,  # E[f_t f_t^T]
            "M_g_rho1": None,  # E[g_next g_next^T]
            "M_f_rho1": None,  # E[f_next f_next^T] (Needed for f-based backward denominator)
            "M_g_rho0": None,  # E[g_t g_t^T] (Needed for g-based backward denominator)
            # Cross-Covariances for CCA
            "T_fg": None,  # E[f_t g_next^T]
            # Full Dynamics for EDMD (Both f and g, both directions) Check Appendix E.
            "T_ff": None,  # E[f_t f_next^T] (EDMD(f) Fwd)
            "T_ff_rev": None,  # E[f_next f_t^T] (EDMD(f) Bwd)
            "T_gg": None,  # E[g_t g_next^T] (EDMD(g) Fwd)
            "T_gg_rev": None,  # E[g_next g_t^T] (EDMD(g) Bwd)
            # Aligned Stats
            "M_gf_rho1": None,  # E[g_next f_next^T]
            "M_gf_rho0": None,  # E[g_t f_t^T]
            # Projections
            "T_fx": None,  # E[f_t x_t^T]
            "T_gy": None,  # E[g_next x_next^T]
            "T_gx": None,  # E[g_t x_t^T]
        }

        # --- Evaluation Accumulators (Test VAMP Score) ---
        self._eval_n = 0
        self._eval_accum = {"M_f_rho0": None, "M_g_rho1": None, "T_fg": None}

        # --- Learned Model States ---
        self.stats = {}
        self.operators = {
            "raw": {"f": {"fwd": None, "bwd": None}, "g": {"fwd": None, "bwd": None}},
            "ali": {"fwd": None, "bwd": None},
        }
        self.alignments = {}
        self.projections = {}
        self._cca_components = {}

    def _init_accum(self, _accum, dim_f, dim_g, dim_x, dim_y):
        _dtype = self.NUMPY_FLOAT_DTYPE
        for key in _accum:
            if key is None:
                continue
            if "fx" in key or "gx" in key:
                if dim_x > 0:
                    _accum[key] = np.zeros(
                        (dim_f if "f" in key else dim_g, dim_x), dtype=_dtype
                    )
            elif "gy" in key:
                if dim_y > 0:
                    _accum[key] = np.zeros((dim_g, dim_y), dtype=_dtype)
            elif "gf" in key:
                _accum[key] = np.zeros((dim_g, dim_f), dtype=_dtype)
            elif "fg" in key:
                _accum[key] = np.zeros((dim_f, dim_g), dtype=_dtype)
            elif "ff" in key or "_f_" in key:
                _accum[key] = np.zeros((dim_f, dim_f), dtype=_dtype)
            elif "gg" in key or "_g_" in key:
                _accum[key] = np.zeros((dim_g, dim_g), dtype=_dtype)

    def partial_fit(self, f_t, g_t, f_next, g_next, x_raw=None, y_raw=None):
        """
        Accumulates moments batch-wise for TRAINING.
        f_t: Output of encoder f at time t. Shape (B, dim_f)
        g_t: Output of lagged encoder g at time t. Shape (B, dim_g)
        f_next: Output of encoder f at time t+1. Shape (B, dim_f)
        g_next: Output of lagged encoder g at time t+1. Shape (B, dim_g)
        x_raw: Raw input data at time t. Shape (B, dim_x)
        y_raw: Raw input data at time t+1. Shape (B, dim_y)
        """
        if self._train_n == 0:
            dim_x = x_raw.shape[1] if x_raw is not None else 0
            dim_y = y_raw.shape[1] if y_raw is not None else 0
            self._init_accum(
                self._train_accum, f_t.shape[1], g_next.shape[1], dim_x, dim_y
            )
        _dtype = self.NUMPY_FLOAT_DTYPE
        f_t = f_t.astype(_dtype)
        g_t = g_t.astype(_dtype)
        f_next = f_next.astype(_dtype)
        g_next = g_next.astype(_dtype)
        if x_raw is not None:
            x_raw = x_raw.astype(_dtype)
        if y_raw is not None:
            y_raw = y_raw.astype(_dtype)

        # 1. Self Covariances
        self._train_accum["M_f_rho0"] += f_t.T @ f_t
        self._train_accum["M_g_rho1"] += g_next.T @ g_next
        self._train_accum["M_g_rho0"] += g_t.T @ g_t
        self._train_accum["M_f_rho1"] += f_next.T @ f_next

        # 2. Dynamics Correlations
        self._train_accum["T_fg"] += f_t.T @ g_next  # CCA
        self._train_accum["T_ff"] += f_t.T @ f_next  # EDMD-f Fwd
        self._train_accum["T_ff_rev"] += f_next.T @ f_t  # EDMD-f Bwd
        self._train_accum["T_gg"] += g_t.T @ g_next  # EDMD-g Fwd
        self._train_accum["T_gg_rev"] += g_next.T @ g_t  # EDMD-g Bwd

        # 3. Aligned Cross-Correlations
        self._train_accum["M_gf_rho1"] += g_next.T @ f_next
        self._train_accum["M_gf_rho0"] += g_t.T @ f_t

        # 4. Projections
        if x_raw is not None:
            self._train_accum["T_fx"] += f_t.T @ x_raw
            self._train_accum["T_gx"] += g_t.T @ x_raw
        if y_raw is not None:
            self._train_accum["T_gy"] += g_next.T @ y_raw

        self._train_n += f_t.shape[0]
        return self

    def finalize(self):
        """Computes Operators using accumulated training statistics."""
        if self._train_n == 0:
            raise RuntimeError("No training data.")
        N = self._train_n

        # Normalize
        S = {k: v / N for k, v in self._train_accum.items() if v is not None}
        self.stats.update(S)
        self.stats["raw"] = linalg.OperatorStats(
            M_f=S["M_f_rho0"], M_g=S["M_g_rho1"], T_fg=S["T_fg"]
        )

        # =========================================================
        # Approach 1: CCA + LoRA (Aligned)
        # =========================================================
        if self.use_cca:
            cca = linalg.perform_cca(self.stats["raw"])
            self.alignments["f"] = cca.align_f
            self.alignments["g"] = cca.align_g
            self._cca_components = {
                "sigma": cca.sigma,
                "U": cca.U,
                "V": cca.V,
                "M_f_inv_sqrt": cca.M_f_inv_sqrt,
                "M_g_inv_sqrt": cca.M_g_inv_sqrt,
            }

            # K_right (Fwd): E[psi(y) phi(y)^T] = A_g E[g(y) f(y)^T] A_f^T
            K_ali_fwd = cca.align_g @ S["M_gf_rho1"] @ cca.align_f.T

            # K_left (Bwd): E[psi(x) phi(x)^T] = A_g E[g(x) f(x)^T] A_f^T
            K_ali_bwd = cca.align_g @ S["M_gf_rho0"] @ cca.align_f.T

            self.operators["ali"]["fwd"] = {"K": K_ali_fwd}
            self.operators["ali"]["bwd"] = {"K": K_ali_bwd}

        # =========================================================
        # Approach 2: EDMD (Raw / Legacy)
        # =========================================================

        # --- Basis f ---
        # Fwd: M_{f,rho0}^+ T_{ff}
        K_ff, U_ff, Q_ff = linalg.compute_edmd_operator(S["M_f_rho0"], S["T_ff"])
        # Bwd: M_{f,rho1}^+ T_{ff,rev}
        K_fb, U_fb, Q_fb = linalg.compute_edmd_operator(S["M_f_rho1"], S["T_ff_rev"])

        self.operators["raw"]["f"] = {
            "fwd": {"K": K_ff, "U": U_ff, "Q": Q_ff},
            "bwd": {"K": K_fb, "U": U_fb, "Q": Q_fb},
        }

        # --- Basis g ---
        # Fwd: M_{g,rho0}^+ T_{gg}
        K_gf, U_gf, Q_gf = linalg.compute_edmd_operator(S["M_g_rho0"], S["T_gg"])
        # Bwd: M_{g,rho1}^+ T_{gg,rev}
        K_gb, U_gb, Q_gb = linalg.compute_edmd_operator(S["M_g_rho1"], S["T_gg_rev"])

        self.operators["raw"]["g"] = {
            "fwd": {"K": K_gf, "U": U_gf, "Q": Q_gf},
            "bwd": {"K": K_gb, "U": U_gb, "Q": Q_gb},
        }

        # =========================================================
        # Projections (Decoders)
        # =========================================================
        if S.get("T_fx") is not None and S.get("T_gx") is not None:
            # P_f: f(x) -> x.
            P_f = np.linalg.pinv(S["M_f_rho0"], rcond=1e-7) @ S["T_fx"]

            # P_g: g(x) -> x.
            # Use M_g_rho0 (covariance of g at current time)
            P_g = np.linalg.pinv(S["M_g_rho0"], rcond=1e-7) @ S["T_gx"]

            self.projections = {"f": P_f, "g": P_g}

        self._is_fitted = True
        return self

    # --- Evaluation Methods ---
    def reset_eval(self):
        """Resets evaluation accumulators."""
        self._eval_n = 0
        self._eval_accum = {"M_f_rho0": None, "M_g_rho1": None, "T_fg": None}

    def partial_evaluate(self, f_t: np.ndarray, g_next: np.ndarray):
        """Accumulates test statistics for VAMP score."""
        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before evaluation.")

        if self._eval_n == 0:
            self._init_accum(self._eval_accum, f_t.shape[1], g_next.shape[1], 0, 0)

        _dtype = self.NUMPY_FLOAT_DTYPE
        f_t = f_t.astype(_dtype)
        g_next = g_next.astype(_dtype)

        self._eval_accum["M_f_rho0"] += f_t.T @ f_t
        self._eval_accum["M_g_rho1"] += g_next.T @ g_next
        self._eval_accum["T_fg"] += f_t.T @ g_next

        self._eval_n += f_t.shape[0]
        return self

    def evaluate(self) -> Dict[str, float]:
        """Computes VAMP-2 and VAMP-E scores."""
        if not self._is_fitted or self._eval_n == 0:
            return {"vamp2": 0.0, "vampe": 0.0}
        if not self.use_cca:
            return {"vamp2": 0.0, "vampe": 0.0}

        _dtype = self.NUMPY_FLOAT_DTYPE
        N = self._eval_n
        # Use accumulated test stats
        cca = linalg.CCAComponents(
            align_f=self.alignments["f"],
            align_g=self.alignments["g"],
            sigma=self._cca_components["sigma"],
            U=self._cca_components["U"],
            V=self._cca_components["V"],
            M_f_inv_sqrt=self._cca_components["M_f_inv_sqrt"],
            M_g_inv_sqrt=self._cca_components["M_g_inv_sqrt"],
        )

        # Calculate scores
        scores = linalg.compute_vamp_scores(
            self._eval_accum["M_f_rho0"].astype(_dtype) / N,
            self._eval_accum["T_fg"].astype(_dtype) / N,
            self._eval_accum["M_g_rho1"].astype(_dtype) / N,
            cca,
        )
        self.reset_eval()
        return scores

    def eig(self):
        """Returns eigenvalues of the learned operator."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted")

        # Use Aligned Operator if available
        if self.use_cca and self.operators.get("ali", {}).get("fwd") is not None:
            K = self.operators["ali"]["fwd"]["K"]
        else:
            # Fallback to Raw-f
            K = self.operators["raw"]["f"]["fwd"]["K"]

        eigs = np.linalg.eigvals(K)
        sort_idx = np.argsort(np.abs(eigs))[::-1]
        return eigs[sort_idx]

    # --- Prediction ---

    def predict(
        self,
        f_x0: np.ndarray,
        g_x0: np.ndarray,
        t: int = 1,
        mode: str = "aligned",
        basis: str = "f",
    ) -> np.ndarray:
        """
        Multi-step prediction.

        Args:
            f_x0: Output of the Left Encoder f(x_0). Shape (B, Rank)
            g_x0: Output of the Right (Lagged) Encoder g(x_0). Shape (B, Rank)
            t: Time step. (+: Forward, -: Backward)
            mode: 'aligned' or 'raw'
        """

        if t == 0:
            raise ValueError("t=0 not supported.")
        is_forward = t > 0
        power = abs(t)

        # ---------------------------------------------------------
        # Approach 2: EDMD (Raw / Legacy)
        # --------------------------------------------- dd------------
        if mode == "raw":
            if basis not in ["f", "g"]:
                raise ValueError("Basis must be 'f' or 'g'")
            b_x0 = f_x0 if basis == "f" else g_x0

            direction = "fwd" if is_forward else "bwd"
            ops = self.operators["raw"][basis][direction]

            # Stable Power Iteration: K^t = Q U^t Q^T
            U_pow = np.linalg.matrix_power(ops["U"], power)
            K_pow = ops["Q"] @ U_pow @ ops["Q"].T

            # Projection P (Maps basis -> image)
            P = self.projections[basis]

            return b_x0 @ K_pow @ P

        # ---------------------------------------------------------
        # Approach 1: CCA + LoRA (Aligned)
        # ---------------------------------------------------------
        elif mode == "aligned":
            if not self.use_cca:
                raise ValueError("CCA not fitted.")

            if is_forward:
                # Align: f(x) -> phi(x) = f @ A_f.T
                phi_x0 = f_x0 @ self.alignments["f"].T
                feat_ali = phi_x0
                # K = M_rho1[psi, phi].
                K = self.operators["ali"]["fwd"]["K"]
                # <coord, psi>_rho1 = A_g @ T_gy
                P_dec = self.alignments["g"] @ self.stats["T_gy"]

            else:
                # Align: g(x) -> psi(x) = g @ A_g.T
                psi_x0 = g_x0 @ self.alignments["g"].T
                feat_ali = psi_x0
                # K_left = M_rho0[psi, phi].
                K = self.operators["ali"]["bwd"]["K"].T
                # <coord, phi>_rho0 = A_f @ T_fx
                P_dec = self.alignments["f"] @ self.stats["T_fx"]

            # Power Iteration
            if power == 1:
                K_pow = np.eye(K.shape[0])
            else:
                K_pow = np.linalg.matrix_power(K, power - 1)

            return feat_ali @ K_pow @ P_dec

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def save(self, path):
        save_dict = {
            "operators": self.operators,
            "stats": self.stats,
            "projections": self.projections,
            "alignments": self.alignments,
            "cca_components": self._cca_components,
        }
        np.savez(path, **save_dict)
