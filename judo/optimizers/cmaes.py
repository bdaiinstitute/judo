# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

from judo.optimizers.base import Optimizer, OptimizerConfig


@dataclass
class CMAESConfig(OptimizerConfig):
    """CMA-ES with noise ramping and minimum variance safeguards."""

    sigma0: float = 0.25  # initial global step-size
    min_std: float = 0.01  # minimum per-dimension std for sampling (post-ramp, post-cov)
    max_std: float = 1.0


class CMAES(Optimizer[CMAESConfig]):
    """The CMA-ES optimizer."""

    def __init__(self, config: CMAESConfig, nu: int) -> None:
        """Initialize CMA-ES optimizer.

        This implementation follows the algorithm described in
        "A Tutorial on the Covariance Matrix Adaptation Evolution Strategy"
        by Nikolaus Hansen (2016), with some additional features:
        - Noise ramping: scales exploration noise linearly over the horizon.
        - Minimum std: ensures a minimum exploration std per dimension, to avoid premature convergence.
        - Maximum std: ensures a maximum exploration std per dimension, to avoid excessive exploration.
        """
        super().__init__(config, nu)

        # total dimension of search space
        self.dim = self.num_nodes * self.nu

        # state
        self.sigma = self.config.sigma0
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.A = self.B * self.D[None, :]  # sqrt(C) columns
        self.invsqrtC = self.B * (1.0 / self.D)[None, :] @ self.B.T

        self.pc = np.zeros(self.dim)  # cov path
        self.ps = np.zeros(self.dim)  # step-size path
        self.iteration = 0
        self._last_num_nodes = self.num_nodes

        # params that depend on lambda (num_rollouts)
        self._set_lambda_dependent_params(self.num_rollouts)

    def _set_lambda_dependent_params(self, lam: int) -> None:
        """(Re)compute parameters that depend on lambda (num_rollouts)."""
        lam = int(max(1, lam))
        self.lam = lam
        mu = lam // 2
        self.mu = int(np.clip(int(max(1, mu)), 1, lam))

        # rank-based positive recombination weights
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)

        # learning rates that depend on mueff (but not on lambda directly otherwise)
        d = self.dim
        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2 + 1.0 / self.mueff) / ((d + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs
        self.chiN = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

    def _ramp_flat(self) -> np.ndarray:
        """Per-dimension ramp vector S, flattened (length = dim)."""
        if not getattr(self, "use_noise_ramp", False):
            return np.ones(self.dim)
        nr = float(getattr(self, "noise_ramp", 1.0))
        node_scale = np.linspace(nr / self.num_nodes, nr, self.num_nodes, endpoint=True)[:, None]
        S = np.repeat(node_scale, self.nu, axis=1).reshape(-1)
        return np.maximum(S, 1e-12)

    def _diag_C(self) -> np.ndarray:  # noqa: N802
        """Diagonal of current covariance C in coordinate basis, via A row norms."""
        return np.sum(self.A**2, axis=1)  # var_j = ||row_j(A)||^2

    def _recompute_eigensystem(self) -> None:
        """Update eigensystem from C with small stabilizers."""
        C = 0.5 * (self.C + self.C.T) + 1e-6 * np.eye(self.dim)
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 1e-6, None)  # eigenvalue floor
        self.D = np.sqrt(evals)
        self.B = evecs
        self.C = (self.B * (self.D**2)[None, :]) @ self.B.T  # keep C consistent with floored evals
        self.A = self.B * self.D[None, :]
        self.invsqrtC = self.B * (1.0 / self.D)[None, :] @ self.B.T

    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """Reinitialize when the horizon (num_nodes) changes."""
        if self.num_nodes != self._last_num_nodes:
            self.dim = self.num_nodes * self.nu
            self.sigma = self.config.sigma0
            self.C = np.eye(self.dim)
            self.B = np.eye(self.dim)
            self.D = np.ones(self.dim)
            self.A = self.B * self.D[None, :]
            self.invsqrtC = self.B * (1.0 / self.D)[None, :] @ self.B.T
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            self.iteration = 0
            self._last_num_nodes = self.num_nodes
            self._set_lambda_dependent_params(self.num_rollouts)

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample lambda-1 candidates + include nominal as first; apply ramp and min-std top-up."""
        # if user changed num_rollouts since last call, adapt lambda-params lazily here.
        if self.num_rollouts != getattr(self, "lam", None):
            self._set_lambda_dependent_params(self.num_rollouts)

        m = nominal_knots.reshape(-1)  # (dim,)
        lam = self.lam
        dim = self.dim

        # draw base correlated steps
        Z = np.random.randn(lam - 1, dim)  # (lam-1, dim)
        S = self._ramp_flat()  # (dim,)
        Y = (Z @ self.A.T) * S[None, :]  # ramp-scaled correlated steps

        # minimum and maximum std enforcement
        # current effective std per coordinate: base_std = sigma * S * sqrt(diag(C))
        diagC = self._diag_C()  # (dim,)
        base_std = self.sigma * S * np.sqrt(diagC)  # (dim,)

        # ramp-scaled + clamped correlated steps
        gamma = np.minimum(1.0, (self.config.max_std + 1e-12) / (base_std + 1e-12))
        Y = (Z @ self.A.T) * (S * gamma)[None, :]

        # effective base std after clamp
        base_std_eff = base_std * gamma
        extra = np.sqrt(np.maximum(0.0, self.config.min_std**2 - base_std_eff**2))

        if np.any(extra > 0):
            Z2 = np.random.randn(lam - 1, dim)
            # add independent diagonal noise so that final std in X is >= min_std
            Y += Z2 * (extra[None, :] / max(self.sigma, 1e-20))

        # compose samples
        X = m[None, :] + self.sigma * Y
        X_all = np.vstack([m[None, :], X])  # include nominal as first rollout
        return X_all.reshape(lam, self.num_nodes, self.nu)

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Rank-mu + rank-one CMA-ES update."""
        # be robust if lambda changed between sampling and update
        lam_obs = sampled_knots.shape[0]
        if lam_obs != getattr(self, "lam", None):
            self._set_lambda_dependent_params(lam_obs)

        X = sampled_knots.reshape(self.lam, -1)  # (lam, dim)
        dim = self.dim

        # sort by reward (desc)
        idx = np.flip(np.argsort(rewards))
        X_sorted = X[idx]
        X_mu = X_sorted[: self.mu]  # (mu, dim)
        w = self.weights[:, None]  # (mu, 1)

        m_old = X[0].copy()  # previous mean (first rollout)

        # observed normalized steps in coordinate space (include ramp + any top-up)
        Y_mu_obs = (X_mu - m_old[None, :]) / max(self.sigma, 1e-20)  # (mu, dim)

        # remove ramp for updates -> transform back to C coordinates
        S = self._ramp_flat()
        S_inv = 1.0 / S
        Y_mu_C = Y_mu_obs * S_inv[None, :]  # (mu, dim)

        # debias Y_mu_C to undo min-std and max-std clamping
        diagC = self._diag_C()
        base_std = self.sigma * S * np.sqrt(diagC)
        gamma = np.minimum(1.0, (self.config.max_std + 1e-12) / (base_std + 1e-12))
        base_std_eff = base_std * gamma  # post-clamp std actually used in sampling

        # scale to remove added diagonal noise when min_std bound
        corr = np.ones_like(base_std)
        mask_min = base_std_eff < self.config.min_std
        corr[mask_min] = np.maximum(1e-12, base_std_eff[mask_min] / self.config.min_std)

        # only undo clamp where min_std did not bind
        mask_no_min = ~mask_min
        corr[mask_no_min] *= 1.0 / np.maximum(gamma[mask_no_min], 1e-12)
        Y_mu_C *= np.clip(corr, 1e-6, 1e6)[None, :]

        # weighted step in C coords
        y_w_C = (w * Y_mu_C).sum(axis=0)  # (dim,)

        # new mean (in original coords, keep ramped obs for construction)
        m_new = m_old + self.sigma * (w * Y_mu_obs).sum(axis=0)

        # step-size path
        Cinv_step = self.invsqrtC @ y_w_C
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinv_step

        # hsig gate
        self.iteration += 1
        one_minus = 1.0 - (1.0 - self.cs) ** (2 * self.iteration)
        norm_ps = np.linalg.norm(self.ps)
        hsig = 1.0 if (norm_ps / np.sqrt(one_minus) / self.chiN) < (1.4 + 2.0 / (dim + 1)) else 0.0

        # covariance path (C coords)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w_C

        # rank-mu covariance (C coords)
        YwC = np.sqrt(self.weights)[None, :] * Y_mu_C.T  # (dim, mu)
        rank_mu = YwC @ YwC.T

        # covariance update with rank-1 + rank-mu
        alpha = (1 - self.c1 - self.cmu) + (1 - hsig) * self.c1 * self.cc * (2 - self.cc)
        self.C = alpha * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu

        # step-size update
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1.0))

        # refresh eigensystem with eigenvalue floor
        self._recompute_eigensystem()

        return m_new.reshape(self.num_nodes, self.nu)
