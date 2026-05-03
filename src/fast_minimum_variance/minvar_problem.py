"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from dataclasses import dataclass

import numpy as np

from ._base import _BaseProblem


@dataclass(frozen=True)
class _MinVarProblem(_BaseProblem):
    """Minimum-variance portfolio solver via primal-dual active-set iteration.

    Solves::

        min  (1-alpha)||X w||^2 + alpha*(||X||_F^2/N)*||w||^2 - rho*mu^T w
        s.t. 1^T w = 1,  w >= 0

    Each inner step solves the equality-constrained subproblem over the current
    active asset set.  Stationarity gives ``2*Sigma_a*w_a = lambda*1 + rho*mu_a``
    where ``Sigma_a = (1-alpha)*X_a^T X_a + ridge*I``.  Solving the ``n_a x n_a``
    SPD system ``Sigma_a v = 1`` (and ``Sigma_a v2 = mu_a`` when ``rho != 0``)
    and recovering ``lambda`` from the budget constraint avoids the indefinite
    ``(n_a+1) x (n_a+1)`` saddle-point system entirely.  The outer primal-dual
    loop enforces ``w >= 0`` and terminates when both primal and dual feasibility
    hold simultaneously.

    Use ``alpha = N/(N+T)`` for Ledoit-Wolf shrinkage intensity::

        T, N = X.shape
        w, iters = Problem(X, alpha=N/(N+T)).solve_kkt()

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance import Problem
        >>> X = np.random.default_rng(0).standard_normal((100, 5))
        >>> w, iters = Problem(X).solve_kkt()
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
    """

    # No extra fields — X, alpha, rho, mu all inherited from _BaseProblem.

    # ------------------------------------------------------------------
    # Outer loop: primal elimination + dual feasibility check
    # ------------------------------------------------------------------
    def _constraint_active_set(self, solve_fn, tol=1e-6, max_iter=10_000):

        n = self.n
        asset_active = np.ones(n, dtype=bool)
        total_iters = 0

        ridge = self._ridge()
        oma = 1.0 - self.alpha

        prev_active = None

        for _ in range(max_iter):
            if prev_active is not None and np.array_equal(prev_active, asset_active):
                break  # pragma: no cover - structurally unreachable safety guard
            prev_active = asset_active.copy()

            # === Solve ===
            w_a, step_iters = solve_fn(asset_active)
            total_iters += step_iters

            # === PRIMAL STEP ===
            neg = w_a < -tol
            if np.any(neg):
                idx = np.where(asset_active)[0]

                strong = w_a < -10 * tol

                if np.any(strong):
                    asset_active[idx[strong]] = False
                else:
                    j = idx[np.argmin(w_a)]
                    asset_active[j] = False

                continue  # CRITICAL

            # === Assemble full vector ===
            w = np.zeros(n)
            w[asset_active] = w_a

            # === Gradient ===
            grad = 2.0 * (oma * (self.X.T @ (self.X @ w)) + ridge * w)

            if self.rho != 0.0 and self.mu is not None:
                grad = grad - self.rho * self.mu

            # === Lambda ===
            g_a = grad[asset_active]
            lambda_ = np.median(g_a) if g_a.size > 5 else g_a.mean()

            # === Dual ===
            nu = grad - lambda_

            excluded = ~asset_active
            if not excluded.any():
                break

            nu_ex = nu[excluded]
            idx_ex = np.where(excluded)[0]

            j = idx_ex[np.argmin(nu_ex)]
            violate = nu[j]

            # === DUAL STEP ===
            if violate >= -tol:
                break

            asset_active[j] = True

        return w, total_iters

    # ------------------------------------------------------------------
    # Inner steps
    # ------------------------------------------------------------------

    def _kkt_step(self, active):
        """Solve the reduced SPD system directly; return ``(w_a, 1)``.

        Stationarity gives ``2*Sigma_a*w_a = lambda*1 + rho*mu_a``.  A single
        solve with two RHS columns yields ``v1 = Sigma_a^{-1} 1`` and
        ``v2 = Sigma_a^{-1} mu_a``; the budget constraint then pins ``lambda``
        analytically as ``lambda = 2*(1 - rho/2 * sum(v2)) / sum(v1)``.
        """
        x_a = self.X[:, active]
        n_a = int(active.sum())
        sigma = (1.0 - self.alpha) * (x_a.T @ x_a) + self._ridge() * np.eye(n_a)

        if self.rho == 0.0 or self.mu is None:
            v = np.linalg.solve(sigma, np.ones(n_a))
            return v / v.sum(), 1

        v1, v2 = np.linalg.solve(sigma, np.column_stack([np.ones(n_a), self.mu[active]])).T
        half_rho = 0.5 * self.rho
        half_lambda = (1.0 - half_rho * v2.sum()) / v1.sum()
        return half_lambda * v1 + half_rho * v2, 1

    def _cvxpy_constraints(self, w, cp):
        """Return budget-equality and long-only inequality constraints for CVXPY."""
        return [cp.sum(w) == 1, w >= 0]
