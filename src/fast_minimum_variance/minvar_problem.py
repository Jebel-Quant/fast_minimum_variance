"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from dataclasses import dataclass

import clarabel
import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, eye, vstack
from scipy.sparse.linalg import LinearOperator, cg

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
        """Run the primal-dual active-set loop enforcing ``w >= 0``.

        Calls ``solve_fn(active_mask)`` repeatedly.  The *primal step* drops assets
        with negative weights; the *dual step* re-adds any excluded asset whose KKT
        gradient condition is violated.  Terminates when both conditions hold
        simultaneously, which together with stationarity is sufficient for global
        optimality.
        """
        n = self.n
        asset_active = np.ones(n, dtype=bool)
        total_iters = 0

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
            if self.target is None:
                grad = 2.0 * (self.X.T @ (self.X @ w)) / self.t
            else:
                grad = 2.0 * ((1 - self.alpha) * (self.X.T @ (self.X @ w)) / self.t + self.alpha * self.target @ w)

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
        if self.target is None:
            sigma = (x_a.T @ x_a) / self.t
        else:
            sigma = (1.0 - self.alpha) * (x_a.T @ x_a) / self.t + self.alpha * self.target[np.ix_(active, active)]

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

    def _cg_step(self, active):
        """Solve the reduced SPD system via matrix-free CG; return ``(w_a, iters)``.

        Builds a ``LinearOperator`` for ``v -> (1-alpha)*X_a'*(X_a*v) + gamma*v``
        and runs conjugate gradients without ever forming ``Sigma_a`` explicitly.
        """
        x_a = self.X[:, active]
        n_a = int(active.sum())
        target_sub = self.target[np.ix_(active, active)] if self.target is not None else None

        def matvec(v):
            """Apply Sigma_LW matrix-free: v -> (1-alpha)/T * X_a'*(X_a*v) + alpha*target_a*v."""
            if target_sub is None:
                return (x_a.T @ (x_a @ v)) / self.t
            return (1.0 - self.alpha) * (x_a.T @ (x_a @ v)) / self.t + self.alpha * (target_sub @ v)

        op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)  # type: ignore[call-arg]

        iters = [0]

        def _count(_):
            """Increment CG iteration counter for the first solve."""
            iters[0] += 1

        if self.rho == 0.0 or self.mu is None:
            v, _ = cg(op, np.ones(n_a), callback=_count)
            return v / v.sum(), iters[0]

        iters2 = [0]

        def _count2(_):
            """Increment CG iteration counter for the second solve."""
            iters2[0] += 1

        v1, _ = cg(op, np.ones(n_a), callback=_count)
        v2, _ = cg(op, self.mu[active], callback=_count2)
        half_rho = 0.5 * self.rho
        half_lambda = (1.0 - half_rho * v2.sum()) / v1.sum()
        return half_lambda * v1 + half_rho * v2, iters[0] + iters2[0]

    def _clarabel_constraints(self):
        """Return budget-equality and long-only inequality constraints for Clarabel."""
        n = self.n
        a_mat = vstack(
            [csc_matrix(np.ones((1, n))), -eye(n, format="csc")],
            format="csc",
        )

        b_vec = np.concatenate([[1.0], np.zeros(n)])
        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n)]  # type: ignore[attr-defined]
        return a_mat, b_vec, cones

    def _osqp_constraints(self):
        """Return budget-equality and long-only inequality constraints for OSQP."""
        n = self.n
        a_mat = vstack(
            [csc_matrix(np.ones((1, n))), eye(n, format="csc")],
            format="csc",
        )
        l_vec = np.concatenate([[1.0], np.zeros(n)])
        u_vec = np.concatenate([[1.0], np.full(n, np.inf)])
        return a_mat, l_vec, u_vec

    def _nnls_solve(self):
        """Solve via NNLS on the augmented return matrix; return ``(w, 1)``.

        Builds ``A = [sqrt(1-alpha)*X ; sqrt(gamma)*I ; M*ones^T]`` and
        solves ``min ||Aw||² s.t. w >= 0``.  The budget row with weight
        ``M = ||X||_F * T`` enforces ``ones^T w ≈ 1``; exact normalisation
        is applied by the ``project`` step in ``solve_nnls``.
        Return tilt (``rho != 0``) is not supported.
        """
        t = self.X.shape[0]
        m = float(np.linalg.norm(self.X, "fro")) * t

        if self.target is not None:
            rows = [np.sqrt((1 - self.alpha) / self.t) * self.X]
            tgt = [np.zeros(t)]
            if self.alpha > 0.0:
                rows.append(np.sqrt(self.alpha) * self.target)
                tgt.append(np.zeros(self.n))
        else:
            rows = [np.sqrt(1.0 / self.t) * self.X]
            tgt = [np.zeros(t)]
        rows.append(m * np.ones((1, self.n)))
        tgt.append(np.array([m]))

        w, _ = nnls(np.vstack(rows), np.concatenate(tgt))
        return w, 1
