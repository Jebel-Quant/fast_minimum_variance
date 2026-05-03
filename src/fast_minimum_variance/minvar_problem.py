"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
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

    def _cg_step(self, active):
        """Solve the reduced SPD system via matrix-free CG; return ``(w_a, iters)``.

        Builds a ``LinearOperator`` for ``v -> (1-alpha)*X_a'*(X_a*v) + gamma*v``
        and runs conjugate gradients without ever forming ``Sigma_a`` explicitly.
        """
        x_a = self.X[:, active]
        n_a = int(active.sum())
        gamma = self._ridge()
        oma = 1.0 - self.alpha

        def matvec(v):
            """Apply Sigma_LW matrix-free: v -> (1-alpha)*X_a'*(X_a*v) + gamma*v."""
            return oma * (x_a.T @ (x_a @ v)) + gamma * v

        op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)

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

    def solve_clarabel(self, *, project: bool = True):
        """Solve via Clarabel interior-point solver (direct API, no CVXPY overhead).

        Assembles ``P = 2·Σ_LW`` as a sparse CSC matrix and calls Clarabel
        directly, bypassing CVXPY's problem-construction overhead.  Return
        ``(w, iters)`` where ``iters`` is the number of interior-point iterations.
        """
        try:
            import clarabel
            from scipy import sparse
        except ImportError as exc:
            msg = "clarabel and scipy are required for solve_clarabel"
            raise ImportError(msg) from exc

        n = self.n
        oma = 1.0 - self.alpha
        gamma = self._ridge()

        p_dense = 2.0 * (oma * (self.X.T @ self.X) + gamma * np.eye(n))
        p_csc = sparse.csc_matrix(p_dense)

        q = np.zeros(n)
        if self.rho != 0.0 and self.mu is not None:
            q = -self.rho * self.mu

        a_mat = sparse.vstack(
            [sparse.csc_matrix(np.ones((1, n))), -sparse.eye(n, format="csc")],
            format="csc",
        )
        b_vec = np.concatenate([[1.0], np.zeros(n)])
        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n)]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        sol = clarabel.DefaultSolver(p_csc, q, a_mat, b_vec, cones, settings).solve()

        w = np.array(sol.x)
        if project:
            w = self._clip_and_renormalize(w)
        return w, sol.iterations

    def _nnls_solve(self):
        """Solve via NNLS on the augmented return matrix; return ``(w, 1)``.

        Builds ``A = [sqrt(1-alpha)*X ; sqrt(gamma)*I ; M*ones^T]`` and
        solves ``min ||Aw||² s.t. w >= 0``.  The budget row with weight
        ``M = ||X||_F * T`` enforces ``ones^T w ≈ 1``; exact normalisation
        is applied by the ``project`` step in ``solve_nnls``.
        Return tilt (``rho != 0``) is not supported.
        """
        t = self.X.shape[0]
        oma = 1.0 - self.alpha
        gamma = self._ridge()
        m = float(np.linalg.norm(self.X, "fro")) * t

        rows = [np.sqrt(oma) * self.X]
        tgt = [np.zeros(t)]
        if gamma > 0.0:
            rows.append(np.sqrt(gamma) * np.eye(self.n))
            tgt.append(np.zeros(self.n))
        rows.append(m * np.ones((1, self.n)))
        tgt.append(np.array([m]))

        w, _ = nnls(np.vstack(rows), np.concatenate(tgt))
        return w, 1
