"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from dataclasses import dataclass

import clarabel
import numpy as np
from cvx.linalg import cholesky
from scipy.linalg import solve as spd_solve
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
    # Shared helpers used by both active-set loop variants
    # ------------------------------------------------------------------

    def _compute_gradient(self, w):
        """Return the full objective gradient at w, including rho*mu adjustment."""
        data_grad = (self.X.T @ (self.X @ w)) / self.t
        if self.target_lr is not None:
            bar_lam, U_k, delta_k = self.target_lr  # noqa: N806
            tgt_w = bar_lam * w + U_k @ (delta_k * (U_k.T @ w))
            grad = 2.0 * ((1 - self.alpha) * data_grad + self.alpha * tgt_w)
        elif self.target is not None:
            grad = 2.0 * ((1 - self.alpha) * data_grad + self.alpha * self.target @ w)
        else:
            grad = 2.0 * data_grad
        if self.rho != 0.0 and self.mu is not None:
            grad = grad - self.rho * self.mu
        return grad

    @staticmethod
    def _primal_drop(w_a, asset_active, tol):
        """Drop negative-weight assets from active set in-place; return True if any dropped."""
        if not np.any(w_a < -tol):
            return False
        idx = np.where(asset_active)[0]
        strong = w_a < -10 * tol
        if np.any(strong):
            asset_active[idx[strong]] = False
        else:
            asset_active[idx[np.argmin(w_a)]] = False
        return True

    def _dual_add(self, grad, asset_active, tol):
        """Return index of excluded asset that violates KKT dual condition, or -1 if none."""
        excluded = ~asset_active
        if not excluded.any():
            return -1
        g_a = grad[asset_active]
        lambda_ = np.median(g_a) if g_a.size > 5 else g_a.mean()
        nu = grad - lambda_
        idx_ex = np.where(excluded)[0]
        j = idx_ex[np.argmin(nu[excluded])]
        return int(j) if nu[j] < -tol else -1

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
        total_inner_iters = 0
        outer_steps = 0
        prev_active = None
        w = np.zeros(n)

        for _ in range(max_iter):
            if prev_active is not None and np.array_equal(prev_active, asset_active):
                break  # pragma: no cover - structurally unreachable safety guard
            prev_active = asset_active.copy()

            w_a, step_iters = solve_fn(asset_active)
            outer_steps += 1
            total_inner_iters += step_iters

            if self._primal_drop(w_a, asset_active, tol):
                continue

            w = np.zeros(n)
            w[asset_active] = w_a

            j = self._dual_add(self._compute_gradient(w), asset_active, tol)
            if j < 0:
                break
            asset_active[j] = True

        return w, outer_steps, total_inner_iters

    # ------------------------------------------------------------------
    # Inner steps
    # ------------------------------------------------------------------

    def _kkt_step(self, active, x0=None):
        """Solve the reduced SPD system directly; return ``(w_a, 1)``.

        Stationarity gives ``2*Sigma_a*w_a = lambda*1 + rho*mu_a``.  A single
        solve with two RHS columns yields ``v1 = Sigma_a^{-1} 1`` and
        ``v2 = Sigma_a^{-1} mu_a``; the budget constraint then pins ``lambda``
        analytically as ``lambda = 2*(1 - rho/2 * sum(v2)) / sum(v1)``.

        When ``alpha=1`` and ``target_lr`` is set the system is purely the RMT
        target ``T0 = bar_lam*I + U_k diag(delta_k) U_k^T``.  The Woodbury
        identity gives the exact inverse in O(n_a*k + k^3) without CG iterations:
        ``T0^{-1} b = b/bar_lam - U_k_a W^{-1}(U_k_a^T b)/bar_lam^2``
        where ``W = diag(1/delta_k) + U_k_a^T U_k_a / bar_lam``.
        """
        n_a = int(active.sum())

        # Woodbury direct solve: O(n_a*k + k^3) for alpha=1, RMT target
        if self.alpha == 1.0 and self.target_lr is not None:
            bar_lam, U_k, delta_k = self.target_lr  # noqa: N806
            U_k_a = U_k[active, :]  # noqa: N806  # (n_a, k)
            W = np.diag(1.0 / delta_k) + (U_k_a.T @ U_k_a) / bar_lam  # noqa: N806

            def _woodbury(b):
                """Apply ``T0^{-1}`` to ``b`` via the Woodbury identity."""
                return b / bar_lam - U_k_a @ (np.linalg.solve(W, U_k_a.T @ b) / bar_lam**2)

            if self.rho == 0.0 or self.mu is None:
                v = _woodbury(np.ones(n_a))
                return v / v.sum(), 1
            v1 = _woodbury(np.ones(n_a))
            v2 = _woodbury(self.mu[active])
            half_rho = 0.5 * self.rho
            half_lambda = (1.0 - half_rho * v2.sum()) / v1.sum()
            return half_lambda * v1 + half_rho * v2, 1

        x_a = self.X[:, active]
        if self.target is None:
            sigma = (x_a.T @ x_a) / self.t
        else:
            sigma = (1.0 - self.alpha) * (x_a.T @ x_a) / self.t + self.alpha * self.target[np.ix_(active, active)]

        if self.rho == 0.0 or self.mu is None:
            v = spd_solve(sigma, np.ones(n_a), assume_a="pos")
            return v / v.sum(), 1

        v1, v2 = spd_solve(sigma, np.column_stack([np.ones(n_a), self.mu[active]]), assume_a="pos").T
        half_rho = 0.5 * self.rho
        half_lambda = (1.0 - half_rho * v2.sum()) / v1.sum()
        return half_lambda * v1 + half_rho * v2, 1

    def _cvxpy_constraints(self, w, cp):
        """Return budget-equality and long-only inequality constraints for CVXPY."""
        return [cp.sum(w) == 1, w >= 0]

    def _cg_step(self, active, x0=None):
        """Solve the reduced SPD system via matrix-free CG; return ``(w_a, iters)``.

        Builds a ``LinearOperator`` for ``v -> (1-alpha)*X_a'*(X_a*v) + alpha*T0_a*v``
        and runs conjugate gradients without ever forming ``Sigma_a`` explicitly.
        When ``target_lr = (bar_lam, U_k, delta_k)`` is supplied the target term is
        applied as ``bar_lam*v + U_k_a @ (delta_k * (U_k_a.T @ v))`` at O(n_a*k)
        per call instead of O(n_a^2) for a dense submatrix.

        Args:
            active: Boolean mask selecting the active asset subset.
            x0: Optional initial guess for the first CG solve (warm start).
                When provided it must have shape ``(active.sum(),)``.
        """
        x_a = self.X[:, active]
        n_a = int(active.sum())

        # Build the target application — prefer low-rank factors over dense matrix.
        if self.target_lr is not None:
            bar_lam_lr, U_k_lr, delta_k_lr = self.target_lr  # noqa: N806
            U_k_a = U_k_lr[active, :]  # noqa: N806  # (n_a, k)
            c_data = 1.0 - self.alpha
            c_lr = self.alpha

            def _apply_target(v):
                """Apply low-rank target: bar_lam * v + U (delta * (U^T v))."""
                return bar_lam_lr * v + U_k_a @ (delta_k_lr * (U_k_a.T @ v))
        else:
            target_sub = self.target[np.ix_(active, active)] if self.target is not None else None
            c_data = 1.0 - self.alpha if target_sub is not None else 1.0
            c_lr = self.alpha if target_sub is not None else 0.0

            def _apply_target(v):
                """Apply dense target submatrix to v."""
                return target_sub @ v  # type: ignore[operator]

        count1 = [0]

        def matvec(v):
            """Apply Sigma_a to v for the budget-constraint CG solve."""
            count1[0] += 1
            result = c_data * (x_a.T @ (x_a @ v)) / self.t
            if c_lr:
                result = result + c_lr * _apply_target(v)
            return result

        op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)  # type: ignore[call-arg, missing-argument, unknown-argument, parameter-already-assigned]  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]

        if self.rho == 0.0 or self.mu is None:
            v, _ = cg(op, np.ones(n_a), x0=x0)
            return v / v.sum(), count1[0]

        count2 = [0]

        def matvec2(v):
            """Apply Sigma_a to v for the return-tilt CG solve."""
            count2[0] += 1
            result = c_data * (x_a.T @ (x_a @ v)) / self.t
            if c_lr:
                result = result + c_lr * _apply_target(v)
            return result

        op2 = LinearOperator((n_a, n_a), matvec=matvec2, dtype=np.float64)  # type: ignore[call-arg, missing-argument, unknown-argument, parameter-already-assigned]  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]
        v1, _ = cg(op, np.ones(n_a), x0=x0)
        v2, _ = cg(op2, self.mu[active], x0=x0)
        half_rho = 0.5 * self.rho
        half_lambda = (1.0 - half_rho * v2.sum()) / v1.sum()
        return half_lambda * v1 + half_rho * v2, count1[0] + count2[0]

    def _pcg_step(self, active, x0=None):
        """Solve the reduced SPD system via PCG with RMT preconditioner; return (w_a, iters).

        The system matrix is the oracle-LW covariance (using self.alpha and self.target).
        The preconditioner P = T0^RMT is applied via the Woodbury identity:
          P^{-1} v = (1/bar_lam) v + U_k diag(1/lambda_k - 1/bar_lam) U_k^T v
        costing O(n_a * k) per application.  Requires self.pcg_lr to be set.
        """
        x_a = self.X[:, active]
        n_a = int(active.sum())

        # System matvec — identical path to _cg_step
        if self.target_lr is not None:
            bar_lam_lr, U_k_lr, delta_k_lr = self.target_lr  # noqa: N806
            U_k_a_sys = U_k_lr[active, :]  # noqa: N806
            c_data, c_lr = 1.0 - self.alpha, self.alpha

            def _apply_system(v):
                """Apply the LR target submatrix to v (RMT low-rank path)."""
                return bar_lam_lr * v + U_k_a_sys @ (delta_k_lr * (U_k_a_sys.T @ v))
        else:
            target_sub = self.target[np.ix_(active, active)] if self.target is not None else None
            c_data = 1.0 - self.alpha if target_sub is not None else 1.0
            c_lr = self.alpha if target_sub is not None else 0.0

            def _apply_system(v):
                """Apply the dense target submatrix to v (full-matrix path)."""
                return target_sub @ v  # type: ignore[operator]

        count = [0]

        def matvec(v):
            """Apply the active-set system matrix Sigma_a to v."""
            count[0] += 1
            result = c_data * (x_a.T @ (x_a @ v)) / self.t
            if c_lr:
                result = result + c_lr * _apply_system(v)
            return result

        op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)  # type: ignore[call-arg, missing-argument, unknown-argument, parameter-already-assigned]  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]

        # Preconditioner P^{-1}: Woodbury inverse of T0^RMT restricted to active set
        pcg_lr = self.pcg_lr
        if pcg_lr is None:
            raise RuntimeError("_pcg_step called without pcg_lr")  # noqa: TRY003
        bar_lam_p, U_k_p, delta_k_p = pcg_lr  # noqa: N806
        U_k_a_p = U_k_p[active, :]  # noqa: N806  # (n_a, k)
        inv_coeff = 1.0 / (bar_lam_p + delta_k_p) - 1.0 / bar_lam_p  # (k,) negative

        def precond(v):
            """Apply P^{-1} to v via the Woodbury identity."""
            return (1.0 / bar_lam_p) * v + U_k_a_p @ (inv_coeff * (U_k_a_p.T @ v))

        M_op = LinearOperator((n_a, n_a), matvec=precond, dtype=np.float64)  # type: ignore[call-arg, missing-argument, unknown-argument, parameter-already-assigned]  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]  # noqa: N806

        v, _ = cg(op, np.ones(n_a), x0=x0, M=M_op)
        return v / v.sum(), count[0]

    def _constraint_active_set_warm(self, solve_fn=None, tol=1e-6, max_iter=10_000, warm_start=None):
        """Active-set loop with warm-starting; returns ``(w, iters, active, w_full)``.

        Generalises ``_constraint_active_set``: accepts an initial active set
        and passes the previous iterate as a starting guess to the inner solver.
        Solvers that support an initial guess (CG via ``_cg_step``) benefit from
        both the warm active set and the x0; direct solvers (KKT via
        ``_kkt_step``) accept and silently ignore the x0, profiting only from
        the warm active set.

        Args:
            solve_fn: Inner solver callable ``(active, x0=None) -> (w_a, iters)``.
                      Defaults to ``self._cg_step``.
            tol: Primal feasibility tolerance; assets with weight below ``-tol``
                 are dropped from the active set.
            max_iter: Maximum number of outer active-set iterations.
            warm_start: Optional ``(active_mask, w_full)`` from a previous call.
                        ``active_mask`` is a boolean array of length ``n``;
                        ``w_full`` is the full ``n``-vector of weights.

        Returns:
            ``(w, total_iters, final_active, w_full)`` — solution, cumulative
            iteration count, final active-set mask, and full weight vector
            suitable as the ``warm_start`` argument for the next solve.
        """
        if solve_fn is None:
            solve_fn = self._cg_step
        n = self.n

        if warm_start is not None:
            asset_active, last_w_full = warm_start
            asset_active = asset_active.copy()
        else:
            asset_active = np.ones(n, dtype=bool)
            last_w_full = None

        total_inner_iters = 0
        outer_steps = 0
        prev_active = None
        w = np.zeros(n)

        for _ in range(max_iter):
            if prev_active is not None and np.array_equal(prev_active, asset_active):
                break  # pragma: no cover - structurally unreachable safety guard
            prev_active = asset_active.copy()

            x0 = None
            if last_w_full is not None:
                sub = last_w_full[asset_active]
                s = sub.sum()
                if s > 1e-12:
                    x0 = sub / s

            w_a, step_iters = solve_fn(asset_active, x0=x0)
            outer_steps += 1
            total_inner_iters += step_iters

            if self._primal_drop(w_a, asset_active, tol):
                continue

            w = np.zeros(n)
            w[asset_active] = w_a
            last_w_full = w.copy()

            j = self._dual_add(self._compute_gradient(w), asset_active, tol)
            if j < 0:
                break
            asset_active[j] = True

        return w, outer_steps, total_inner_iters, asset_active.copy(), last_w_full

    def solve_cg_warm(self, *, project=True, warm_start=None):
        """Solve via matrix-free CG with warm-starting.

        Like ``solve_cg`` but accepts and returns warm-start state so that a
        sequence of related problems (e.g. an efficient-frontier sweep over
        many ``rho`` values) can chain solves together.  Adjacent problems share
        a similar active set and similar solution, so subsequent solves need
        far fewer outer iterations and CG steps.

        Args:
            project:    Clip weights to ``[0, ∞)`` and renormalize to sum to 1.
            warm_start: ``(active_mask, w_full)`` returned by a previous call,
                        or ``None`` for a cold start.

        Returns:
            ``(w, outer_steps, inner_iters, warm_state)`` — weight vector,
            number of outer active-set steps, cumulative CG iteration count,
            and warm state for the next call in the sequence.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> mu = np.ones(5) * 0.01
            >>> warm = None
            >>> for rho in [0.0, 0.5, 1.0]:
            ...     p = Problem(X, rho=rho, mu=mu)
            ...     w, outer, inner, warm = p.solve_cg_warm(warm_start=warm)
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, outer, inner, final_active, final_w = self._constraint_active_set_warm(
            solve_fn=self._cg_step, warm_start=warm_start
        )
        if project:
            w = self._clip_and_renormalize(w)
        return w, outer, inner, (final_active, final_w)

    def solve_kkt_warm(self, *, project=True, warm_start=None):
        """Solve via direct KKT factorisation with active-set warm-starting.

        Like ``solve_kkt`` but accepts and returns warm-start state for chaining
        a sequence of related problems. KKT is a direct solver, so only the
        active-set mask is warm-started (not an iterative initial guess); the
        benefit is fewer outer primal-dual iterations when consecutive problems
        share a similar active set.

        Args:
            project:    Clip weights to ``[0, ∞)`` and renormalize to sum to 1.
            warm_start: ``(active_mask, w_full)`` returned by a previous call,
                        or ``None`` for a cold start.

        Returns:
            ``(w, outer_steps, warm_state)`` — weight vector, number of outer
            active-set steps, and warm state for the next call in the sequence.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> mu = np.ones(5) * 0.01
            >>> warm = None
            >>> for rho in [0.0, 0.5, 1.0]:
            ...     p = Problem(X, rho=rho, mu=mu)
            ...     w, outer, warm = p.solve_kkt_warm(warm_start=warm)
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, outer, _inner, final_active, final_w = self._constraint_active_set_warm(
            solve_fn=self._kkt_step, warm_start=warm_start
        )
        if project:
            w = self._clip_and_renormalize(w)
        return w, outer, (final_active, final_w)

    def _clarabel_constraints(self):
        """Return budget-equality and long-only inequality constraints for Clarabel."""
        n = self.n
        a_mat = vstack(
            [csc_matrix(np.ones((1, n))), -eye(n, format="csc")],
            format="csc",
        )

        b_vec = np.concatenate([[1.0], np.zeros(n)])
        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n)]  # type: ignore[attr-defined, unresolved-attribute]  # ty:ignore[unresolved-attribute]
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
            # target is the penalty matrix M; Cholesky gives chol s.t. chol @ chol.T = M,
            # so sqrt(alpha)*chol.T rows enforce alpha * w^T M w in the LS objective.
            chol = cholesky(self.target)
            rows = [np.sqrt((1 - self.alpha) / self.t) * self.X]
            tgt = [np.zeros(t)]
            if self.alpha > 0.0:
                rows.append(np.sqrt(self.alpha) * chol.T)
                tgt.append(np.zeros(self.n))
        else:
            rows = [np.sqrt(1.0 / self.t) * self.X]
            tgt = [np.zeros(t)]
        rows.append(m * np.ones((1, self.n)))
        tgt.append(np.array([m]))

        w, _ = nnls(np.vstack(rows), np.concatenate(tgt))
        return w, 1
