"""Minimum-variance solver with shrinking active-set strategy."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr as _scipy_qr
from scipy.sparse.linalg import LinearOperator, cg, minres

from ._base import _BaseProblem


@dataclass(frozen=True)
class MinVarProblem(_BaseProblem):
    """Minimum-variance portfolio with shrinking active-set solvers.

    Specialised for::

        min  (1-alpha)||X w||^2 + alpha*(||X||_F^2/N)*||w||^2 - rho*mu^T w
        s.t. 1^T w = 1,  w >= 0

    Unlike :class:`~fast_minimum_variance.problem.Problem`, the active-set
    here *removes* assets with negative weights from the subproblem instead
    of pinning them as equality constraints.  The KKT system shrinks from
    ``(n+1)x(n+1)`` down to ``(n*+1)x(n*+1)`` where ``n*`` is the final
    portfolio size, yielding dramatically fewer MINRES iterations on real
    equity data (e.g. 214 vs 1065 on S&P 500 without shrinkage).

    Use ``alpha = N/(N+T)`` for Ledoit-Wolf shrinkage intensity::

        T, N = X.shape
        w, iters = MinVarProblem(X, alpha=N/(N+T)).solve_minres()

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.minvar_problem import MinVarProblem
        >>> X = np.random.default_rng(0).standard_normal((100, 5))
        >>> w, iters = MinVarProblem(X).solve_minres()
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
    """

    # No extra fields — X, alpha, rho, mu all inherited from _BaseProblem.

    # ------------------------------------------------------------------
    # Active-set loop (shrinking: remove assets with negative weights)
    # ------------------------------------------------------------------

    def _constraint_active_set(self, solve_fn):
        """Shrinking active-set: remove assets with negative weights.

        ``solve_fn(asset_active)`` receives a boolean mask of shape ``(n,)``
        and returns ``(w_a, step_iters)`` where ``w_a`` is the weight vector
        for active assets only (shape ``(sum(asset_active),)``).
        """
        n = self.n
        asset_active = np.ones(n, dtype=bool)
        total_iters = 0

        while True:
            w_a, step_iters = solve_fn(asset_active)
            total_iters += step_iters
            if np.all(w_a >= -1e-10):
                break
            idx = np.where(asset_active)[0]
            asset_active[idx[w_a < 0]] = False

        w = np.zeros(n)
        w[asset_active] = w_a
        return w, total_iters

    # ------------------------------------------------------------------
    # Inner steps
    # ------------------------------------------------------------------

    def _kkt_step(self, asset_active):
        """Solve the reduced KKT system directly; return ``(w_a, 1)``."""
        K, rhs = self._kkt_reduced(asset_active)  # noqa: N806
        n_a = int(asset_active.sum())
        return np.linalg.solve(K, rhs)[:n_a], 1

    def _minres_step(self, asset_active):
        """Solve the reduced KKT system via MINRES; return ``(w_a, iters)``."""
        kkt, rhs = self._kkt_operator_reduced(asset_active)
        n_a = int(asset_active.sum())
        iters = [0]
        sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        return sol[:n_a], iters[0]

    def _cg_step(self, asset_active):
        """Solve via CG in the null space of the budget constraint; return ``(w_a, iters)``."""
        op, rhs, w0, reconstruct = self._null_space_operator_reduced(asset_active)
        if op is None:
            return w0, 0
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        return reconstruct(sol), iters[0]

    def _cvxpy_constraints(self, w, cp):
        """Return budget-equality and long-only inequality constraints for CVXPY."""
        return [cp.sum(w) == 1, w >= 0]

    # ------------------------------------------------------------------
    # Operator builders for the active-asset subproblem
    # ------------------------------------------------------------------

    def _kkt_reduced(self, asset_active: np.ndarray):
        """Build the ``(n_a+1)x(n_a+1)`` explicit KKT matrix for active assets."""
        x_a = self.X[:, asset_active]
        n_a = int(asset_active.sum())
        ridge = self._ridge()
        oma = 1.0 - self.alpha

        K = np.zeros((n_a + 1, n_a + 1))  # noqa: N806
        K[:n_a, :n_a] = 2.0 * (oma * (x_a.T @ x_a) + ridge * np.eye(n_a))
        K[:n_a, n_a] = 1.0
        K[n_a, :n_a] = 1.0

        rhs = np.zeros(n_a + 1)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:n_a] = self.rho * self.mu[asset_active]
        rhs[n_a] = 1.0

        return K, rhs

    def _kkt_operator_reduced(self, asset_active: np.ndarray):
        """Matrix-free ``(n_a+1)x(n_a+1)`` KKT operator for active assets."""
        x_a = self.X[:, asset_active]
        n_a = int(asset_active.sum())
        ridge = self._ridge()
        oma = 1.0 - self.alpha

        def _matvec(x, ra=x_a, na=n_a, oma_=oma, rid=ridge):
            """Apply the KKT saddle-point matrix to vector ``x``."""
            out = np.empty(na + 1)
            out[:na] = 2.0 * (oma_ * (ra.T @ (ra @ x[:na])) + rid * x[:na]) + x[na]
            out[na] = x[:na].sum()
            return out

        rhs = np.zeros(n_a + 1)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:n_a] = self.rho * self.mu[asset_active]
        rhs[n_a] = 1.0

        return LinearOperator(shape=(n_a + 1, n_a + 1), matvec=_matvec), rhs  # type: ignore[call-arg]

    def _null_space_operator_reduced(self, asset_active: np.ndarray):
        """CG null-space operator for active assets (budget constraint only).

        Uses a single Householder reflector (WY with m_ext=1) that maps
        ``ones(n_a)`` to ``sqrt(n_a)*e_1``.  For m_ext=1 the reflector is
        symmetric, so apply-Q = apply-Q^T.

        Returns ``(op, rhs, w0, reconstruct)`` or ``(None, None, w0, None)``
        when the system is fully determined (``n_a <= 1``).
        """
        x_a = self.X[:, asset_active]
        n_a = int(asset_active.sum())
        ridge = self._ridge()
        oma = 1.0 - self.alpha

        if n_a <= 1:
            return None, None, np.ones(n_a) / max(n_a, 1), None

        aa = np.ones((n_a, 1))
        (qr_packed, tau), r_upper = _scipy_qr(aa, mode="raw")

        Y = np.tril(qr_packed[:, :1], k=-1) + np.eye(n_a, 1)  # n_a x 1  # noqa: N806
        t_scalar = -tau[0]
        y_h = Y.ravel()

        def _apply_h(v):
            """Apply the single Householder reflector H = I + t * y * y^T."""
            return v + t_scalar * (y_h @ v) * y_h

        z = np.zeros(n_a)
        z[0] = np.linalg.solve(r_upper[:1, :1].T, np.ones(1))[0]
        w0 = _apply_h(z)

        g0 = oma * (x_a.T @ (x_a @ w0)) + ridge * w0
        if self.rho != 0.0 and self.mu is not None:
            g0 = g0 - (self.rho / 2.0) * self.mu[asset_active]

        rhs_ns = -_apply_h(g0)[1:]

        def _matvec(v, xa=x_a, oma_=oma, rid=ridge):
            """Apply the reduced null-space operator to vector ``v``."""
            z_ = np.zeros(n_a)
            z_[1:] = v
            w = _apply_h(z_)
            return _apply_h(oma_ * (xa.T @ (xa @ w)) + rid * w)[1:]

        def _reconstruct(v):
            """Reconstruct full active-asset weights from null-space coordinate ``v``."""
            z_ = np.zeros(n_a)
            z_[1:] = v
            return w0 + _apply_h(z_)

        op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)  # type: ignore[call-arg]
        return op, rhs_ns, w0, _reconstruct
