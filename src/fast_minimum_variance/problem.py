"""General mean-variance portfolio problem with growing-constraint active-set."""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr as _scipy_qr
from scipy.sparse.linalg import LinearOperator, cg, minres

from ._base import _BaseProblem


@dataclass(frozen=True)
class _Problem(_BaseProblem):
    """Mean-variance portfolio problem with arbitrary linear constraints.

    Encodes the optimization problem::

        min  (1-alpha)||X w||^2 + alpha*(||X||_F^2/N)*||w||^2 - rho * mu^T w
        s.t. A^T w  = b      (equality constraints)
             C^T w <= d      (inequality constraints)

    The first term is the sample portfolio variance (X is the demeaned return
    matrix of shape T x N).  The ``alpha`` term adds a Ledoit-Wolf ridge
    ``alpha * (||X||_F^2 / N) * I`` to the covariance, improving conditioning.
    The ``rho * mu`` term tilts the portfolio toward higher-expected-return
    assets (Markowitz).

    Defaults reproduce the long-only minimum-variance problem:

    * ``A = ones(N, 1)``, ``b = [1]``  — budget constraint: sum(w) = 1
    * ``C = -I``,         ``d = 0``    — long-only: -w <= 0, i.e. w >= 0

    The active-set loop *adds* violated inequality constraints as equalities
    (growing approach), operating on the full N-dimensional system throughout.
    See :class:`~fast_minimum_variance.minvar_problem._MinVarProblem` for the
    complementary shrinking approach optimised for the default long-only case.

    Solvers::

        w, iters = Problem(X, A=A, b=b).solve_kkt()
        w, iters = Problem(X, A=A, b=b).solve_minres()
        w, iters = Problem(X, A=A, b=b).solve_cg()
        w, iters = Problem(X, A=A, b=b).solve_cvxpy()   # requires [convex] extra
    """

    A: np.ndarray = field(default=None)  # type: ignore[assignment]
    b: np.ndarray = field(default=None)  # type: ignore[assignment]
    C: np.ndarray = field(default=None)  # type: ignore[assignment]
    d: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        """Fill in default constraint matrices when not supplied."""
        n = self.n
        if self.A is None:
            object.__setattr__(self, "A", np.ones((n, 1)))
        if self.b is None:
            object.__setattr__(self, "b", np.ones(1))
        if self.C is None:
            object.__setattr__(self, "C", -np.eye(n))
        if self.d is None:
            object.__setattr__(self, "d", np.zeros(n))

    @property
    def _m(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[1]

    # ------------------------------------------------------------------
    # Active-set loop (growing: add violated inequality constraints)
    # ------------------------------------------------------------------

    def _constraint_active_set(self, solve_fn):
        """Run the active-set loop, promoting violated inequalities to equalities."""
        p = self.d.size
        active = np.zeros(p, dtype=bool)
        total_iters = 0

        while True:
            w, step_iters = solve_fn(active)
            violations = self.C[:, ~active].T @ w - self.d[~active]
            total_iters += step_iters
            if np.all(violations <= 1e-10):
                break
            active[~active] |= violations > 1e-10

        return w, total_iters

    # ------------------------------------------------------------------
    # Inner steps (called by the template solve_* methods on the base)
    # ------------------------------------------------------------------

    def _kkt_step(self, active):
        """Solve the full KKT system directly; return ``(w, 1)``."""
        K, rhs = self._kkt(active=active)  # noqa: N806
        return np.linalg.solve(K, rhs)[: self.n], 1

    def _minres_step(self, active):
        """Solve the KKT saddle-point system via MINRES; return ``(w, iters)``."""
        kkt, rhs = self._kkt_operator(active)
        iters = [0]
        sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        return sol[: self.n], iters[0]

    def _cg_step(self, active):
        """Solve via CG in the null space of active constraints; return ``(w, iters)``."""
        op, rhs, w0, reconstruct = self._null_space_operator(active)
        if op is None:
            return w0, 0
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        return reconstruct(sol), iters[0]

    def _cvxpy_constraints(self, w, cp):
        """Return equality and inequality constraints for CVXPY."""
        return [self.A.T @ w == self.b, self.C.T @ w <= self.d]

    # ------------------------------------------------------------------
    # Operator builders (also accessed directly by tests)
    # ------------------------------------------------------------------

    def _kkt(self, active=None):
        """Build the (N+m) x (N+m) KKT saddle-point system."""
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        A = np.hstack([self.A, self.C[:, active]])  # noqa: N806
        b = np.concatenate([self.b, self.d[active]])
        m = A.shape[1]

        ridge = self._ridge()
        oma = 1.0 - self.alpha
        K = np.zeros((self.n + m, self.n + m))  # noqa: N806
        K[: self.n, : self.n] = 2 * (oma * (self.X.T @ self.X) + ridge * np.eye(self.n))
        K[: self.n, self.n :] = A
        K[self.n :, : self.n] = A.T

        rhs = np.zeros(self.n + m)
        if self.rho != 0.0 and self.mu is not None:
            rhs[: self.n] = self.rho * self.mu
        rhs[self.n :] = b

        return K, rhs

    def _kkt_operator(self, active=None):
        """Build the matrix-free KKT saddle-point operator and RHS for MINRES."""
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        na, ma = self.n, aa.shape[1]

        ridge = self._ridge()
        oma = 1.0 - self.alpha

        def _matvec(x, xx=self.X, a=aa, n_=na, m_=ma, oma_=oma, rid=ridge):
            """Apply the KKT saddle-point matrix to vector ``x``."""
            out = np.empty(n_ + m_)
            out[:n_] = 2.0 * (oma_ * (xx.T @ (xx @ x[:n_])) + rid * x[:n_]) + a @ x[n_:]
            out[n_:] = a.T @ x[:n_]
            return out

        rhs = np.zeros(na + ma)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:na] = self.rho * self.mu
        rhs[na:] = np.concatenate([self.b, self.d[active]])

        return LinearOperator(shape=(na + ma, na + ma), matvec=_matvec), rhs  # type: ignore[call-arg]

    def _null_space_operator(self, active=None):
        """Build the reduced null-space operator and RHS for CG.

        Uses the WY representation Q = I + Y T Y^T built from the Householder QR
        of the active constraint matrix.  No explicit null-space basis P is stored.
        Setup: O(n m²) for QR + WY factor.  Per-matvec: O(n m) via BLAS, where
        m = number of active equality + inequality constraints.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        m_ext = aa.shape[1]
        n_free = self.n - m_ext
        b_ext = np.concatenate([self.b, self.d[active]])

        if n_free <= 0:
            w0 = np.linalg.lstsq(aa.T, b_ext, rcond=None)[0]
            return None, None, w0, None

        (qr_packed, tau), R_upper = _scipy_qr(aa, mode="raw")  # noqa: N806
        Y = np.tril(qr_packed, k=-1) + np.eye(self.n, m_ext)  # noqa: N806
        T = np.zeros((m_ext, m_ext))  # noqa: N806
        T[0, 0] = -tau[0]
        for k in range(1, m_ext):
            T[:k, k] = -tau[k] * (T[:k, :k] @ (Y[k:, :k].T @ Y[k:, k]))
            T[k, k] = -tau[k]

        def _apply_q(v):
            """Apply Q = I + Y T Y^T to vector ``v``."""
            return v + Y @ (T @ (Y.T @ v))

        def _apply_qt(v):
            """Apply Q^T = I + Y T^T Y^T to vector ``v``."""
            return v + Y @ (T.T @ (Y.T @ v))

        ridge = self._ridge()
        oma = 1.0 - self.alpha
        z = np.zeros(self.n)
        z[:m_ext] = np.linalg.solve(R_upper.T, b_ext)
        w0 = _apply_q(z)

        g0 = oma * (self.X.T @ (self.X @ w0)) + ridge * w0
        if self.rho != 0.0 and self.mu is not None:
            g0 = g0 - (self.rho / 2.0) * self.mu

        rhs = -_apply_qt(g0)[m_ext:]

        def _matvec(y, xx=self.X, oma_=oma, rid=ridge):
            """Apply the null-space reduced Hessian to vector ``y``."""
            z_ = np.zeros(self.n)
            z_[m_ext:] = y
            w = _apply_q(z_)
            qw = oma_ * (xx.T @ (xx @ w)) + rid * w
            return _apply_qt(qw)[m_ext:]

        def _reconstruct(v):
            """Reconstruct full weights from null-space coordinate ``v``."""
            z_ = np.zeros(self.n)
            z_[m_ext:] = v
            return w0 + _apply_q(z_)

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        return op, rhs, w0, _reconstruct
