"""General mean-variance portfolio problem with growing-constraint active-set."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.sparse.linalg import LinearOperator, minres

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

    def _cg_step(self, active):
        """Solve the KKT saddle-point system via MINRES; return ``(w, iters)``."""
        op, rhs = self._kkt_operator(active=active)
        iters = [0]

        def _count(_):
            iters[0] += 1

        x, _ = minres(op, rhs, callback=_count)
        return x[: self.n], iters[0]

    A: np.ndarray | None = None
    b: np.ndarray | None = None
    C: np.ndarray | None = None
    d: np.ndarray | None = None

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
        assert self.A is not None  # noqa: S101
        return self.A.shape[1]

    # ------------------------------------------------------------------
    # Active-set loop (growing: add violated inequality constraints)
    # ------------------------------------------------------------------

    def _constraint_active_set(self, solve_fn, tol=1e-6, max_iter=10_000):
        """Run the active-set loop, promoting violated inequalities to equalities."""
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
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

    def _cvxpy_constraints(self, w, cp):
        """Return equality and inequality constraints for CVXPY."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        return [self.A.T @ w == self.b, self.C.T @ w <= self.d]

    # ------------------------------------------------------------------
    # Operator builders (also accessed directly by tests)
    # ------------------------------------------------------------------

    def _kkt(self, active=None):
        """Build the (N+m) x (N+m) KKT saddle-point system."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
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
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
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

    def _nnls_solve(self):
        """Solve via NNLS on the augmented return matrix; return ``(w, 1)``.

        Augments ``X`` with rows for the LW ridge term and all equality
        constraints (scaled by ``M = ||X||_F * T``); non-negativity is
        handled natively by Lawson-Hanson.  Inequality constraints beyond
        ``w >= 0`` are not enforced; use ``solve_kkt`` for general ``C``.
        """
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        t = self.X.shape[0]
        oma = 1.0 - self.alpha
        gamma = self._ridge()
        m = float(np.linalg.norm(self.X, "fro")) * t

        rows = [np.sqrt(oma) * self.X]
        tgt = [np.zeros(t)]
        if gamma > 0.0:
            rows.append(np.sqrt(gamma) * np.eye(self.n))
            tgt.append(np.zeros(self.n))
        rows.append(m * self.A.T)
        tgt.append(m * self.b)

        w, _ = nnls(np.vstack(rows), np.concatenate(tgt))
        return w, 1
