"""General mean-variance portfolio problem with growing-constraint active-set."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import clarabel
import cvxpy as cp
import numpy as np
from cvx.linalg import cholesky
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, vstack
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

    def _cg_step(self, active: np.ndarray) -> tuple[np.ndarray, int]:
        """Solve the KKT saddle-point system via MINRES; return ``(w, iters)``."""
        op, rhs = self._kkt_operator(active=active)
        iters = [0]

        def _count(_: np.ndarray) -> None:
            """Increment iteration counter on each MINRES callback."""
            iters[0] += 1

        x, _ = minres(op, rhs, callback=_count)
        return x[: self.n], iters[0]

    A: np.ndarray | None = None
    b: np.ndarray | None = None
    C: np.ndarray | None = None
    d: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Fill in default constraint matrices when not supplied."""
        super().__post_init__()
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
        return int(self.A.shape[1])

    # ------------------------------------------------------------------
    # Active-set loop (growing: add violated inequality constraints)
    # ------------------------------------------------------------------

    def _constraint_active_set(
        self,
        solve_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
        tol: float = 1e-6,  # noqa: ARG002
        max_iter: int = 10_000,  # noqa: ARG002
    ) -> tuple[np.ndarray, int, int]:
        """Run the active-set loop, promoting violated inequalities to equalities."""
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        p = self.d.size
        active = np.zeros(p, dtype=bool)
        outer_steps = 0
        total_iters = 0

        while True:
            w, step_iters = solve_fn(active)
            violations = self.C[:, ~active].T @ w - self.d[~active]
            outer_steps += 1
            total_iters += step_iters
            if np.all(violations <= 1e-10):
                break
            active[~active] |= violations > 1e-10

        return w, outer_steps, total_iters

    # ------------------------------------------------------------------
    # Inner steps (called by the template solve_* methods on the base)
    # ------------------------------------------------------------------

    def _kkt_step(self, active: np.ndarray) -> tuple[np.ndarray, int]:
        """Solve the full KKT system directly; return ``(w, 1)``."""
        K, rhs = self._kkt(active=active)  # noqa: N806
        return np.linalg.solve(K, rhs)[: self.n], 1

    def _cvxpy_constraints(self, w: cp.Variable, cp: object) -> list[Any]:  # noqa: ARG002
        """Return equality and inequality constraints for CVXPY."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        return [self.A.T @ w == self.b, self.C.T @ w <= self.d]

    # ------------------------------------------------------------------
    # Operator builders (also accessed directly by tests)
    # ------------------------------------------------------------------

    def _kkt(self, active: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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

        # ridge = self._ridge()
        # oma = 1.0 - self.alpha
        K = np.zeros((self.n + m, self.n + m))  # noqa: N806
        if self.target is None:
            K[: self.n, : self.n] = 2 * (self.X.T @ self.X) / self.t
        else:
            K[: self.n, : self.n] = 2 * ((1 - self.alpha) * (self.X.T @ self.X) / self.t + self.alpha * self.target)
        K[: self.n, self.n :] = A
        K[self.n :, : self.n] = A.T

        rhs = np.zeros(self.n + m)
        if self.rho != 0.0 and self.mu is not None:
            rhs[: self.n] = self.rho * self.mu
        rhs[self.n :] = b

        return K, rhs

    def _kkt_operator(self, active: np.ndarray | None = None) -> tuple[LinearOperator, np.ndarray]:
        """Build the matrix-free KKT saddle-point operator and RHS for MINRES."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        na, ma = self.n, aa.shape[1]

        def _matvec(
            x: np.ndarray,
            xx: np.ndarray = self.X,
            a: np.ndarray = aa,
            n_: int = na,
            m_: int = ma,
        ) -> np.ndarray:
            """Apply the KKT saddle-point matrix to vector ``x``."""
            out = np.empty(n_ + m_)
            if self.target is None:
                out[:n_] = 2.0 * (xx.T @ (xx @ x[:n_])) / self.t + a @ x[n_:]
            else:
                out[:n_] = (
                    2.0 * ((1.0 - self.alpha) * (xx.T @ (xx @ x[:n_])) / self.t + self.alpha * (self.target @ x[:n_]))
                    + a @ x[n_:]
                )
            out[n_:] = a.T @ x[:n_]
            return out

        rhs = np.zeros(na + ma)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:na] = self.rho * self.mu
        rhs[na:] = np.concatenate([self.b, self.d[active]])

        op = LinearOperator(shape=(na + ma, na + ma), matvec=_matvec)  # ty:ignore[missing-argument, unknown-argument]
        return op, rhs

    def _clarabel_constraints(self) -> tuple[csc_matrix, np.ndarray, list[Any]]:
        """Return equality and inequality constraints for Clarabel."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        a_mat = vstack([csc_matrix(self.A.T), csc_matrix(self.C.T)], format="csc")
        b_vec = np.concatenate([self.b, self.d])
        cones = [clarabel.ZeroConeT(self._m), clarabel.NonnegativeConeT(len(self.d))]  # ty:ignore[unresolved-attribute]
        return a_mat, b_vec, cones

    def _osqp_constraints(self) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
        """Return equality and inequality constraints for OSQP."""
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        assert self.C is not None  # noqa: S101
        assert self.d is not None  # noqa: S101
        a_mat = vstack([csc_matrix(self.A.T), csc_matrix(self.C.T)], format="csc")
        l_vec = np.concatenate([self.b, np.full(len(self.d), -np.inf)])
        u_vec = np.concatenate([self.b, self.d])
        return a_mat, l_vec, u_vec

    def _nnls_solve(self) -> tuple[np.ndarray, int]:
        """Solve via NNLS on the augmented return matrix; return ``(w, 1)``.

        Augments ``X`` with rows for the LW ridge term and all equality
        constraints (scaled by ``M = ||X||_F * T``); non-negativity is
        handled natively by Lawson-Hanson.  Inequality constraints beyond
        ``w >= 0`` are not enforced; use ``solve_kkt`` for general ``C``.
        """
        assert self.A is not None  # noqa: S101
        assert self.b is not None  # noqa: S101
        t = self.X.shape[0]
        # oma = 1.0 - self.alpha
        # gamma = self._ridge()
        m = float(np.linalg.norm(self.X, "fro")) * t

        if self.target is not None:
            chol = cholesky(self.target)
            rows = [np.sqrt((1 - self.alpha) / self.t) * self.X]
            tgt = [np.zeros(t)]
            if self.alpha > 0.0:
                rows.append(np.sqrt(self.alpha) * chol.T)
                tgt.append(np.zeros(self.n))
        else:
            rows = [np.sqrt(1.0 / self.t) * self.X]
            tgt = [np.zeros(t)]
        rows.append(m * self.A.T)
        tgt.append(m * self.b)

        w, _ = nnls(np.vstack(rows), np.concatenate(tgt))
        return w, 1
