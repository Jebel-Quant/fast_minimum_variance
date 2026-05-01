"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator


@dataclass
class API:
    """Dataclass for the portfolio solver API."""

    X: np.ndarray
    rho: float = 0.0
    mu: np.ndarray | None = None
    gamma: float = 0.0
    A: np.ndarray = field(default=None)  # type: ignore[assignment]
    b: np.ndarray = field(default=None)  # type: ignore[assignment]
    C: np.ndarray = field(default=None)  # type: ignore[assignment]
    d: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        """Fill in default constraint matrices when not supplied."""
        n = self.n
        if self.A is None:
            self.A = np.ones((n, 1))
        if self.b is None:
            self.b = np.ones(1)
        if self.C is None:
            self.C = -np.eye(n)
        if self.d is None:
            self.d = np.zeros(n)

    @property
    def n(self) -> int:
        """Number of assets in the return matrix."""
        return self.X.shape[1]

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self.A.shape[1]

    def kkt(self, active=None):
        """Build the (N+m) x (N+m) KKT saddle-point system, optionally pinning active inequalities.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (K, rhs) where K is the KKT matrix and rhs is the right-hand side.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        A = np.hstack([self.A, self.C[:, active]])  # noqa: N806
        b = np.concatenate([self.b, self.d[active]])

        m = A.shape[1]

        # The KKT matrix is an (N+m) x (N+m) indefinite saddle-point system.
        # The top-left block is the Hessian of the objective (2(X^T X + gamma I)); the
        # off-diagonal blocks enforce the equality constraints via Lagrange
        # multipliers; the bottom-right block is zero because there is no
        # quadratic term in the dual variable.
        K = np.zeros((self.n + m, self.n + m))  # noqa: N806
        K[: self.n, : self.n] = 2 * (self.X.T @ self.X + self.gamma * np.eye(self.n))
        K[: self.n, self.n :] = A
        K[self.n :, : self.n] = A.T

        # The primal block of the RHS is the return term (zero for pure min-var);
        # the dual block is the equality RHS b (e.g. [1] for the budget constraint).
        rhs = np.zeros(self.n + m)
        if self.rho != 0.0 and self.mu is not None:
            rhs[: self.n] = self.rho * self.mu
        rhs[self.n :] = b

        return K, rhs

    def kkt_operator(self, active=None):
        """Build the matrix-free KKT saddle-point operator and RHS for MINRES.

        Returns a ``LinearOperator`` that applies the (N+m) x (N+m) KKT system
        matrix-free (never forming X^T X explicitly), and the matching RHS vector.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (operator, rhs) where operator is a ``LinearOperator`` of shape
            (N+m, N+m) and rhs is the right-hand side vector of length N+m.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        na, ma = self.n, aa.shape[1]

        def _matvec(x, xx=self.X, a=aa, n_=na, m_=ma, gam=self.gamma):
            """Apply the KKT saddle-point operator to vector x."""
            out = np.empty(n_ + m_)
            out[:n_] = 2.0 * (xx.T @ (xx @ x[:n_]) + gam * x[:n_]) + a @ x[n_:]
            out[n_:] = a.T @ x[:n_]
            return out

        rhs = np.zeros(na + ma)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:na] = self.rho * self.mu
        rhs[na:] = np.concatenate([self.b, self.d[active]])

        return LinearOperator(shape=(na + ma, na + ma), matvec=_matvec), rhs  # type: ignore[call-arg]

    def null_space_operator(self, active=None):
        """Build the reduced null-space operator and RHS for CG.

        Computes a ``LinearOperator`` for ``P^T (X^T X + gamma I) P``, the
        projected RHS, the particular solution ``w0``, and the null-space basis
        ``P``.  When the constraints fully determine ``w`` (no free directions),
        returns ``(None, None, w0, None)``.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (op, rhs, w0, P) where op is a ``LinearOperator`` of shape
            (n_free, n_free), rhs has length n_free, w0 has length N, and P
            has shape (N, n_free).  Returns (None, None, w0, None) when the
            system is fully determined.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        m_ext = aa.shape[1]
        n_free = self.n - m_ext
        b_ext = np.concatenate([self.b, self.d[active]])

        w0 = np.linalg.lstsq(aa.T, b_ext, rcond=None)[0]

        if n_free <= 0:
            return None, None, w0, None

        Q, _ = np.linalg.qr(aa, mode="complete")  # noqa: N806
        P = Q[:, m_ext:]  # noqa: N806

        # w0 lies in range(aa), P spans null(aa^T), so P^T w0 = 0 and the
        # gamma regularisation term drops out of the projected gradient.
        g0 = self.X.T @ (self.X @ w0) + self.gamma * w0
        if self.rho != 0.0 and self.mu is not None:
            g0 = g0 - (self.rho / 2.0) * self.mu

        rhs = -(P.T @ g0)

        def _matvec(y, pp=P, xx=self.X, gam=self.gamma):
            pv = pp @ y
            return pp.T @ (xx.T @ (xx @ pv)) + gam * y

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        return op, rhs, w0, P

    def constraint_active_set(self, solve_fn):
        """Run the constraint active-set loop, promoting violated inequalities to equalities.

        Starts with no inequality constraints active and iteratively adds violated
        ones until all inactive constraints are satisfied.

        Args:
            C:        Inequality constraint matrix of shape (N, p) for C.T @ w <= d.
            d:        Inequality RHS of shape (p,).
            solve_fn: Callable ``(active) -> (w, n_iters)`` that solves the equality-constrained
                      subproblem for the given active-constraint mask and returns the
                      full weight vector of shape (N,) and the number of solver iterations.

        Returns:
            Tuple (w, total_iters).
        """
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
