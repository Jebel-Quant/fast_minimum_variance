"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class API:
    """Dataclass for the portfolio solver API."""

    X: np.ndarray
    rho: float = 0.0
    mu: np.ndarray | None = None
    gamma: float = 0.0
    A: np.ndarray | None = field(default=None)
    b: np.ndarray | None = field(default=None)
    C: np.ndarray | None = field(default=None)
    d: np.ndarray | None = field(default=None)

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
