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

    @property
    def kkt(self):
        """Build the KKT system matrix and RHS for the general mean-variance problem.

        Constructs the (N+m) x (N+m) indefinite saddle-point system for::

            min  ||X w||_2^2 + gamma ||w||_2^2 - rho * mu @ w
            s.t. A.T @ w == b

        The system has the form::

            [ 2(X^T X + gamma I)   A ] [ w ]   [ rho * mu ]
            [ A^T                  0 ] [ λ ] = [ b        ]

        Defaults (A = ones((N,1)), b = [1], gamma = 0) recover the
        minimum variance KKT system of the companion paper.

        Args:
            X:     Return matrix of shape (T, N).
            A:     Equality constraint matrix of shape (N, m).
                   Defaults to ones((N, 1)) (budget constraint).
            b:     Equality RHS of shape (m,). Defaults to [1.0].
            rho:   Risk-aversion parameter (>= 0). Default 0.
            mu:    Expected return vector of shape (N,). Required when rho > 0.
            gamma: Diagonal regularisation added to the Hessian (default 0.0).

        Returns:
            Tuple (K, rhs) where K is the (N+m) x (N+m) KKT matrix and rhs is
            the (N+m,) right-hand side vector.

        Examples:
            >>> import numpy as np
            >>> X = np.eye(3)
            >>> K, rhs = build_kkt(X)
            >>> K.shape
            (4, 4)
            >>> rhs
            array([0., 0., 0., 1.])
        """
        # The KKT matrix is an (N+m) x (N+m) indefinite saddle-point system.
        # The top-left block is the Hessian of the objective (2(X^T X + gamma I)); the
        # off-diagonal blocks enforce the equality constraints via Lagrange
        # multipliers; the bottom-right block is zero because there is no
        # quadratic term in the dual variable.
        K = np.zeros((self.n + self.m, self.n + self.m))  # noqa: N806
        K[: self.n, : self.n] = 2 * (self.X.T @ self.X + self.gamma * np.eye(self.n))
        K[: self.n, self.n :] = self.A
        K[self.n :, : self.n] = self.A.T

        # The primal block of the RHS is the return term (zero for pure min-var);
        # the dual block is the equality RHS b (e.g. [1] for the budget constraint).
        rhs = np.zeros(self.n + self.m)
        if self.rho != 0.0 and self.mu is not None:
            rhs[: self.n] = self.rho * self.mu
        rhs[self.n :] = self.b

        return K, rhs
