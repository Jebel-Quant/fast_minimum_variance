"""Convex optimization solver for the minimum variance and Markowitz portfolio."""

import cvxpy as cp
import numpy as np


def solve_cvxpy(X, A=None, b=None, C=None, d=None, rho=0.0, mu=None):  # noqa: N803
    """Solve the general mean-variance portfolio via CVXPY.

    Solves::

        min  ||X w||_2^2 - rho * mu @ w
        s.t. A.T @ w == b
             C.T @ w <= d

    Defaults recover the long-only minimum variance problem::

        min  ||X w||_2^2
        s.t. sum(w) == 1,  w >= 0

    Args:
        X:   Return matrix of shape (T, N).
        A:   Equality constraint matrix of shape (N, m).
             Defaults to ones((N, 1)) (budget constraint).
        b:   Equality RHS of shape (m,).
             Defaults to [1.0].
        C:   Inequality constraint matrix of shape (N, p).
             Defaults to -eye(N) (long-only constraint).
        d:   Inequality RHS of shape (p,).
             Defaults to zeros(N).
        rho: Risk-aversion parameter (>= 0). Default 0.
        mu:  Expected return vector of shape (N,). Required when rho > 0.

    Returns:
        Weight vector of shape (N,).

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w = solve_cvxpy(X)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= -1e-6).all())
        True
    """
    n = X.shape[1]

    if A is None:
        A = np.ones((n, 1))  # noqa: N806
    if b is None:
        b = np.ones(1)
    if C is None:
        C = -np.eye(n)  # noqa: N806
    if d is None:
        d = np.zeros(n)

    w = cp.Variable(n)
    objective = cp.sum_squares(X @ w)
    if rho != 0.0:
        objective = objective - rho * (mu @ w)

    constraints = [A.T @ w == b, C.T @ w <= d]
    cp.Problem(cp.Minimize(objective), constraints).solve(solver=cp.CLARABEL)
    return w.value
