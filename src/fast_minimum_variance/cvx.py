"""Convex optimization solver for the minimum variance portfolio."""

import cvxpy as cp


def minvar_cvxpy(R):  # noqa: N803
    """Solve the minimum variance portfolio via CVXPY.

    Solves the long-only minimum variance problem::

        min  ||R w||_2^2
        s.t. sum(w) = 1,  w >= 0

    using CVXPY with its default solver.

    Args:
        R: Return matrix of shape (T, N).

    Returns:
        Weight vector of shape (N,) summing to 1 with all non-negative entries.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w = minvar_cvxpy(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= -1e-6).all())
        True
    """
    n = R.shape[1]
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.sum_squares(R @ w)), [cp.sum(w) == 1, w >= 0]).solve()
    return w.value
