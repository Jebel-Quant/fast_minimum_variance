"""Convex optimization solver for the minimum variance portfolio."""

import cvxpy as cp


def minvar_cvxpy(R):  # noqa: N803
    """Solve the minimum variance portfolio via CVXPY."""
    n = R.shape[1]
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.sum_squares(R @ w)), [cp.sum(w) == 1, w >= 0]).solve()
    return w.value
