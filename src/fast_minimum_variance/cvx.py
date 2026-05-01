"""Convex optimization solver for the minimum variance and Markowitz portfolio."""

import cvxpy as cp

from .api import API, clip_and_renormalize


def solve_cvxpy(api: API, *, project: bool = True):
    """Solve the general mean-variance portfolio via CVXPY.

    Solves::

        min  ||X w||_2^2 - rho * mu @ w
        s.t. A.T @ w == b
             C.T @ w <= d

    Defaults recover the long-only minimum variance problem::

        min  ||X w||_2^2
        s.t. sum(w) == 1,  w >= 0

    Args:
        api:     API dataclass holding X, A, b, C, d, rho, mu.
        project: If True (default), clip weights to non-negative and renormalize
                 to sum to one after solving.  Only correct for the default
                 long-only minimum-variance problem; set to False when using
                 custom constraints.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the number of interior-point iterations reported by CLARABEL.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance.api import API
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_cvxpy(API(X))
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= -1e-6).all())
        True
    """
    # CVXPY decision variable: weight vector of length N.
    w = cp.Variable(api.n)

    # Objective: portfolio variance ||X w||^2 = w^T (X^T X) w.
    # CVXPY forms X^T X implicitly; sum_squares(X @ w) is recognised as a
    # quadratic form and passed to the solver as a second-order cone constraint.
    objective = cp.sum_squares(api.X @ w)

    # Return term: subtract rho * mu^T w to tilt toward higher-expected-return
    # assets.  Omitted when rho == 0 to keep the problem purely quadratic.
    if api.rho != 0.0:
        objective = objective - api.rho * (api.mu @ w)

    # Equality constraints enforce A^T w = b (e.g. sum(w) = 1 for budget).
    # Inequality constraints enforce C^T w <= d (e.g. w >= 0 for long-only,
    # encoded as -I w <= 0).
    constraints = [api.A.T @ w == api.b, api.C.T @ w <= api.d]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    # CLARABEL is an interior-point solver for conic programs; it handles the
    # quadratic objective and linear constraints natively.
    problem.solve(solver=cp.CLARABEL)

    result = w.value
    if project:
        result = clip_and_renormalize(result)
    # solver_stats.num_iters counts the interior-point iterations taken.
    return result, problem.solver_stats.num_iters
