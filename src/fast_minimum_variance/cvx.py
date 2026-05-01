"""Convex optimization solver for the minimum variance and Markowitz portfolio."""

import cvxpy as cp

from .api import API, clip_and_renormalize


def solve_cvxpy(api: API, *, project: bool = True):
    """Solve the general mean-variance portfolio via CVXPY.

    Solves::

        min  ||X w||_2^2 + gamma * ||w||_2^2 - rho * mu @ w
        s.t. A.T @ w == b
             C.T @ w <= d

    Defaults recover the long-only minimum variance problem::

        min  ||X w||_2^2
        s.t. sum(w) == 1,  w >= 0

    Args:
        api:     API dataclass holding X, A, b, C, d, rho, mu, gamma.
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
    w = cp.Variable(api.n)

    # sum_squares(X @ w) = ||X w||^2 = w^T (X^T X) w.
    # CVXPY recognises this as a quadratic form and passes it to CLARABEL as a
    # second-order cone constraint without ever forming X^T X explicitly.
    objective = cp.sum_squares(api.X @ w)

    # Ridge / Ledoit-Wolf regularization: add gamma * ||w||^2.
    # sum_squares(w) is also a quadratic form recognised by CVXPY, so the
    # combined objective remains a single SOCP handled natively by CLARABEL.
    if api.gamma != 0.0:
        objective = objective + api.gamma * cp.sum_squares(w)

    # Subtract the return term when rho > 0 (Markowitz mean-variance tilt).
    # Omitting it for the pure minimum-variance case avoids introducing mu as a
    # required parameter.
    if api.rho != 0.0:
        objective = objective - api.rho * (api.mu @ w)

    # Equality constraints: A^T w = b  (e.g. sum(w) = 1 for budget).
    # Inequality constraints: C^T w <= d  (e.g. -I w <= 0 for long-only).
    constraints = [api.A.T @ w == api.b, api.C.T @ w <= api.d]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    # CLARABEL is an interior-point solver for second-order cone programs.
    # It is CVXPY's default for QP/SOCP and handles the saddle-point structure
    # without any user-supplied starting point.
    problem.solve(solver=cp.CLARABEL)

    result = w.value
    if result is None:
        raise RuntimeError("CVXPY solver failed to find a solution")  # noqa: TRY003
    if project:
        result = clip_and_renormalize(result)
    # solver_stats.num_iters is the interior-point iteration count reported by
    # CLARABEL; it is analogous to the Krylov iteration counts returned by the
    # other solvers, enabling a fair comparison.
    return result, problem.solver_stats.num_iters
