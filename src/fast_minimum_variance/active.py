"""Active-set loop helpers for the minimum variance and Markowitz portfolio."""

import numpy as np


def constraint_active_set(C, d, solve_fn):  # noqa: N803
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
    p = d.size
    active = np.zeros(p, dtype=bool)
    total_iters = 0

    while True:
        w, step_iters = solve_fn(active)
        violations = C[:, ~active].T @ w - d[~active]
        total_iters += step_iters
        if np.all(violations <= 1e-10):
            break
        active[~active] |= violations > 1e-10

    return w, total_iters
