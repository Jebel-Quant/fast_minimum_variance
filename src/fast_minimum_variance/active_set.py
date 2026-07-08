"""Primal-dual active-set loop enforcing the long-only constraint ``w >= 0``.

The algorithm layer of the solver: it drives repeated equality-constrained
sub-solves (a ``solve_fn`` supplied by the caller), dropping negative-weight
assets in a *primal step* and re-adding KKT-violating assets in a *dual step*
until both feasibility conditions hold. Kept separate from the problem
definition so the loop is independently testable and free of linear-algebra
detail. Functions take the problem via duck typing (attributes ``X``, ``t``,
``n``, ``alpha``, ``rho``, ``mu``, ``target``, ``target_lr``, ``B``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .minvar_problem import _MinVarProblem


def compute_gradient(problem: _MinVarProblem, w: np.ndarray) -> np.ndarray:
    """Return the full objective gradient at ``w``, including the ``rho*mu`` tilt."""
    data_grad = (problem.X.T @ (problem.X @ w)) / problem.t
    if problem.target_lr is not None:
        bar_lam, u_k, delta_k = problem.target_lr
        tgt_w = bar_lam * w + u_k @ (delta_k * (u_k.T @ w))
        grad = 2.0 * ((1 - problem.alpha) * data_grad + problem.alpha * tgt_w)
    elif problem.target is not None:
        grad = 2.0 * ((1 - problem.alpha) * data_grad + problem.alpha * problem.target @ w)
    else:
        grad = 2.0 * data_grad
    if problem.rho != 0.0 and problem.mu is not None:
        grad = grad - problem.rho * problem.mu
    result: np.ndarray = grad
    return result


def _primal_drop(w_a: np.ndarray, asset_active: np.ndarray, tol: float) -> bool:
    """Drop negative-weight assets from the active set in-place; return True if any dropped."""
    if not np.any(w_a < -tol):
        return False
    idx = np.where(asset_active)[0]
    strong = w_a < -10 * tol
    if np.any(strong):
        asset_active[idx[strong]] = False
    else:
        asset_active[idx[np.argmin(w_a)]] = False
    return True


def _dual_add(problem: _MinVarProblem, grad: np.ndarray, asset_active: np.ndarray, tol: float) -> int:
    """Return the index of an excluded asset violating the KKT dual condition, or -1 if none.

    The multiplier is estimated from the active gradient: for the budget the
    stationary ``lambda`` is a location estimate of ``g_a`` (median for
    robustness on larger sets); for a general balance system it is the
    least-squares solution of ``B_a^T lambda = g_a``.  The bound multiplier
    estimate is then ``nu = grad - B^T lambda``, which must be non-negative on
    excluded assets at the optimum.
    """
    excluded = ~asset_active
    if not excluded.any():
        return -1
    g_a = grad[asset_active]
    if problem.B is None:
        lambda_ = np.median(g_a) if g_a.size > 5 else g_a.mean()
        nu = grad - lambda_
    else:
        b_a = problem.B[:, asset_active]
        lam, *_ = np.linalg.lstsq(b_a.T, g_a, rcond=None)
        nu = grad - problem.B.T @ lam
    idx_ex = np.where(excluded)[0]
    j = idx_ex[np.argmin(nu[excluded])]
    return int(j) if nu[j] < -tol else -1


def run_active_set(
    problem: _MinVarProblem,
    solve_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, int, int]:
    """Run the primal-dual active-set loop enforcing ``w >= 0``.

    Calls ``solve_fn(active_mask)`` repeatedly.  The *primal step* drops assets
    with negative weights; the *dual step* re-adds any excluded asset whose KKT
    gradient condition is violated.  Terminates when both conditions hold
    simultaneously, which together with stationarity is sufficient for global
    optimality.  Returns ``(w, outer_steps, total_inner_iters)``.
    """
    n = problem.n
    asset_active = np.ones(n, dtype=bool)
    total_inner_iters = 0
    outer_steps = 0
    prev_active = None
    w = np.zeros(n)

    for _ in range(max_iter):
        if prev_active is not None and np.array_equal(prev_active, asset_active):
            break  # pragma: no cover - structurally unreachable safety guard
        prev_active = asset_active.copy()

        w_a, step_iters = solve_fn(asset_active)
        outer_steps += 1
        total_inner_iters += step_iters

        if _primal_drop(w_a, asset_active, tol):
            continue

        w = np.zeros(n)
        w[asset_active] = w_a

        j = _dual_add(problem, compute_gradient(problem, w), asset_active, tol)
        if j < 0:
            break
        asset_active[j] = True

    return w, outer_steps, total_inner_iters
