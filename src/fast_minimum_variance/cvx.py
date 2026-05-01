"""CVXPY reference solver — convenience wrapper for Problem.solve_cvxpy."""

from .api import Problem, clip_and_renormalize

__all__ = ["solve_cvxpy"]

# clip_and_renormalize is re-exported for any callers that import it from here.
__all__ += ["clip_and_renormalize"]


def solve_cvxpy(problem: Problem, *, project: bool = True):
    """Solve the mean-variance portfolio via CVXPY (reference interior-point solver).

    Convenience wrapper for :meth:`Problem.solve_cvxpy`.  Requires the
    ``convex`` extra: ``pip install fast-minimum-variance[convex]``.

    Args:
        problem: Fully specified :class:`Problem` instance.
        project: Passed through to :meth:`Problem.solve_cvxpy`.

    Returns:
        Tuple (w, n_iters).

    Examples:
        >>> from fast_minimum_variance.api import Problem
        >>> from fast_minimum_variance.cvx import solve_cvxpy
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_cvxpy(Problem(X))
        >>> float(round(w.sum(), 6))
        1.0
    """
    return problem.solve_cvxpy(project=project)
