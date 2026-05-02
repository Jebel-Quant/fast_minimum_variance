"""fast_minimum_variance — fast solvers for the minimum-variance portfolio."""

import numpy as np

from .minvar_problem import _MinVarProblem
from .problem import _Problem

_UNSET = object()


def Problem(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    A=_UNSET,  # noqa: N803
    b=_UNSET,
    C=_UNSET,  # noqa: N803
    d=_UNSET,
    alpha: float = 0.0,
    rho: float = 0.0,
    mu=None,
):
    """Create a portfolio optimisation problem.

    Returns a :class:`_MinVarProblem` (shrinking active-set) when no custom
    constraints are supplied, or a :class:`_Problem` (growing active-set) when
    any of ``A``, ``b``, ``C``, ``d`` are provided.

    Args:
        X:     Returns matrix of shape ``(T, N)``.
        A:     Equality constraint matrix ``(N, m)``: ``A^T w = b``.
        b:     Equality RHS ``(m,)``.
        C:     Inequality constraint matrix ``(N, p)``: ``C^T w <= d``.
        d:     Inequality RHS ``(p,)``.
        alpha: Ledoit-Wolf shrinkage intensity; ridge = ``alpha * ||X||_F^2 / N``.
        rho:   Return tilt strength (Markowitz mean-variance).
        mu:    Expected returns vector ``(N,)``; required when ``rho != 0``.

    Returns:
        A solver instance with ``solve_kkt()``, ``solve_minres()``,
        ``solve_cg()``, and ``solve_cvxpy()`` methods, each returning
        ``(w, n_iters)``.

    Examples:
        >>> import numpy as np
        >>> X = np.random.default_rng(42).standard_normal((500, 20))
        >>> w, _ = Problem(X).solve_kkt()
        >>> float(round(w.sum(), 8))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    if all(v is _UNSET for v in (A, b, C, d)):
        return _MinVarProblem(X, alpha=alpha, rho=rho, mu=mu)
    return _Problem(
        X,
        A=None if A is _UNSET else A,
        b=None if b is _UNSET else b,
        C=None if C is _UNSET else C,
        d=None if d is _UNSET else d,
        alpha=alpha,
        rho=rho,
        mu=mu,
    )


__all__ = ["Problem"]
