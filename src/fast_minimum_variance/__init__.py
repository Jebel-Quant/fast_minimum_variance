"""fast_minimum_variance — fast solvers for the minimum-variance portfolio."""

import numpy as np

from .minvar_problem import _MinVarProblem
from .problem import _Problem


def Problem(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    A: np.ndarray | None = None,  # noqa: N803
    b: np.ndarray | None = None,
    C: np.ndarray | None = None,  # noqa: N803
    d: np.ndarray | None = None,
    alpha: float = 0.0,
    rho: float = 0.0,
    mu: np.ndarray | None = None,
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
    if A is None and b is None and C is None and d is None:
        return _MinVarProblem(X, alpha=alpha, rho=rho, mu=mu)

    # number of assets
    n = X.shape[1]

    A = A if A is not None else np.ones((n, 0))  # noqa: N806
    b = b if b is not None else np.ones(1)
    C = C if C is not None else -np.eye(n)  # noqa: N806
    d = d if d is not None else np.zeros(n)

    return _Problem(X, A=A, b=b, C=C, d=d, alpha=alpha, rho=rho, mu=mu)


__all__ = ["Problem"]
