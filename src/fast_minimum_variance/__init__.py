"""fast_minimum_variance — fast solvers for the minimum-variance portfolio."""

import numpy as np

from .minvar_problem import _MinVarProblem


def Problem(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    target: np.ndarray | None = None,
    B: np.ndarray | None = None,  # noqa: N803
    c: np.ndarray | None = None,
    alpha: float = 0.0,
    rho: float = 0.0,
    mu: np.ndarray | None = None,
    target_lr: tuple[float, np.ndarray, np.ndarray] | None = None,
) -> _MinVarProblem:
    """Create a long-only minimum-variance portfolio optimisation problem.

    Returns a :class:`_MinVarProblem` (shrinking active-set) for the long-only
    minimum-variance problem, optionally with a balance system ``(B, c)`` in
    place of the default budget constraint.

    Args:
        X:      Returns matrix of shape ``(T, N)``.
        target: Optional ``(N, N)`` regularisation matrix; when supplied the
                shrinkage term ``alpha * ||target @ w||^2`` is added to the
                objective.  ``None`` disables shrinkage entirely.
        B:      Balance system ``(p, N)`` for the fast shrinking active-set
                path: ``B w = c`` replaces the budget ``1^T w = 1``.  ``B``
                must have full row rank on every active set the loop visits.
        c:      Balance RHS ``(p,)``; required together with ``B``.
        alpha:     Shrinkage intensity; only active when ``target`` is provided.
        rho:       Return tilt strength (Markowitz mean-variance).
        mu:        Expected returns vector ``(N,)``; required when ``rho != 0``.
        target_lr: Low-rank factored target ``(bar_lam, U_k, delta_k)`` for
                   RMT eigenvalue-cleaning; replaces ``target`` in the CG matvec.

    Returns:
        A solver instance with a ``solve_cg()`` method.

    Examples:
        >>> import numpy as np
        >>> X = np.random.default_rng(42).standard_normal((500, 20))
        >>> w, *_ = Problem(X).solve_cg()
        >>> float(round(w.sum(), 8))
        1.0
        >>> bool((w >= 0).all())
        True

        A two-sleeve balance system — each half of the universe holds half
        of the budget:

        >>> B = np.zeros((2, 20)); B[0, :10] = 1.0; B[1, 10:] = 1.0
        >>> w, *_ = Problem(X, B=B, c=np.array([0.5, 0.5])).solve_cg()
        >>> [float(round(s, 8)) for s in B @ w]
        [0.5, 0.5]
        >>> bool((w >= -1e-6).all())
        True
    """
    return _MinVarProblem(X, target=target, alpha=alpha, rho=rho, mu=mu, target_lr=target_lr, B=B, c=c)


__all__ = ["Problem"]
