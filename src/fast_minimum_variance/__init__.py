"""fast_minimum_variance — fast solvers for the minimum-variance portfolio."""

import numpy as np

from .minvar_problem import _MinVarProblem
from .problem import _Problem


def Problem(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    target: np.ndarray | None = None,
    A: np.ndarray | None = None,  # noqa: N803
    b: np.ndarray | None = None,
    C: np.ndarray | None = None,  # noqa: N803
    d: np.ndarray | None = None,
    B: np.ndarray | None = None,  # noqa: N803
    c: np.ndarray | None = None,
    alpha: float = 0.0,
    rho: float = 0.0,
    mu: np.ndarray | None = None,
    target_lr: tuple[float, np.ndarray, np.ndarray] | None = None,
    pcg_lr: tuple[float, np.ndarray, np.ndarray] | None = None,
) -> _MinVarProblem | _Problem:
    """Create a portfolio optimisation problem.

    Returns a :class:`_MinVarProblem` (shrinking active-set) when no custom
    constraints — or only a balance system ``(B, c)`` — are supplied, or a
    :class:`_Problem` (growing active-set) when any of ``A``, ``b``, ``C``,
    ``d`` are provided.

    Args:
        X:      Returns matrix of shape ``(T, N)``.
        target: Optional ``(N, N)`` regularisation matrix; when supplied the
                shrinkage term ``alpha * ||target @ w||^2`` is added to the
                objective.  ``None`` disables shrinkage entirely.
        A:      Equality constraint matrix ``(N, m)``: ``A^T w = b``.
        b:      Equality RHS ``(m,)``.
        C:      Inequality constraint matrix ``(N, p)``: ``C^T w <= d``.
        d:      Inequality RHS ``(p,)``.
        B:      Balance system ``(p, N)`` for the fast shrinking active-set
                path: ``B w = c`` replaces the budget ``1^T w = 1``.  ``B``
                must have full row rank on every active set the loop visits.
                Cannot be combined with ``A``/``b``/``C``/``d``.
        c:      Balance RHS ``(p,)``; required together with ``B``.
        alpha:     Shrinkage intensity; only active when ``target`` is provided.
        rho:       Return tilt strength (Markowitz mean-variance).
        mu:        Expected returns vector ``(N,)``; required when ``rho != 0``.
        target_lr: Low-rank factored target ``(bar_lam, U_k, delta_k)`` for
                   RMT eigenvalue-cleaning; replaces ``target`` in the CG matvec.
        pcg_lr:    RMT preconditioner ``(bar_lam, U_k, delta_k)`` for
                   ``solve_pcg``; ignored unless PCG is invoked.

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

        A two-sleeve balance system — each half of the universe holds half
        of the budget:

        >>> B = np.zeros((2, 20)); B[0, :10] = 1.0; B[1, 10:] = 1.0
        >>> w, _ = Problem(X, B=B, c=np.array([0.5, 0.5])).solve_kkt()
        >>> [float(round(s, 8)) for s in B @ w]
        [0.5, 0.5]
        >>> bool((w >= -1e-6).all())
        True
    """
    if A is None and b is None and C is None and d is None:
        return _MinVarProblem(
            X, target=target, alpha=alpha, rho=rho, mu=mu, target_lr=target_lr, pcg_lr=pcg_lr, B=B, c=c
        )

    if B is not None or c is not None:
        raise ValueError("B/c (balance system) cannot be combined with A/b/C/d constraints")  # noqa: TRY003

    # number of assets
    n = X.shape[1]

    A = A if A is not None else np.ones((n, 0))  # noqa: N806
    b = b if b is not None else np.ones(1)
    C = C if C is not None else -np.eye(n)  # noqa: N806
    d = d if d is not None else np.zeros(n)

    return _Problem(X, target=target, A=A, b=b, C=C, d=d, alpha=alpha, rho=rho, mu=mu)


from .data import simulate_equity_returns  # noqa: E402

__all__ = ["Problem", "simulate_equity_returns"]
