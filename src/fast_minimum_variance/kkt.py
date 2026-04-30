"""KKT system construction for the minimum variance and Markowitz portfolio."""

import numpy as np


def build_kkt(X, A=None, b=None, rho=0.0, mu=None):  # noqa: N803
    """Build the KKT system matrix and RHS for the general mean-variance problem.

    Constructs the (N+m) x (N+m) indefinite saddle-point system for::

        min  ||X w||_2^2 - rho * mu @ w
        s.t. A.T @ w == b

    The system has the form::

        [ 2 X^T X   A ] [ w ]   [ rho * mu ]
        [ A^T       0 ] [ λ ] = [ b        ]

    Defaults (A = ones((N,1)), b = [1]) recover the minimum variance KKT
    system of the companion paper.

    Args:
        X:   Return matrix of shape (T, N).
        A:   Equality constraint matrix of shape (N, m).
             Defaults to ones((N, 1)) (budget constraint).
        b:   Equality RHS of shape (m,). Defaults to [1.0].
        rho: Risk-aversion parameter (>= 0). Default 0.
        mu:  Expected return vector of shape (N,). Required when rho > 0.

    Returns:
        Tuple (K, rhs) where K is the (N+m) x (N+m) KKT matrix and rhs is
        the (N+m,) right-hand side vector.

    Examples:
        >>> import numpy as np
        >>> X = np.eye(3)
        >>> K, rhs = build_kkt(X)
        >>> K.shape
        (4, 4)
        >>> rhs
        array([0., 0., 0., 1.])
    """
    n = X.shape[1]

    if A is None:
        A = np.ones((n, 1))  # noqa: N806
    if b is None:
        b = np.ones(1)

    m = A.shape[1]
    K = np.zeros((n + m, n + m))  # noqa: N806
    K[:n, :n] = 2 * X.T @ X
    K[:n, n:] = A
    K[n:, :n] = A.T

    rhs = np.zeros(n + m)
    if rho != 0.0 and mu is not None:
        rhs[:n] = rho * mu
    rhs[n:] = b

    return K, rhs


def solve_kkt(X, A=None, b=None, C=None, d=None, rho=0.0, mu=None):  # noqa: N803
    """Solve the general mean-variance portfolio via the KKT system with active-set method.

    Iteratively promotes violated inequality constraints to equalities until
    all inactive constraints are satisfied, solving the KKT system exactly at
    each iteration via ``numpy.linalg.solve``.

    Args:
        X:   Return matrix of shape (T, N).
        A:   Equality constraint matrix of shape (N, m).
             Defaults to ones((N, 1)) (budget constraint).
        b:   Equality RHS of shape (m,). Defaults to [1.0].
        C:   Inequality constraint matrix of shape (N, p) for C.T @ w <= d.
             Defaults to -eye(N) (long-only constraint).
        d:   Inequality RHS of shape (p,). Defaults to zeros(N).
        rho: Risk-aversion parameter (>= 0). Default 0.
        mu:  Expected return vector of shape (N,). Required when rho > 0.

    Returns:
        Weight vector of shape (N,).

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w = solve_kkt(X)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 10))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    n = X.shape[1]

    if A is None:
        A = np.ones((n, 1))  # noqa: N806
    if b is None:
        b = np.ones(1)
    if C is None:
        C = -np.eye(n)  # noqa: N806
    if d is None:
        d = np.zeros(n)

    p = d.shape[0]
    active = np.zeros(p, dtype=bool)

    while True:
        if active.any():
            A_ext = np.hstack([A, C[:, active]])  # noqa: N806
            b_ext = np.concatenate([b, d[active]])
        else:
            A_ext, b_ext = A, b  # noqa: N806

        K, rhs = build_kkt(X, A_ext, b_ext, rho=rho, mu=mu)  # noqa: N806
        sol = np.linalg.solve(K, rhs)
        w = sol[:n]

        inactive = ~active
        if not inactive.any():
            break
        violations = C[:, inactive].T @ w - d[inactive]
        if np.all(violations <= 1e-10):
            break
        active[np.where(inactive)[0][violations > 1e-10]] = True

    return w
