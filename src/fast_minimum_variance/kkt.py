"""KKT system construction for the minimum variance portfolio."""

import numpy as np


def build_kkt(R):  # noqa: N803
    """Build the KKT system matrix and RHS for the minimum variance problem.

    Constructs the (N+1) x (N+1) indefinite KKT system for::

        min  w^T (R^T R) w
        s.t. sum(w) = 1

    The system has the form::

        [ 2 R^T R   1 ] [ w ]   [ 0 ]
        [ 1^T       0 ] [ λ ] = [ 1 ]

    Args:
        R: Return matrix of shape (T, N).

    Returns:
        Tuple (A, b) where A is the (N+1) x (N+1) KKT matrix and b is the
        (N+1,) right-hand side vector.

    Examples:
        >>> import numpy as np
        >>> R = np.eye(3)
        >>> A, b = build_kkt(R)
        >>> A.shape
        (4, 4)
        >>> b
        array([0., 0., 0., 1.])
    """
    n_a = R.shape[1]
    A = np.zeros((n_a + 1, n_a + 1))  # noqa: N806
    A[:n_a, :n_a] = 2 * R.T @ R
    A[:n_a, n_a] = 1
    A[n_a, :n_a] = 1
    b = np.zeros(n_a + 1)
    b[n_a] = 1
    return A, b


def minvar_kkt(R):  # noqa: N803
    """Solve the minimum variance portfolio via the KKT system with active-set method.

    Iteratively drops assets with negative weights until all remaining weights
    are non-negative, solving the KKT system exactly at each iteration via
    ``numpy.linalg.solve``.

    Args:
        R: Return matrix of shape (T, N).

    Returns:
        Weight vector of shape (N,) summing to 1 with all non-negative entries.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w = minvar_kkt(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 10))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    while True:
        A, b = build_kkt(R[:, active])  # noqa: N806
        sol = np.linalg.solve(A, b)
        w_a = sol[: active.sum()]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    return w
