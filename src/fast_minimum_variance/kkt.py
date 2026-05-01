"""KKT system construction for the minimum variance and Markowitz portfolio."""

import numpy as np

from ._util import API
from .active import constraint_active_set


def build_kkt(X, A=None, b=None, rho=0.0, mu=None, gamma=0.0):  # noqa: N803
    """Build the KKT system matrix and RHS for the general mean-variance problem.

    Constructs the (N+m) x (N+m) indefinite saddle-point system for::

        min  ||X w||_2^2 + gamma ||w||_2^2 - rho * mu @ w
        s.t. A.T @ w == b

    The system has the form::

        [ 2(X^T X + gamma I)   A ] [ w ]   [ rho * mu ]
        [ A^T                  0 ] [ λ ] = [ b        ]

    Defaults (A = ones((N,1)), b = [1], gamma = 0) recover the
    minimum variance KKT system of the companion paper.

    Args:
        X:     Return matrix of shape (T, N).
        A:     Equality constraint matrix of shape (N, m).
               Defaults to ones((N, 1)) (budget constraint).
        b:     Equality RHS of shape (m,). Defaults to [1.0].
        rho:   Risk-aversion parameter (>= 0). Default 0.
        mu:    Expected return vector of shape (N,). Required when rho > 0.
        gamma: Diagonal regularisation added to the Hessian (default 0.0).

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

    # The KKT matrix is an (N+m) x (N+m) indefinite saddle-point system.
    # The top-left block is the Hessian of the objective (2(X^T X + gamma I)); the
    # off-diagonal blocks enforce the equality constraints via Lagrange
    # multipliers; the bottom-right block is zero because there is no
    # quadratic term in the dual variable.
    K = np.zeros((n + m, n + m))  # noqa: N806
    K[:n, :n] = 2 * (X.T @ X + gamma * np.eye(n))
    K[:n, n:] = A
    K[n:, :n] = A.T

    # The primal block of the RHS is the return term (zero for pure min-var);
    # the dual block is the equality RHS b (e.g. [1] for the budget constraint).
    rhs = np.zeros(n + m)
    if rho != 0.0 and mu is not None:
        rhs[:n] = rho * mu
    rhs[n:] = b

    return K, rhs


def solve_kkt(api: API):
    """Solve the general mean-variance portfolio via the KKT system with active-set method.

    Iteratively promotes violated inequality constraints to equalities until
    all inactive constraints are satisfied, solving the KKT system exactly at
    each iteration via ``numpy.linalg.solve``.

    Args:
        api: API dataclass holding X, A, b, C, d, rho, mu.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the number of active-set steps taken.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance._util import API
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_kkt(API(X))
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 10))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    n, A, b, C, d = api.n, api.A, api.b, api.C, api.d  # noqa: N806

    def fn(active):
        """Solve the KKT system for the current active set."""
        # Pin active inequalities as equalities by appending their columns to A.
        # When active is empty, hstack returns A unchanged (C[:,active] is (n, 0)).
        K, rhs = build_kkt(api.X, np.hstack([A, C[:, active]]), np.concatenate([b, d[active]]), rho=api.rho, mu=api.mu)  # noqa: N806
        return np.linalg.solve(K, rhs)[:n], 1

    w, iters = constraint_active_set(C, d, fn)
    w = np.maximum(w, 0)
    w /= w.sum()
    return w, iters
