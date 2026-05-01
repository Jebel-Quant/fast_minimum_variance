"""KKT system construction for the minimum variance and Markowitz portfolio."""

import numpy as np

from .api import API


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

    def fn(active):
        """Solve the KKT system for the current active set."""
        # Pin active inequalities as equalities by appending their columns to A.
        # When active is empty, hstack returns A unchanged (C[:,active] is (n, 0)).
        K, rhs = api.kkt(active=active)  # noqa: N806
        return np.linalg.solve(K, rhs)[: api.n], 1

    w, iters = api.constraint_active_set(fn)
    w = np.maximum(w, 0)
    w /= w.sum()
    return w, iters
