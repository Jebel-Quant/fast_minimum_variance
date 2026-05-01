"""KKT system construction for the minimum variance and Markowitz portfolio."""

import numpy as np

from .api import API, clip_and_renormalize


def solve_kkt(api: API, *, project: bool = True):
    """Solve the general mean-variance portfolio via the KKT system with active-set method.

    Iteratively promotes violated inequality constraints to equalities until
    all inactive constraints are satisfied, solving the KKT system exactly at
    each iteration via ``numpy.linalg.solve``.

    Args:
        api:     API dataclass holding X, A, b, C, d, rho, mu.
        project: If True (default), clip weights to non-negative and renormalize
                 to sum to one after solving.  Only correct for the default
                 long-only minimum-variance problem; set to False when using
                 custom constraints.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the number of active-set steps taken.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance.api import API
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
    if project:
        w = clip_and_renormalize(w)
    return w, iters
