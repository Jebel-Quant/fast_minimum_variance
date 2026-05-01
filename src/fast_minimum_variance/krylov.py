"""Krylov subspace solvers for the minimum variance and Markowitz portfolio."""

from scipy.sparse.linalg import cg, minres

from .api import API, clip_and_renormalize


def solve_minres(api: API, *, project: bool = True):
    """Solve the general mean-variance portfolio via MINRES with active-set method.

    Iteratively promotes violated inequality constraints to equalities.  At each
    outer iteration the KKT saddle-point system for all assets with the currently
    active constraints pinned as equalities

        [ 2(X^T X + gamma I)   A_ext ] [ w   ]   [ rho * mu ]
        [ A_ext^T               0    ] [ λ   ] = [ b_ext    ]

    is solved matrix-free via MINRES, where ``A_ext = [A, C[:, active]]`` and
    ``b_ext = [b, d[active]]``.  No explicit matrix is ever formed.

    With the defaults (``A = ones``, ``b = [1]``, ``C = -I``, ``d = 0``) this
    recovers the long-only minimum variance solver of the companion paper.

    To apply Ledoit-Wolf shrinkage, pre-scale the return matrix before calling::

        T, N     = X.shape
        frob_sq  = (X * X).sum()
        alpha    = N / (N + T)
        X_scaled = np.sqrt(1.0 - alpha) * X
        gamma    = frob_sq / (N + T)

    and pass ``X_scaled`` and ``gamma`` explicitly via the API dataclass.

    Args:
        api:     API dataclass holding X, A, b, C, d, rho, mu, gamma.
        project: If True (default), clip weights to non-negative and renormalize
                 to sum to one after solving.  Only correct for the default
                 long-only minimum-variance problem; set to False when using
                 custom constraints.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of MINRES iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance.api import API
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_minres(API(X))
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
        True
    """

    def _solve(active):
        """Solve the MINRES saddle-point system for the current active set."""
        # MINRES (not CG) is required here because the KKT saddle-point matrix
        # is symmetric but indefinite: the zero bottom-right block causes negative
        # eigenvalues, ruling out CG which requires positive definiteness.
        kkt, rhs = api.kkt_operator(active)
        # Use a mutable list to count iterations from inside the callback.
        # A plain int cannot be rebound in the enclosing scope via the callback;
        # mutating a list element sidesteps that restriction without `nonlocal`.
        iters = [0]
        sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        # The solution vector is [w; lambda].  Discard the dual part (lambda)
        # and return only the primal weights.
        return sol[: api.n], iters[0]

    w, iters = api.constraint_active_set(_solve)
    if project:
        w = clip_and_renormalize(w)
    return w, iters


def solve_cg(api: API, *, project: bool = True):
    """Solve the general mean-variance portfolio via CG in the constraint-reduced space.

    At each active-set iteration the equality-constrained subproblem (with
    currently active inequalities pinned as equalities) is solved by projecting
    onto the null space of ``A_ext^T`` via QR factorisation, then applying CG
    to the reduced positive-definite system

        ((X P)^T (X P) + gamma I) v = -P^T (X^T X w0 + gamma w0 - (rho/2) mu)

    where ``A_ext = [A, C[:, active]]``, ``P`` is an orthonormal null-space
    basis for ``A_ext^T``, and ``w0`` is the minimum-norm particular solution
    of ``A_ext^T w = b_ext``.  The full weight vector is recovered as
    ``w = w0 + P v``.

    The reduced operator ``P^T(X^T X + gamma I)P`` is SPD, so CG converges
    in at most ``n_free`` iterations and benefits from any spectral clustering
    induced by shrinkage (larger ``gamma`` compresses the eigenvalue spread).

    See ``solve_minres`` for the Ledoit-Wolf shrinkage recipe.

    Args:
        api:     API dataclass holding X, A, b, C, d, rho, mu, gamma.
        project: If True (default), clip weights to non-negative and renormalize
                 to sum to one after solving.  Only correct for the default
                 long-only minimum-variance problem; set to False when using
                 custom constraints.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of CG iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_cg(API(X))
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
        True
    """

    def _solve(active):
        """Solve the CG null-space subproblem for the current active set."""
        op, rhs, w0, P = api.null_space_operator(active)  # noqa: N806
        if op is None:
            # All free directions are pinned by the active constraints; the
            # constraints uniquely determine w = w0 with no further optimisation.
            return w0, 0
        # Same mutable-list trick as in solve_minres: the callback cannot rebind
        # a plain int from the enclosing scope.
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        # Reconstruct the full weight vector: w0 is the particular solution that
        # satisfies the constraints exactly, and P @ sol is the correction in the
        # constraint null space that minimises the objective.
        return w0 + P @ sol, iters[0]

    w, iters = api.constraint_active_set(_solve)
    if project:
        w = clip_and_renormalize(w)
    return w, iters
