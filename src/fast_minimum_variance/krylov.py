"""Krylov subspace solvers for the minimum variance and Markowitz portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres

from ._util import API
from .active import constraint_active_set


def solve_minres(api: API):
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
        api: API dataclass holding X, A, b, C, d, rho, mu, gamma.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of MINRES iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance._util import API
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
    X, A, b, C, d = api.X, api.A, api.b, api.C, api.d  # noqa: N806
    n, rho, mu, gamma = api.n, api.rho, api.mu, api.gamma

    def _solve(active):
        # Build A_ext = [A, C[:, active]]: the budget constraint columns plus one
        # column per currently-pinned inequality.  When active is all-False on the
        # first iteration, hstack returns A unchanged (C[:,active] has 0 columns).
        aa = np.hstack([A, C[:, active]])
        na = n  # primal block size — all N assets are present
        ma = aa.shape[1]  # dual block size — m budget + n_pinned inequality multipliers

        # The saddle-point matvec applies the (N+ma) x (N+ma) KKT operator:
        #
        #   [ 2(X^T X + gamma I)   aa ] [ x[:na] ]
        #   [ aa^T                  0  ] [ x[na:] ]
        #
        # without forming X^T X explicitly (T x N instead of N x N storage).
        def _matvec(x, xx=X, a=aa, n_=na, m_=ma, gam=gamma):
            """Apply the KKT saddle-point operator to vector x."""
            out = np.empty(n_ + m_)
            # Primal row: Hessian term 2(X^T X + gamma I) w + aa lambda
            out[:n_] = 2.0 * (xx.T @ (xx @ x[:n_]) + gam * x[:n_]) + a @ x[n_:]
            # Dual row: primal feasibility aa^T w = b_ext
            out[n_:] = a.T @ x[:n_]
            return out

        # RHS: primal block is the return term (zero for pure min-var);
        # dual block is [b, d[active]] — budget RHS followed by the pinned
        # inequality bounds (zero for the default long-only constraint).
        rhs = np.zeros(na + ma)
        if rho != 0.0 and mu is not None:
            rhs[:na] = rho * mu
        rhs[na:] = np.concatenate([b, d[active]])

        # MINRES handles symmetric indefinite systems; the KKT saddle-point
        # matrix is symmetric but not positive definite (it has negative
        # eigenvalues from the zero bottom-right block).
        kkt = LinearOperator(shape=(na + ma, na + ma), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        # Return only the primal part w; discard the dual multipliers lambda.
        return sol[:na], iters[0]

    # MINRES is iterative and may not satisfy the budget constraint exactly;
    # clip and renormalise to enforce non-negativity and budget feasibility.
    w, iters = constraint_active_set(C, d, _solve)
    w = np.maximum(w, 0)
    w /= w.sum()
    return w, iters


def solve_cg(api: API):
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

    See ``solve_minres`` for the Ledoit-Wolf shrinkage recipe.

    Args:
        api: API dataclass holding X, A, b, C, d, rho, mu, gamma.

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of CG iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> from fast_minimum_variance._util import API
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
    X, A, b, C, d = api.X, api.A, api.b, api.C, api.d  # noqa: N806
    n, rho, mu, gamma = api.n, api.rho, api.mu, api.gamma

    def _solve(active):
        # Extend equality constraints with pinned inequalities.
        aa = np.hstack([A, C[:, active]])
        m_ext = aa.shape[1]
        n_free = n - m_ext
        b_ext = np.concatenate([b, d[active]])

        # When the constraints fully determine w (no free directions), solve directly.
        if n_free <= 0:
            return np.linalg.lstsq(aa.T, b_ext, rcond=None)[0], 0

        # QR of aa gives orthonormal columns: first m_ext span range(aa),
        # remaining n_free span null(aa^T).  P is the null-space basis.
        Q, _ = np.linalg.qr(aa, mode="complete")  # noqa: N806
        P = Q[:, m_ext:]  # noqa: N806
        w0 = np.linalg.lstsq(aa.T, b_ext, rcond=None)[0]

        # Half-gradient of objective at w0.  Since P^T w0 = 0, the gamma
        # term in the null-space gradient vanishes (P^T (gamma w0) = 0).
        g0 = X.T @ (X @ w0) + gamma * w0
        if rho != 0.0 and mu is not None:
            g0 = g0 - (rho / 2.0) * mu

        rhs = -(P.T @ g0)

        # CG operates on the reduced (n_free x n_free) positive-definite system.
        # The operator is P^T (X^T X + gamma I) P, applied without forming X^T X.
        def _matvec(y, pp=P, gam=gamma):
            """Apply the reduced positive-definite operator to vector y."""
            pv = pp @ y
            return pp.T @ (X.T @ (X @ pv)) + gam * y

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
        return w0 + P @ sol, iters[0]

    w, iters = constraint_active_set(C, d, _solve)
    w = np.maximum(w, 0)
    w /= w.sum()
    return w, iters
