"""Krylov subspace solvers for the minimum variance and Markowitz portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


def solve_minres(X, A=None, b=None, C=None, d=None, rho=0.0, mu=None, c=1.0, gamma=0.0):  # noqa: N803
    """Solve the general mean-variance portfolio via MINRES with active-set method.

    Applies the active-set method for the long-only constraint, dropping assets
    with negative weights and resolving.  At each outer iteration the KKT
    saddle-point system for the active assets

        [ 2(c X_a^T X_a + gamma I)   A_a ] [ w_a ]   [ rho * mu_a ]
        [ A_a^T                       0   ] [ λ   ] = [ b          ]

    is solved matrix-free via MINRES, where ``X_a = X[:, active]`` and
    ``A_a = A[active, :]``.  No explicit matrix is ever formed.

    With the defaults (``A = ones``, ``b = [1]``, ``C = -I``, ``d = 0``) this
    recovers the long-only minimum variance solver of the companion paper.
    The ``C`` and ``d`` parameters are accepted for interface symmetry; the
    active-set loop is optimised for the default long-only constraint.

    To apply Ledoit-Wolf shrinkage without materialising a stacked return
    matrix, compute::

        T, N    = X.shape
        frob_sq = (X * X).sum()
        alpha   = N / (N + T)
        c       = 1.0 - alpha
        gamma   = frob_sq / (N + T)

    and pass ``c`` and ``gamma`` explicitly.

    Args:
        X:     Return matrix of shape (T, N).
        A:     Equality constraint matrix of shape (N, m).
               Defaults to ones((N, 1)) (budget constraint).
        b:     Equality RHS of shape (m,). Defaults to [1.0].
        C:     Inequality constraint matrix of shape (N, p) for C.T @ w <= d.
               Defaults to -eye(N) (long-only constraint).
        d:     Inequality RHS of shape (p,). Defaults to zeros(N).
        rho:   Risk-aversion parameter (>= 0). Default 0.
        mu:    Expected return vector of shape (N,). Required when rho > 0.
        c:     Scaling factor for X^T X (default 1.0).
        gamma: Diagonal regularisation added to the (N, N) block (default 0.0).

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of MINRES iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_minres(X)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
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

    m = A.shape[1]
    active = np.ones(n, dtype=bool)
    total_iters = 0

    while True:
        x_a = X[:, active]
        a_a = A[active, :]
        n_a = int(active.sum())

        def _matvec(x, ra=x_a, aa=a_a, na=n_a, ma=m, cc=c, gam=gamma):
            """Apply the KKT saddle-point operator to vector x."""
            out = np.empty(na + ma)
            out[:na] = 2.0 * (cc * (ra.T @ (ra @ x[:na])) + gam * x[:na]) + aa @ x[na:]
            out[na:] = aa.T @ x[:na]
            return out

        rhs = np.zeros(n_a + m)
        if rho != 0.0 and mu is not None:
            rhs[:n_a] = rho * mu[active]
        rhs[n_a:] = b

        kkt = LinearOperator(shape=(n_a + m, n_a + m), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))  # noqa: B023
        total_iters += iters[0]
        w_a = sol[:n_a]

        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False

    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w, total_iters


def solve_cg(X, A=None, b=None, C=None, d=None, rho=0.0, mu=None, c=1.0, gamma=0.0):  # noqa: N803
    """Solve the general mean-variance portfolio via CG in the constraint-reduced space.

    At each active-set iteration the equality-constrained subproblem for the
    active assets is solved by projecting onto the null space of ``A_a^T``
    via QR factorisation of ``A_a = A[active, :]``, then applying CG to the
    reduced positive-definite system

        (c (X_a P)^T (X_a P) + gamma I) v = -P^T (c X_a^T X_a w0 + gamma w0 - (rho/2) mu_a)

    where ``P`` is an orthonormal null-space basis for ``A_a^T`` and ``w0`` is
    the minimum-norm particular solution of ``A_a^T w = b``.  The active
    asset weights are recovered as ``w_a = w0 + P v``.

    For the default ``A = ones`` this QR approach is equivalent to the
    Householder trick used in the companion paper; general ``A`` is handled
    identically.  The ``C`` and ``d`` parameters are accepted for interface
    symmetry; the active-set loop is optimised for the default long-only
    constraint.  See ``minvar_minres`` for the Ledoit-Wolf shrinkage recipe.

    Args:
        X:     Return matrix of shape (T, N).
        A:     Equality constraint matrix of shape (N, m).
               Defaults to ones((N, 1)) (budget constraint).
        b:     Equality RHS of shape (m,). Defaults to [1.0].
        C:     Inequality constraint matrix of shape (N, p) for C.T @ w <= d.
               Defaults to -eye(N) (long-only constraint).
        d:     Inequality RHS of shape (p,). Defaults to zeros(N).
        rho:   Risk-aversion parameter (>= 0). Default 0.
        mu:    Expected return vector of shape (N,). Required when rho > 0.
        c:     Scaling factor for X^T X (default 1.0).
        gamma: Diagonal regularisation added to the objective (default 0.0).

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) and
        n_iters is the total number of CG iterations across all active-set
        steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> X = make_returns(100, 5, seed=0)
        >>> w, iters = solve_cg(X)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
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

    m = A.shape[1]
    active = np.ones(n, dtype=bool)
    total_iters = 0

    while True:
        x_a = X[:, active]
        a_a = A[active, :]
        n_a = int(active.sum())
        n_free = n_a - m

        if n_free == 0:
            w_a = np.linalg.lstsq(a_a.T, b, rcond=None)[0]
            break

        # Null-space basis P (n_a x n_free) and particular solution w0 via QR
        Q, _ = np.linalg.qr(a_a, mode="complete")  # noqa: N806
        P = Q[:, m:]  # noqa: N806
        w0 = np.linalg.lstsq(a_a.T, b, rcond=None)[0]

        # Half-gradient of scaled objective at w0; P^T w0 = 0 so gamma term vanishes
        g0 = c * (x_a.T @ (x_a @ w0)) + gamma * w0
        if rho != 0.0 and mu is not None:
            g0 = g0 - (rho / 2.0) * mu[active]

        rhs = -(P.T @ g0)

        def _matvec(y, ra=x_a, pp=P, cc=c, gam=gamma):
            """Apply the reduced positive-definite operator to vector y."""
            pv = pp @ y
            return cc * (pp.T @ (ra.T @ (ra @ pv))) + gam * y

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))  # noqa: B023
        total_iters += iters[0]
        w_a = w0 + P @ sol

        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False

    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w, total_iters
