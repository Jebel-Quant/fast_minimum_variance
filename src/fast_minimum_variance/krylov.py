"""Krylov subspace solvers for the minimum variance portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


def minvar_minres(R, c=1.0, gamma=0.0):  # noqa: N803
    """Solve the minimum variance portfolio via MINRES with active-set method.

    Applies the active-set method, dropping assets with negative weights, and
    solves the KKT saddle-point system at each iteration using MINRES. The KKT
    matrix is applied as a LinearOperator — no explicit matrix is ever formed.
    The matvec for x = [v; mu] is::

        out[:n_a] = 2 (c R^T(Rv) + gamma v) + mu * 1
        out[n_a]  = 1^T v

    With the defaults ``c=1, gamma=0`` this solves the standard sample-covariance
    problem. To apply dimension-based Ledoit-Wolf shrinkage without materialising
    a stacked return matrix, compute::

        T, N   = R.shape
        frob_sq = (R * R).sum()
        alpha  = N / (N + T)          # shrinkage intensity
        c      = 1.0 - alpha          # = T / (N + T)
        gamma  = frob_sq / (N + T)    # diagonal regularisation  (= alpha * frob_sq / N)

    and call ``minvar_minres(R, c=c, gamma=gamma)``.

    Args:
        R: Return matrix of shape (T, N).
        c: Scaling factor for R^T R (default 1.0).
        gamma: Diagonal regularisation added to the (1,1) block (default 0.0).

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) summing
        to 1 with all non-negative entries and n_iters is the total number of
        MINRES iterations across all active-set steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w, iters = minvar_minres(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
        True
    """
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    total_iters = 0
    while True:
        r_a = R[:, active]
        n_a = r_a.shape[1]

        def _matvec(x, ra=r_a, na=n_a, cc=c, gam=gamma):
            """Apply KKT operator [[2(c R^TR + gamma I), 1],[1^T, 0]] to x."""
            out = np.empty(na + 1)
            out[:na] = 2.0 * (cc * (ra.T @ (ra @ x[:na])) + gam * x[:na]) + x[na]
            out[na] = x[:na].sum()
            return out

        b = np.zeros(n_a + 1)
        b[n_a] = 1.0
        kkt = LinearOperator(shape=(n_a + 1, n_a + 1), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = minres(kkt, b, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))  # noqa: B023
        total_iters += iters[0]
        w_a = sol[:n_a]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w, total_iters


def minvar_cg(R, c=1.0, gamma=0.0):  # noqa: N803
    """Solve the minimum variance portfolio via CG in the constraint-reduced space.

    Projects the problem onto the constraint-satisfying subspace using an
    implicit Householder reflector as the null-space basis of the budget
    constraint, then applies CG to the reduced positive-definite system
    ``P^T (c R^T R + gamma I) P``. An active-set loop drops assets with
    negative weights until feasibility is reached.

    The Householder vector ``v = [1+sqrt(n_a), 1, ..., 1]`` with
    ``beta = 1/(n_a + sqrt(n_a))`` defines the reflector H = I - beta*v*v^T.
    Its last n_a-1 columns span the null space of 1^T and form the implicit
    basis P. Applying P or P^T costs O(n_a) instead of O(n_a^2) for the
    explicit QR matrix, and no O(n_a^2) matrix is ever formed. Because P has
    orthonormal columns, ``P^T (gamma I) P = gamma I``, so the gamma term
    adds only a scalar shift to each matvec.

    With the defaults ``c=1, gamma=0`` this solves the standard sample-covariance
    problem. See ``minvar_minres`` for the Ledoit-Wolf shrinkage recipe.

    Args:
        R: Return matrix of shape (T, N).
        c: Scaling factor for R^T R (default 1.0).
        gamma: Diagonal regularisation added to the objective (default 0.0).

    Returns:
        Tuple (w, n_iters) where w is the weight vector of shape (N,) summing
        to 1 with all non-negative entries and n_iters is the total number of
        CG iterations across all active-set steps.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w, iters = minvar_cg(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
        >>> iters > 0
        True
    """
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    total_iters = 0
    while True:
        r_a = R[:, active]
        n_a = r_a.shape[1]

        if n_a == 1:
            w_a = np.array([1.0])
            break

        # Implicit Householder basis for null(1^T): v = [1+sqrt(n_a), 1,...,1]
        sqrt_na = np.sqrt(float(n_a))
        beta = 1.0 / (n_a + sqrt_na)
        v0 = 1.0 + sqrt_na

        def _p_apply(y, b=beta, vv=v0, na=n_a):
            """Apply implicit P (n_a x n_a-1) to y: O(n_a)."""
            s = y.sum()
            out = np.empty(na)
            out[0] = -b * vv * s
            out[1:] = y - (b * s)
            return out

        def _pt_apply(u, b=beta, vv=v0):
            """Apply implicit P^T (n_a-1 x n_a) to u: O(n_a)."""
            s = vv * u[0] + u[1:].sum()
            return u[1:] - (b * s)

        w0 = np.ones(n_a) / n_a
        r0 = r_a @ w0

        def _matvec(y, ra=r_a, b=beta, vv=v0, na=n_a, cc=c, gam=gamma):
            """Apply P^T (c R^T R + gamma I) P to y via implicit Householder."""
            s = y.sum()
            pv = np.empty(na)
            pv[0] = -b * vv * s
            pv[1:] = y - (b * s)
            rpv = cc * (ra.T @ (ra @ pv)) + gam * pv
            sv = vv * rpv[0] + rpv[1:].sum()
            return rpv[1:] - (b * sv)

        g0 = c * (r_a.T @ r0) + gamma * w0
        rhs = -_pt_apply(g0)
        op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)  # type: ignore[call-arg]
        iters = [0]
        sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))  # noqa: B023
        total_iters += iters[0]
        w_a = w0 + _p_apply(sol)
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w, total_iters
