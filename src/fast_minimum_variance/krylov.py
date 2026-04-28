"""Krylov subspace solvers for the minimum variance portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


def minvar_minres(R):  # noqa: N803
    """Solve the minimum variance portfolio via MINRES with active-set method.

    Applies the active-set method, dropping assets with negative weights, and
    solves the KKT saddle-point system at each iteration using MINRES. The KKT
    matrix is applied as a LinearOperator — no explicit R^T R or (N+1)x(N+1)
    matrix is ever formed. The matvec for x = [v; mu] is::

        out[:n_a] = 2 R^T (R v) + mu * 1   # two O(T*N) passes
        out[n_a]  = 1^T v                   # O(N)

    Args:
        R: Return matrix of shape (T, N).

    Returns:
        Weight vector of shape (N,) summing to 1 with all non-negative entries.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w = minvar_minres(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    while True:
        r_a = R[:, active]
        n_a = r_a.shape[1]

        def _matvec(x, ra=r_a, na=n_a):
            """Apply KKT operator [[2R^TR, 1],[1^T, 0]] to x."""
            out = np.empty(na + 1)
            out[:na] = 2.0 * (ra.T @ (ra @ x[:na])) + x[na]
            out[na] = x[:na].sum()
            return out

        b = np.zeros(n_a + 1)
        b[n_a] = 1.0
        kkt = LinearOperator(shape=(n_a + 1, n_a + 1), matvec=_matvec)  # type: ignore[call-arg]
        sol, _ = minres(kkt, b)
        w_a = sol[:n_a]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w


def minvar_cg(R):  # noqa: N803
    """Solve the minimum variance portfolio via CG in the constraint-reduced space.

    Projects the problem onto the constraint-satisfying subspace using a QR
    basis ``P`` of the null space of the budget constraint, then applies CG to
    the reduced positive-definite system ``P^T R^T R P``. An active-set loop
    drops assets with negative weights until feasibility is reached.

    Args:
        R: Return matrix of shape (T, N).

    Returns:
        Weight vector of shape (N,) summing to 1 with all non-negative entries.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance.random import make_returns
        >>> R = make_returns(100, 5, seed=0)
        >>> w = minvar_cg(R)
        >>> w.shape
        (5,)
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
    """
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    while True:
        R_a = R[:, active]  # noqa: N806
        n_a = R_a.shape[1]
        P = np.linalg.qr(np.ones((n_a, 1)), mode="complete")[0][:, 1:]  # noqa: N806
        w0 = np.ones(n_a) / n_a
        r0 = R_a @ w0

        def _matvec(v, P=P, R_a=R_a):  # noqa: N803
            """Apply the reduced Hessian P^T R^T R P to v."""
            return P.T @ (R_a.T @ (R_a @ (P @ v)))

        op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)  # type: ignore[call-arg]
        rhs = -(P.T @ (R_a.T @ r0))
        v, _ = cg(op, rhs)
        w_a = w0 + P @ v
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w
