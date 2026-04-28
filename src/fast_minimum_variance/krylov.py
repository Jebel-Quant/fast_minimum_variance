"""Krylov subspace solvers for the minimum variance portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


def minvar_minres(R):  # noqa: N803
    """Solve the minimum variance portfolio via preconditioned MINRES.

    Applies the active-set method, dropping assets with negative weights, and
    solves the KKT saddle-point system at each iteration using MINRES. Both
    the KKT operator and the preconditioner are applied as LinearOperators —
    no explicit R^T R or (N+1)x(N+1) matrix is ever formed.

    The preconditioner is the Murphy-Golub-Wathen (2000) block diagonal::

        M = diag(2 R^T R + delta*I,  |S|)

    where S = -1^T (2R^TR + delta*I)^{-1} 1 is the Schur complement and
    (2R^TR + delta*I)^{-1} is applied via the thin SVD of R. For T < N the
    Woodbury identity handles the rank-deficient (2,2) block. M clusters the
    eigenvalues of M^{-1} K at three values regardless of N, bounding the
    MINRES iteration count independently of problem size.

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

        # Build MGW preconditioner from thin SVD of r_a.
        _, s_svd, vt = np.linalg.svd(r_a, full_matrices=False)
        lam = 2.0 * s_svd * s_svd
        delta = 1e-8 * float(lam[0]) if lam[0] > 0 else 1e-14
        k = lam.shape[0]

        if k == n_a:  # T >= N: A + delta*I is PD, direct formula
            d_inv = 1.0 / (lam + delta)

            def _a_inv(v, vvt=vt, dd=d_inv):
                """Apply (2R^TR + delta*I)^{-1} via full-rank SVD."""
                return vvt.T @ (dd * (vvt @ v))

        else:  # T < N: null-space directions via Woodbury
            d_wb = 1.0 / (lam + delta) - 1.0 / delta

            def _a_inv(v, vvt=vt, dd=d_wb, dt=delta):
                """Apply (2R^TR + delta*I)^{-1} via Woodbury."""
                return v / dt + vvt.T @ (dd * (vvt @ v))

        ones_a = np.ones(n_a)
        s_abs = float(ones_a @ _a_inv(ones_a))

        def _prec(x, na=n_a, ai=_a_inv, sa=s_abs):
            """Apply block diagonal MGW preconditioner M^{-1} to x."""
            out = np.empty(na + 1)
            out[:na] = ai(x[:na])
            out[na] = x[na] / sa
            return out

        b = np.zeros(n_a + 1)
        b[n_a] = 1.0
        kkt = LinearOperator(shape=(n_a + 1, n_a + 1), matvec=_matvec)  # type: ignore[call-arg]
        m = LinearOperator(shape=(n_a + 1, n_a + 1), matvec=_prec)  # type: ignore[call-arg]
        sol, _ = minres(kkt, b, M=m)
        w_a = sol[:n_a]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w


def minvar_cg(R):  # noqa: N803
    """Solve the minimum variance portfolio via preconditioned CG.

    Projects the problem onto the constraint-satisfying subspace using a QR
    basis ``P`` of the null space of the budget constraint, then applies CG to
    the reduced positive-definite system ``P^T R^T R P`` with a diagonal
    (Jacobi) preconditioner.

    The preconditioner diagonal is ``d[i] = ||R P[:,i]||^2``, the squared
    column norms of ``R @ P``, computed once in O(T*N) before the CG call and
    applied in O(N) per iteration. This scales each search direction by the
    local curvature, accelerating convergence when assets have heterogeneous
    variances.

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
        r_a = R[:, active]
        n_a = r_a.shape[1]
        p = np.linalg.qr(np.ones((n_a, 1)), mode="complete")[0][:, 1:]

        w0 = np.ones(n_a) / n_a
        r0 = r_a @ w0

        def _matvec(v, pp=p, ra=r_a):
            """Apply the reduced Hessian P^T R^T R P to v."""
            return pp.T @ (ra.T @ (ra @ (pp @ v)))

        # Jacobi preconditioner: diagonal of P^T R^T R P.
        rp = r_a @ p  # (T, N-1), O(T*N)
        diag_q = np.maximum(np.einsum("ti,ti->i", rp, rp), 1e-14)

        def _prec(v, dq=diag_q):
            """Apply Jacobi preconditioner diag(P^T R^T R P)^{-1} to v."""
            return v / dq

        op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)  # type: ignore[call-arg]
        m = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_prec)  # type: ignore[call-arg]
        rhs = -(p.T @ (r_a.T @ r0))
        v, _ = cg(op, rhs, M=m)
        w_a = w0 + p @ v
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    w /= w.sum()
    return w
