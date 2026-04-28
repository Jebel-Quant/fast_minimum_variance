"""Krylov subspace solvers for the minimum variance portfolio."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres

from fast_minimum_variance.kkt import build_kkt


def minvar_minres(R):  # noqa: N803
    """Solve the minimum variance portfolio via MINRES with active-set method."""
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    while True:
        A, b = build_kkt(R[:, active])  # noqa: N806
        sol, _ = minres(A, b)
        w_a = sol[: active.sum()]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    return w


def minvar_cg(R):  # noqa: N803
    """Solve the minimum variance portfolio via CG in the constraint-reduced space."""
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

        op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)
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
