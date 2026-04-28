"""KKT system construction for the minimum variance portfolio."""

import numpy as np


def build_kkt(R):  # noqa: N803
    """Build the KKT system matrix and RHS for the minimum variance problem."""
    n_a = R.shape[1]
    A = np.zeros((n_a + 1, n_a + 1))  # noqa: N806
    A[:n_a, :n_a] = 2 * R.T @ R
    A[:n_a, n_a] = 1
    A[n_a, :n_a] = 1
    b = np.zeros(n_a + 1)
    b[n_a] = 1
    return A, b


def minvar_kkt(R):  # noqa: N803
    """Solve the minimum variance portfolio via the KKT system with active-set method."""
    n = R.shape[1]
    active = np.ones(n, dtype=bool)
    while True:
        A, b = build_kkt(R[:, active])  # noqa: N806
        sol = np.linalg.solve(A, b)
        w_a = sol[: active.sum()]
        if np.all(w_a >= -1e-10):
            break
        active[np.where(active)[0][w_a < 0]] = False
    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0)
    return w
