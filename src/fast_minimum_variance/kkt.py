"""Projected primal-dual solver for the long-only minimum-variance QP."""

import numpy as np


def pd_projected_qp_solver(X, mu=None, rho=None, alpha=0, tol=1e-5, maxiter=200):  # noqa: N803
    """Solve the long-only minimum-variance QP via projected primal-dual gradient.

    Minimises ``(1-alpha)||Xw||^2 + ridge*||w||^2 - rho*mu^T w`` subject to
    ``1^T w = 1`` and ``w >= 0`` using a primal-dual gradient loop with
    non-negativity projection and budget-constraint renormalisation.

    Parameters
    ----------
    X : ndarray (T, n)
    mu : ndarray (n,) or None
    rho : float or None
    alpha : float
    tol : float
    maxiter : int

    Returns:
    -------
    w : ndarray (n,)
    lam : float
    iters : int
    """
    n = X.shape[1]
    oma = 1.0 - alpha

    ridge = alpha * (np.sum(X * X) / X.shape[0])

    def H(v):  # noqa: N802
        """Apply the regularised Hessian 2*(oma*X^T X + ridge*I) to v."""
        return 2.0 * (oma * (X.T @ (X @ v)) + ridge * v)

    b = rho * mu if mu is not None else np.zeros(n)

    # start feasible in simplex (IMPORTANT)
    w = np.ones(n) / n
    lam = 0.0

    eta_w = 0.05
    eta_l = 0.02
    beta = 2.0

    ones = np.ones(n)

    for _ in range(maxiter):
        hw = H(w)

        c = np.sum(w) - 1.0

        # corrected geometry:
        # project constraint in correct direction (rank-1 structure)
        r_w = hw + lam * ones + beta * c * ones - b
        r_l = c

        res = np.linalg.norm(r_w) + abs(r_l)
        if res < tol:
            break

        # ----------------------------
        # CRITICAL FIX: remove drift in nullspace direction
        # ----------------------------
        mean_r = np.mean(r_w)
        r_w = r_w - mean_r * ones  # removes artificial symmetric drift

        # primal update
        w -= eta_w * r_w

        # dual update
        lam -= eta_l * r_l

        # projection
        w = np.maximum(w, 0.0)

        # renormalize to avoid simplex drift explosion
        s = np.sum(w)
        if s > 1e-12:
            w /= s

        lam = np.clip(lam, -1e6, 1e6)

    return w, lam, _ + 1
