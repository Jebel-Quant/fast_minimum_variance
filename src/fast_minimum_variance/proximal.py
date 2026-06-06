"""Fast solver for 0.5 ||mat @ x - vec||^2 s. t. {x >= 0, sum(x) = 1}.

This module implements proximal gradient descent for constrained linear least squares
optimization on the probability simplex. The algorithm is based on iterative projection
using the efficient simplex projection from Duchi et al. (2008).

The gradient is evaluated matrix-free as mat.T @ (mat @ w), avoiding explicit assembly
of the n x n normal matrix mat.T @ mat. The Lipschitz constant is estimated via power
iteration, also matrix-free.

References:
----------

Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
"Efficient Projections onto the l1-Ball for Learning in High Dimensions."
Proceedings of the 25th International Conference on Machine Learning (ICML).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def proj_simplex(
    vec: NDArray[np.floating],
    rad: float = 1.0,
) -> NDArray[np.floating]:
    """Project a vector onto the probability simplex.

    This function computes the Euclidean projection of a given vector onto the probability
    simplex. The simplex is defined as the set of non-negative vectors that sum to a
    given radius, typically 1. The projection ensures that the resulting vector satisfies
    these constraints.

    The algorithm is based on Duchi et al. (2008) "Efficient Projections onto the
    l1-Ball for Learning in High Dimensions".

    Parameters
    ----------
    vec : NDArray[np.floating]
        Input vector that is to be projected onto the simplex.
    rad : float, optional
        Radius of the simplex. The projected vector will have components summing
        to this value. Default is 1.0.

    Returns:
    -------
    NDArray[np.floating]
        The projected vector that lies on the probability simplex.

    Raises:
    ------
    ValueError
        If the input vector is empty.

    Examples:
    --------
    >>> import numpy as np
    >>> vec = np.array([1.0, 2.0, 3.0])
    >>> result = proj_simplex(vec)
    >>> bool(np.isclose(result.sum(), 1.0))
    True
    >>> bool(np.all(result >= 0))
    True

    """
    muu = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(muu) - rad)
    rho = max(np.where(muu > cummeans)[0])
    result: NDArray[np.floating] = np.maximum(vec - cummeans[rho], 0)
    return result


def _lipschitz(
    mat: NDArray[np.floating],
    extra_matvec=None,
    n_iter: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate lambda_max(mat.T @ mat + extra) via power iteration (matrix-free).

    extra_matvec: optional callable v -> extra @ v for a second SPD contribution.
    Each iteration costs O(rows * cols) — two matrix-vector products with mat —
    and never forms the cols x cols normal matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(mat.shape[1])
    v /= np.linalg.norm(v)
    lip = 1.0
    for _ in range(n_iter):
        w = mat.T @ (mat @ v)
        if extra_matvec is not None:
            w = w + extra_matvec(v)
        lip = float(np.linalg.norm(w))
        if lip < 1e-15:
            return lip
        v = w / lip
    return lip


def prox_gradient(
    mat: NDArray[np.floating],
    vec: NDArray[np.floating],
    *,
    extra_grad=None,
    eps_rel: float = 1e-6,
    max_iter: int = 100000,
) -> tuple[NDArray[np.floating], int]:
    """Perform proximal gradient descent to solve a constrained optimization problem.

    Solves the optimization problem:
        minimize 0.5 ||mat @ x - vec||^2 + g(x)
        subject to x >= 0, sum(x) = 1

    where g captures an optional extra gradient term supplied via ``extra_grad``.
    The gradient is evaluated matrix-free at each step; the normal matrix
    mat.T @ mat is never formed. The Lipschitz constant is estimated once via
    power iteration at O(n_power_iter * rows * cols) setup cost.

    Parameters
    ----------
    mat : NDArray[np.floating]
        A matrix of shape (n_samples, n_features).
    vec : NDArray[np.floating]
        A vector of shape (n_samples,).
    extra_grad : callable, optional
        v -> additional gradient term (e.g. ``alpha * target @ v`` for
        Ledoit-Wolf shrinkage). Must be SPD for convergence guarantees.
        When provided, the Lipschitz estimate accounts for this term.
    eps_rel : float, optional
        Relative step-size change stopping tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 100000.

    Returns:
    -------
    tuple[NDArray[np.floating], int]
        ``(w, n_iters)`` — weight vector of shape (n_features,) and the
        number of gradient steps taken.

    Examples:
    --------
    >>> import numpy as np
    >>> mat = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> vec = np.ones(2)
    >>> result, _ = prox_gradient(mat, vec)
    >>> bool(np.isclose(result.sum(), 1.0))
    True

    """
    rng = np.random.default_rng()
    prim_var: NDArray[np.floating] = np.asarray(rng.standard_normal(size=mat.shape[1]))
    lip = _lipschitz(mat, extra_matvec=extra_grad, rng=rng)
    step = 0.5 / lip if lip > 1e-15 else 1.0

    # Precompute mat.T @ vec once; zero for minimum-variance (vec = 0).
    out_prod = mat.T @ vec
    ite = 0
    err_rel = eps_rel + 1
    while err_rel > eps_rel and ite < max_iter:
        grad = mat.T @ (mat @ prim_var) - out_prod
        if extra_grad is not None:
            grad = grad + extra_grad(prim_var)
        prim_var_new = proj_simplex(prim_var - step * grad)
        err_rel = float(np.linalg.norm(prim_var - prim_var_new, 2))
        prim_var = prim_var_new.copy()
        ite += 1
    return prim_var, ite
