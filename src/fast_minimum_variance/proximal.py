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
    n_iter: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate lambda_max(mat.T @ mat) via power iteration (matrix-free).

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
        lip = float(np.linalg.norm(w))
        if lip < 1e-15:
            return lip
        v = w / lip
    return lip


def prox_gradient(
    mat: NDArray[np.floating],
    vec: NDArray[np.floating],
    eps_rel: float = 1e-6,
    max_iter: int = 1000,
) -> NDArray[np.floating]:
    """Perform proximal gradient descent to solve a constrained optimization problem.

    Solves the optimization problem:
        minimize 0.5 ||mat @ x - vec||^2
        subject to x >= 0, sum(x) = 1

    The gradient mat.T @ (mat @ x) - mat.T @ vec is evaluated matrix-free at each
    step; the normal matrix mat.T @ mat is never formed. The Lipschitz constant
    L = lambda_max(mat.T @ mat) is estimated once via power iteration, also
    matrix-free, at O(n_power_iter * rows * cols) setup cost.

    Parameters
    ----------
    mat : NDArray[np.floating]
        A matrix of shape (n_samples, n_features) used in the optimization
        problem.
    vec : NDArray[np.floating]
        A vector of shape (n_samples,) used in the optimization problem.
    eps_rel : float, optional
        The relative error threshold for stopping criteria. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the algorithm. Default is 1000.

    Returns:
    -------
    NDArray[np.floating]
        The solution vector of shape (n_features,) obtained after the
        optimization process.

    Examples:
    --------
    >>> import numpy as np
    >>> mat = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> vec = np.ones(2)
    >>> result = prox_gradient(mat, vec)
    >>> bool(np.isclose(result.sum(), 1.0))
    True

    """
    rng = np.random.default_rng()
    prim_var: NDArray[np.floating] = np.asarray(rng.standard_normal(size=mat.shape[1]))
    lip = _lipschitz(mat, rng=rng)
    step = 0.5 / lip if lip > 1e-15 else 1.0

    # Precompute mat.T @ vec once; zero for minimum-variance (vec = 0).
    out_prod = mat.T @ vec
    ite = 0
    err_rel = eps_rel + 1
    while err_rel > eps_rel and ite < max_iter:
        grad = mat.T @ (mat @ prim_var) - out_prod
        prim_var_new = proj_simplex(prim_var - step * grad)
        err_rel = float(np.linalg.norm(prim_var - prim_var_new, 2))
        prim_var = prim_var_new.copy()
        ite += 1
    return prim_var
