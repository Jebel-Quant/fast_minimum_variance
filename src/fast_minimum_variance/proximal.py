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

from typing import TYPE_CHECKING, cast

import numpy as np
from cvx.linalg import power_iteration

if TYPE_CHECKING:
    from collections.abc import Callable

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
    extra_matvec: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    n_iter: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate lambda_max(mat.T @ mat + extra) via power iteration (matrix-free).

    extra_matvec: optional callable v -> extra @ v for a second SPD contribution.
    Each iteration costs O(rows * cols) — two matrix-vector products with mat —
    and never forms the cols x cols normal matrix. The iteration is delegated to
    cvx-linalg's operator-aware power_iteration, applied matrix-free to the normal
    operator v -> mat.T @ (mat @ v) (+ extra).
    """
    seed = None if rng is None else int(rng.integers(np.iinfo(np.int64).max))

    def normal_matvec(v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the normal operator mat.T @ mat (+ extra) to v, matrix-free."""
        w = mat.T @ (mat @ v)
        if extra_matvec is not None:
            w = w + extra_matvec(v)
        return w

    matvec = cast("Callable[[NDArray[np.float64]], NDArray[np.float64]]", normal_matvec)
    eigenvalue, _ = power_iteration(matvec, n=mat.shape[1], n_iter=n_iter, seed=seed)
    return max(float(eigenvalue), 0.0)


def fista_gradient(
    mat: NDArray[np.floating],
    vec: NDArray[np.floating],
    *,
    extra_grad: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    eps_rel: float = 1e-6,
    max_iter: int = 100000,
) -> tuple[NDArray[np.floating], int]:
    r"""FISTA (Nesterov-accelerated proximal gradient) on the probability simplex.

    Same interface as ``prox_gradient`` but uses the Beck-Teboulle momentum
    sequence $t_{k+1} = (1 + \\sqrt{1+4t_k^2})/2$ to achieve $O(1/k^2)$
    convergence for convex objectives (versus $O(1/k)$ for plain gradient
    descent).  For strongly convex $f$ with condition number $\\kappa$ the
    linear convergence rate is $(1 - 1/\\sqrt{\\kappa})^k$, matching CG's
    asymptotic iteration count.

    The gradient is evaluated at the extrapolated point $y_k$; the simplex
    projection is applied to obtain $x_k$; the momentum update then forms
    $y_{k+1} = x_k + \\frac{t_k-1}{t_{k+1}}(x_k - x_{k-1})$.

    References:
    ----------
    Beck, A., & Teboulle, M. (2009). "A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems." SIAM Journal on Imaging Sciences.

    Examples:
    --------
    >>> import numpy as np
    >>> mat = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> vec = np.ones(2)
    >>> result, _ = fista_gradient(mat, vec)
    >>> bool(np.isclose(result.sum(), 1.0))
    True

    """
    rng = np.random.default_rng()
    lip = _lipschitz(mat, extra_matvec=extra_grad, rng=rng)
    step = 1.0 / lip if lip > 1e-15 else 1.0
    out_prod = mat.T @ vec

    x = proj_simplex(np.asarray(rng.standard_normal(mat.shape[1])))
    y = x.copy()
    t = 1.0

    for ite in range(1, max_iter + 1):  # noqa: B007
        grad = mat.T @ (mat @ y) - out_prod
        if extra_grad is not None:
            grad = grad + extra_grad(y)
        x_new = proj_simplex(y - step * grad)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)

        err = float(np.linalg.norm(x - x_new))
        x = x_new
        t = t_new

        if err < eps_rel:
            break

    return x, ite


def prox_gradient(
    mat: NDArray[np.floating],
    vec: NDArray[np.floating],
    *,
    extra_grad: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
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
    step = 1.0 / lip if lip > 1e-15 else 1.0

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
