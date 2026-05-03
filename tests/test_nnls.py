"""Cross-validation: NNLS solver vs CVXPY reference for _MinVarProblem."""

import numpy as np
import pytest

from fast_minimum_variance import Problem


def make_returns(T, N, seed=0):  # noqa: N803
    """Generate a T x N matrix of i.i.d. standard normal returns."""
    return np.random.default_rng(seed).standard_normal((T, N))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 5) with a fixed seed."""
    return make_returns(T=100, N=5, seed=7)


# ---------------------------------------------------------------------------
# NNLS vs CVXPY
# ---------------------------------------------------------------------------


class TestNnlsVsCvxpy:
    """NNLS and CVXPY must return the same portfolio up to solver tolerance."""

    def test_plain_minvar(self, X):  # noqa: N803
        """Plain minimum variance (alpha=0, rho=0)."""
        w_nnls, _ = Problem(X).solve_nnls()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_nnls, w_cvx, atol=1e-3)

    def test_with_shrinkage(self, X):  # noqa: N803
        """Ledoit-Wolf shrinkage (alpha > 0)."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w_nnls, _ = Problem(X, alpha=alpha).solve_nnls()
        w_cvx, _ = Problem(X, alpha=alpha).solve_cvxpy()
        np.testing.assert_allclose(w_nnls, w_cvx, atol=1e-3)

    def test_small_problem(self, X_small):  # noqa: N803
        """Small problem (T=100, N=5)."""
        w_nnls, _ = Problem(X_small).solve_nnls()
        w_cvx, _ = Problem(X_small).solve_cvxpy()
        np.testing.assert_allclose(w_nnls, w_cvx, atol=1e-3)

    def test_shrinkage_only(self, X):  # noqa: N803
        """Shrinkage without return tilt."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w_nnls, _ = Problem(X, alpha=alpha).solve_nnls()
        w_cvx, _ = Problem(X, alpha=alpha).solve_cvxpy()
        np.testing.assert_allclose(w_nnls, w_cvx, atol=1e-3)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N):  # noqa: N803
        """Agreement holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        w_nnls, _ = Problem(X).solve_nnls()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_nnls, w_cvx, atol=1e-3)
