"""Tests for fast_minimum_variance.cvx."""

import numpy as np
import pytest

from fast_minimum_variance.cvx import solve_cvxpy
from fast_minimum_variance.kkt import minvar_kkt


class TestMinvarCvxpy:
    """Tests for minvar_cvxpy."""

    def test_shape(self, X):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = solve_cvxpy(X)
        assert w.shape == (X.shape[1],)

    def test_weights_sum_to_one(self, X):  # noqa: N803
        """Weights sum to 1."""
        w = solve_cvxpy(X)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, X):  # noqa: N803
        """All weights are non-negative."""
        w = solve_cvxpy(X)
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, X_small):  # noqa: N803
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt = minvar_kkt(X_small)
        w_cvxpy = solve_cvxpy(X_small)
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X):  # noqa: N803
        """Optimal portfolio has variance ≤ equal-weight portfolio."""
        w_opt = solve_cvxpy(X)
        N = X.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_term_tilts_weights(self, X):  # noqa: N803
        """With rho > 0 the solution tilts toward the highest-return asset."""
        mu = np.zeros(X.shape[1])
        mu[-1] = 1.0
        w_mv = solve_cvxpy(X)
        w_mk = solve_cvxpy(X, rho=1.0, mu=mu)
        assert w_mk[-1] > w_mv[-1]

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        X = rng.standard_normal((200, N))  # noqa: N806
        w = solve_cvxpy(X)
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)
