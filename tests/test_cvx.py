"""Tests for fast_minimum_variance.cvx."""

import numpy as np
import pytest

from fast_minimum_variance.api import API
from fast_minimum_variance.cvx import solve_cvxpy
from fast_minimum_variance.kkt import solve_kkt


class TestMinvarCvxpy:
    """Tests for minvar_cvxpy."""

    def test_shape(self, api):
        """Output weight vector has shape (N,)."""
        w, _ = solve_cvxpy(api)
        assert w.shape == (api.X.shape[1],)

    def test_weights_sum_to_one(self, api):
        """Weights sum to 1."""
        w, _ = solve_cvxpy(api)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, api):
        """All weights are non-negative."""
        w, _ = solve_cvxpy(api)
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, api_small):
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt, _ = solve_kkt(api_small)
        w_cvxpy, _ = solve_cvxpy(api_small)
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X, api):  # noqa: N803
        """Optimal portfolio has variance ≤ equal-weight portfolio."""
        w_opt, _ = solve_cvxpy(api)
        N = X.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_term_tilts_weights(self, X, api):  # noqa: N803
        """With rho > 0 the solution tilts toward the highest-return asset."""
        mu = np.zeros(X.shape[1])
        mu[-1] = 1.0
        w_mv, _ = solve_cvxpy(api)
        w_mk, _ = solve_cvxpy(API(X, rho=1.0, mu=mu))
        assert w_mk[-1] > w_mv[-1]

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        X = rng.standard_normal((200, N))  # noqa: N806
        w, _ = solve_cvxpy(API(X))
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)
