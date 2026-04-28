"""Tests for fast_minimum_variance.cvx."""

import numpy as np
import pytest

from fast_minimum_variance.cvx import minvar_cvxpy
from fast_minimum_variance.kkt import minvar_kkt


class TestMinvarCvxpy:
    """Tests for minvar_cvxpy."""

    def test_shape(self, R):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = minvar_cvxpy(R)
        assert w.shape == (R.shape[1],)

    def test_weights_sum_to_one(self, R):  # noqa: N803
        """Weights sum to 1."""
        w = minvar_cvxpy(R)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, R):  # noqa: N803
        """All weights are non-negative."""
        w = minvar_cvxpy(R)
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, R_small):  # noqa: N803
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt = minvar_kkt(R_small)
        w_cvxpy = minvar_cvxpy(R_small)
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, R):  # noqa: N803
        """Optimal portfolio has variance ≤ equal-weight portfolio."""
        w_opt = minvar_cvxpy(R)
        N = R.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(R @ w_opt) <= np.linalg.norm(R @ w_eq) + 1e-6

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w = minvar_cvxpy(R)
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)
