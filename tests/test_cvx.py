"""Tests for Problem.solve_cvxpy."""

import numpy as np
import pytest

from fast_minimum_variance.api import Problem


class TestSolveCvxpy:
    """Tests for Problem.solve_cvxpy."""

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_cvxpy()
        assert w.shape == (problem.X.shape[1],)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_cvxpy()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, problem_small):
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt, _ = problem_small.solve_kkt()
        w_cvxpy, _ = problem_small.solve_cvxpy()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X, problem):  # noqa: N803
        """Optimal portfolio has variance ≤ equal-weight portfolio."""
        w_opt, _ = problem.solve_cvxpy()
        N = X.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_term_tilts_weights(self, X, problem):  # noqa: N803
        """With rho > 0 the solution tilts toward the highest-return asset."""
        mu = np.zeros(X.shape[1])
        mu[-1] = 1.0
        w_mv, _ = problem.solve_cvxpy()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_cvxpy()
        assert w_mk[-1] > w_mv[-1]

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        X = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(X).solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)
