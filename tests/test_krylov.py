"""Tests for Problem.solve_minres and Problem.solve_cg."""

import numpy as np
import pytest

from fast_minimum_variance.api import Problem


class TestSolveMinres:
    """Tests for Problem.solve_minres."""

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_minres()
        assert w.shape == (problem.n,)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_minres()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_minres()
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, problem):
        """Iteration count is a positive integer."""
        _, iters = problem.solve_minres()
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, problem_small):
        """MINRES solution is close to the exact KKT solution."""
        w_kkt, _ = problem_small.solve_kkt()
        w_minres, _ = problem_small.solve_minres()
        np.testing.assert_allclose(w_minres, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(R).solve_minres()
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-10)

    def test_active_set_drops_asset(self):
        """Active-set iteration drops a high-variance correlated asset."""
        R = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w, _ = Problem(R).solve_minres()
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the MINRES solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = Problem(X).solve_minres()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_minres()
        assert w_mk[4] > w_mv[4]


class TestSolveCg:
    """Tests for Problem.solve_cg."""

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_cg()
        assert w.shape == (problem.n,)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_cg()
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, problem):
        """Iteration count is a positive integer."""
        _, iters = problem.solve_cg()
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, problem_small):
        """CG solution is close to the exact KKT solution."""
        w_kkt, _ = problem_small.solve_kkt()
        w_cg, _ = problem_small.solve_cg()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(R).solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-10)

    def test_active_set_drops_asset(self):
        """Active-set iteration drops a high-variance correlated asset."""
        R = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w, _ = Problem(R).solve_cg()
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_single_asset(self):
        """Fast-path for n_a==1: weight is exactly 1 for the sole active asset."""
        rng = np.random.default_rng(42)
        R = rng.standard_normal((50, 1))  # noqa: N806
        w, _ = Problem(R).solve_cg()
        assert w.shape == (1,)
        assert w[0] == pytest.approx(1.0)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the CG solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = Problem(X).solve_cg()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_cg()
        assert w_mk[4] > w_mv[4]


if __name__ == "__main__":
    pytest.main()
