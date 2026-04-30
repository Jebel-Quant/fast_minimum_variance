"""Tests for fast_minimum_variance.krylov."""

import numpy as np
import pytest

from fast_minimum_variance.kkt import solve_kkt
from fast_minimum_variance.krylov import solve_cg, solve_minres


class TestMinvarMinres:
    """Tests for minvar_minres."""

    def test_shape(self, X):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w, _ = solve_minres(X)
        assert w.shape == (X.shape[1],)

    def test_weights_sum_to_one(self, X):  # noqa: N803
        """Weights sum to 1."""
        w, _ = solve_minres(X)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, X):  # noqa: N803
        """All weights are non-negative."""
        w, _ = solve_minres(X)
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, X):  # noqa: N803
        """Iteration count is a positive integer."""
        _, iters = solve_minres(X)
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, X_small):  # noqa: N803
        """MINRES solution is close to the exact KKT solution."""
        w_kkt = solve_kkt(X_small)
        w_minres, _ = solve_minres(X_small)
        np.testing.assert_allclose(w_minres, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = solve_minres(R)
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
        w, _ = solve_minres(R)
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the MINRES solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = solve_minres(X)
        w_mk, _ = solve_minres(X, rho=1.0, mu=mu)
        assert w_mk[4] > w_mv[4]


class TestMinvarCg:
    """Tests for minvar_cg."""

    def test_shape(self, X):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w, _ = solve_cg(X)
        assert w.shape == (X.shape[1],)

    def test_weights_sum_to_one(self, X):  # noqa: N803
        """Weights sum to 1."""
        w, _ = solve_cg(X)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, X):  # noqa: N803
        """All weights are non-negative."""
        w, _ = solve_cg(X)
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, X):  # noqa: N803
        """Iteration count is a positive integer."""
        _, iters = solve_cg(X)
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, X_small):  # noqa: N803
        """CG solution is close to the exact KKT solution."""
        w_kkt = solve_kkt(X_small)
        w_cg, _ = solve_cg(X_small)
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = solve_cg(R)
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
        w, _ = solve_cg(R)
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_single_asset(self):
        """Fast-path for n_a==1: weight is exactly 1 for the sole active asset."""
        rng = np.random.default_rng(42)
        R = rng.standard_normal((50, 1))  # noqa: N806
        w, _ = solve_cg(R)
        assert w.shape == (1,)
        assert w[0] == pytest.approx(1.0)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the CG solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = solve_cg(X)
        w_mk, _ = solve_cg(X, rho=1.0, mu=mu)
        assert w_mk[4] > w_mv[4]
