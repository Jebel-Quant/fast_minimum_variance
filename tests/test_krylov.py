"""Tests for fast_minimum_variance.krylov."""

import numpy as np
import pytest

from fast_minimum_variance.kkt import minvar_kkt
from fast_minimum_variance.krylov import minvar_cg, minvar_minres


class TestMinvarMinres:
    """Tests for minvar_minres."""

    def test_shape(self, R):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = minvar_minres(R)
        assert w.shape == (R.shape[1],)

    def test_weights_sum_to_one(self, R):  # noqa: N803
        """Weights sum to 1."""
        w = minvar_minres(R)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, R):  # noqa: N803
        """All weights are non-negative."""
        w = minvar_minres(R)
        assert np.all(w >= -1e-10)

    def test_close_to_kkt(self, R_small):  # noqa: N803
        """MINRES solution is close to the exact KKT solution."""
        w_kkt = minvar_kkt(R_small)
        w_minres = minvar_minres(R_small)
        np.testing.assert_allclose(w_minres, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w = minvar_minres(R)
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
        w = minvar_minres(R)
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)


class TestMinvarCg:
    """Tests for minvar_cg."""

    def test_shape(self, R):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = minvar_cg(R)
        assert w.shape == (R.shape[1],)

    def test_weights_sum_to_one(self, R):  # noqa: N803
        """Weights sum to 1."""
        w = minvar_cg(R)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, R):  # noqa: N803
        """All weights are non-negative."""
        w = minvar_cg(R)
        assert np.all(w >= -1e-10)

    def test_close_to_kkt(self, R_small):  # noqa: N803
        """CG solution is close to the exact KKT solution."""
        w_kkt = minvar_kkt(R_small)
        w_cg = minvar_cg(R_small)
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w = minvar_cg(R)
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
        w = minvar_cg(R)
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_single_asset(self):
        """Fast-path for n_a==1: weight is exactly 1 for the sole active asset."""
        rng = np.random.default_rng(42)
        R = rng.standard_normal((50, 1))  # noqa: N806
        w = minvar_cg(R)
        assert w.shape == (1,)
        assert w[0] == pytest.approx(1.0)
