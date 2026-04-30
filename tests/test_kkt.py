"""Tests for fast_minimum_variance.kkt."""

import numpy as np
import pytest

from fast_minimum_variance.kkt import build_kkt, solve_kkt


class TestBuildKkt:
    """Tests for build_kkt."""

    def test_shape(self, X_small):  # noqa: N803
        """A has shape (N+1, N+1) and b has shape (N+1,)."""
        N = X_small.shape[1]  # noqa: N806
        A, b = build_kkt(X_small)  # noqa: N806
        assert A.shape == (N + 1, N + 1)
        assert b.shape == (N + 1,)

    def test_rhs(self, X_small):  # noqa: N803
        """B is zero everywhere except the last entry which equals 1."""
        N = X_small.shape[1]  # noqa: N806
        _, b = build_kkt(X_small)
        np.testing.assert_array_equal(b[:N], 0.0)
        assert b[N] == 1.0

    def test_constraint_row_col(self, X_small):  # noqa: N803
        """Last row and column (excluding corner) are all ones."""
        N = X_small.shape[1]  # noqa: N806
        A, _ = build_kkt(X_small)  # noqa: N806
        np.testing.assert_array_equal(A[:N, N], 1.0)
        np.testing.assert_array_equal(A[N, :N], 1.0)
        assert A[N, N] == 0.0

    def test_hessian_block_symmetry(self, X_small):  # noqa: N803
        """The (N, N) Hessian block 2 R^T R is symmetric."""
        N = X_small.shape[1]  # noqa: N806
        A, _ = build_kkt(X_small)  # noqa: N806
        np.testing.assert_allclose(A[:N, :N], A[:N, :N].T)

    def test_hessian_block_positive_semidefinite(self, X_small):  # noqa: N803
        """The (N, N) Hessian block is positive semi-definite."""
        N = X_small.shape[1]  # noqa: N806
        A, _ = build_kkt(X_small)  # noqa: N806
        eigenvalues = np.linalg.eigvalsh(A[:N, :N])
        assert np.all(eigenvalues >= -1e-10)

    def test_rhs_with_return_term(self):
        """With rho > 0 the first N entries of rhs equal rho * mu."""
        N = 3  # noqa: N806
        X = np.eye(N)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = build_kkt(X, rho=0.5, mu=mu)
        np.testing.assert_allclose(rhs[:N], 0.5 * mu)
        assert rhs[N] == 1.0


class TestMinvarKkt:
    """Tests for minvar_kkt."""

    def test_shape(self, X):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = solve_kkt(X)
        assert w.shape == (X.shape[1],)

    def test_weights_sum_to_one(self, X):  # noqa: N803
        """Weights sum to 1."""
        w = solve_kkt(X)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_non_negative(self, X):  # noqa: N803
        """All weights are non-negative."""
        w = solve_kkt(X)
        assert np.all(w >= -1e-10)

    def test_trivial_case(self):
        """With a single asset the full weight goes to it."""
        R = np.ones((10, 1))  # noqa: N806
        w = solve_kkt(R)
        np.testing.assert_allclose(w, [1.0], atol=1e-10)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w = solve_kkt(R)
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv = solve_kkt(X)
        w_mk = solve_kkt(X, rho=1.0, mu=mu)
        assert w_mk[4] > w_mv[4]

    def test_all_inequalities_active_break(self):
        """Active-set loop terminates when every inequality constraint is binding."""
        X = np.eye(2)  # noqa: N806
        C = np.array([[1.0], [0.0]])  # w1 <= 0.3 # noqa: N806
        d = np.array([0.3])
        w = solve_kkt(X, C=C, d=d)
        np.testing.assert_allclose(w[0], 0.3, atol=1e-8)
        np.testing.assert_allclose(w[1], 0.7, atol=1e-8)

    def test_active_set_drops_asset(self):
        """Active-set iteration drops a high-variance correlated asset."""
        X = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w = solve_kkt(X)
        assert w[2] == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-8)
