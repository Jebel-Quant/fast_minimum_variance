"""Tests for fast_minimum_variance.kkt."""

import numpy as np
import pytest

from fast_minimum_variance.kkt import build_kkt, minvar_kkt


class TestBuildKkt:
    """Tests for build_kkt."""

    def test_shape(self, R_small):  # noqa: N803
        """A has shape (N+1, N+1) and b has shape (N+1,)."""
        N = R_small.shape[1]  # noqa: N806
        A, b = build_kkt(R_small)  # noqa: N806
        assert A.shape == (N + 1, N + 1)
        assert b.shape == (N + 1,)

    def test_rhs(self, R_small):  # noqa: N803
        """B is zero everywhere except the last entry which equals 1."""
        N = R_small.shape[1]  # noqa: N806
        _, b = build_kkt(R_small)
        np.testing.assert_array_equal(b[:N], 0.0)
        assert b[N] == 1.0

    def test_constraint_row_col(self, R_small):  # noqa: N803
        """Last row and column (excluding corner) are all ones."""
        N = R_small.shape[1]  # noqa: N806
        A, _ = build_kkt(R_small)  # noqa: N806
        np.testing.assert_array_equal(A[:N, N], 1.0)
        np.testing.assert_array_equal(A[N, :N], 1.0)
        assert A[N, N] == 0.0

    def test_hessian_block_symmetry(self, R_small):  # noqa: N803
        """The (N, N) Hessian block 2 R^T R is symmetric."""
        N = R_small.shape[1]  # noqa: N806
        A, _ = build_kkt(R_small)  # noqa: N806
        np.testing.assert_allclose(A[:N, :N], A[:N, :N].T)

    def test_hessian_block_positive_semidefinite(self, R_small):  # noqa: N803
        """The (N, N) Hessian block is positive semi-definite."""
        N = R_small.shape[1]  # noqa: N806
        A, _ = build_kkt(R_small)  # noqa: N806
        eigenvalues = np.linalg.eigvalsh(A[:N, :N])
        assert np.all(eigenvalues >= -1e-10)


class TestMinvarKkt:
    """Tests for minvar_kkt."""

    def test_shape(self, R):  # noqa: N803
        """Output weight vector has shape (N,)."""
        w = minvar_kkt(R)
        assert w.shape == (R.shape[1],)

    def test_weights_sum_to_one(self, R):  # noqa: N803
        """Weights sum to 1."""
        w = minvar_kkt(R)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_non_negative(self, R):  # noqa: N803
        """All weights are non-negative."""
        w = minvar_kkt(R)
        assert np.all(w >= -1e-10)

    def test_trivial_case(self):
        """With a single asset the full weight goes to it."""
        R = np.ones((10, 1))  # noqa: N806
        w = minvar_kkt(R)
        np.testing.assert_allclose(w, [1.0], atol=1e-10)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w = minvar_kkt(R)
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)
