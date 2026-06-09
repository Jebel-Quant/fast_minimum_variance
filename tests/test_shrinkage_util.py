"""Tests for fast_minimum_variance.shrinkage.util — shrinkage target utilities."""

import numpy as np
import pytest

from fast_minimum_variance.shrinkage.util import (
    cc_target,
    lw_alpha_and_target,
    lw_alpha_and_target_hard,
    lw_alpha_for_target,
    oas_alpha_and_target,
    rmt_target_and_alpha,
)


@pytest.fixture(scope="module")
def X():  # noqa: N802
    """Return a demeaned (200, 10) return matrix."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))  # noqa: N806
    return X - X.mean(axis=0)


# ---------------------------------------------------------------------------
# lw_alpha_and_target
# ---------------------------------------------------------------------------


class TestLwAlphaAndTarget:
    """Tests for lw_alpha_and_target."""

    def test_alpha_in_unit_interval(self, X):  # noqa: N803
        """LW alpha is in [0, 1]."""
        alpha, _ = lw_alpha_and_target(X)
        assert 0.0 <= alpha <= 1.0

    def test_alpha_is_float(self, X):  # noqa: N803
        """Returned alpha is a Python float."""
        alpha, _ = lw_alpha_and_target(X)
        assert isinstance(alpha, float)

    def test_target_is_scaled_identity(self, X):  # noqa: N803
        """Target is bar_lam * I: diagonal elements equal, off-diagonal zero."""
        _, target = lw_alpha_and_target(X)
        n = X.shape[1]
        assert target.shape == (n, n)
        diag = np.diag(target)
        np.testing.assert_allclose(diag, diag[0])
        np.testing.assert_allclose(target - np.diag(diag), 0.0, atol=1e-12)

    def test_target_bar_lam_equals_mean_eigenvalue(self, X):  # noqa: N803
        """bar_lam equals the mean eigenvalue of the sample covariance."""
        T, n = X.shape  # noqa: N806
        _, target = lw_alpha_and_target(X)
        bar_lam = target[0, 0]
        cov = (X.T @ X) / T
        assert bar_lam == pytest.approx(np.trace(cov) / n, rel=1e-6)


# ---------------------------------------------------------------------------
# lw_alpha_and_target_hard
# ---------------------------------------------------------------------------


class TestLwAlphaAndTargetHard:
    """Tests for lw_alpha_and_target_hard."""

    def test_default_alpha_is_half(self, X):  # noqa: N803
        """Default alpha argument is 0.5."""
        alpha, _ = lw_alpha_and_target_hard(X)
        assert alpha == 0.5

    def test_custom_alpha_passthrough(self, X):  # noqa: N803
        """Supplied alpha is returned unchanged."""
        alpha, _ = lw_alpha_and_target_hard(X, alpha=0.3)
        assert alpha == 0.3

    def test_target_is_scaled_identity(self, X):  # noqa: N803
        """Target is bar_lam * I regardless of alpha."""
        _, target = lw_alpha_and_target_hard(X)
        n = X.shape[1]
        assert target.shape == (n, n)
        diag = np.diag(target)
        np.testing.assert_allclose(diag, diag[0])
        np.testing.assert_allclose(target - np.diag(diag), 0.0, atol=1e-12)

    def test_same_target_as_lw(self, X):  # noqa: N803
        """Target matches lw_alpha_and_target (same bar_lam * I formula)."""
        _, lw_tgt = lw_alpha_and_target(X)
        _, hard_tgt = lw_alpha_and_target_hard(X)
        np.testing.assert_allclose(hard_tgt, lw_tgt, rtol=1e-10)


# ---------------------------------------------------------------------------
# oas_alpha_and_target
# ---------------------------------------------------------------------------


class TestOasAlphaAndTarget:
    """Tests for oas_alpha_and_target."""

    def test_alpha_in_unit_interval(self, X):  # noqa: N803
        """OAS alpha is in [0, 1]."""
        alpha, _ = oas_alpha_and_target(X)
        assert 0.0 <= alpha <= 1.0

    def test_alpha_is_float(self, X):  # noqa: N803
        """Returned alpha is a Python float."""
        alpha, _ = oas_alpha_and_target(X)
        assert isinstance(alpha, float)

    def test_target_is_scaled_identity(self, X):  # noqa: N803
        """Target is bar_lam * I (same formula as LW)."""
        _, target = oas_alpha_and_target(X)
        n = X.shape[1]
        assert target.shape == (n, n)
        diag = np.diag(target)
        np.testing.assert_allclose(target - np.diag(diag), 0.0, atol=1e-12)

    def test_same_target_as_lw(self, X):  # noqa: N803
        """OAS and LW use the same bar_lam * I target."""
        _, lw_tgt = lw_alpha_and_target(X)
        _, oas_tgt = oas_alpha_and_target(X)
        np.testing.assert_allclose(oas_tgt, lw_tgt, rtol=1e-10)


# ---------------------------------------------------------------------------
# cc_target
# ---------------------------------------------------------------------------


class TestCcTarget:
    """Tests for cc_target (constant-correlation shrinkage target)."""

    def test_shape(self, X):  # noqa: N803
        """Target has shape (n, n)."""
        target, _ = cc_target(X)
        assert target.shape == (X.shape[1], X.shape[1])

    def test_diagonal_equals_sample_variance(self, X):  # noqa: N803
        """Diagonal entries equal the per-asset sample variances."""
        T, _ = X.shape  # noqa: N806
        cov = (X.T @ X) / T
        target, _ = cc_target(X)
        np.testing.assert_allclose(np.diag(target), np.diag(cov), rtol=1e-10)

    def test_rho_bar_in_unit_interval(self, X):  # noqa: N803
        """Mean correlation rho_bar is in (-1, 1)."""
        _, rho_bar = cc_target(X)
        assert -1.0 < rho_bar < 1.0

    def test_target_is_symmetric(self, X):  # noqa: N803
        """Target matrix is symmetric."""
        target, _ = cc_target(X)
        np.testing.assert_allclose(target, target.T, atol=1e-12)

    def test_target_is_psd(self, X):  # noqa: N803
        """Target matrix is positive semi-definite."""
        target, _ = cc_target(X)
        eigs = np.linalg.eigvalsh(target)
        assert np.all(eigs >= -1e-10)


# ---------------------------------------------------------------------------
# lw_alpha_for_target
# ---------------------------------------------------------------------------


class TestLwAlphaForTarget:
    """Tests for lw_alpha_for_target."""

    def test_alpha_in_unit_interval(self, X):  # noqa: N803
        """Returned alpha is in [0, 1]."""
        _, target = lw_alpha_and_target(X)
        alpha = lw_alpha_for_target(X, target)
        assert 0.0 <= alpha <= 1.0

    def test_agrees_with_lw_alpha_and_target(self, X):  # noqa: N803
        """Result matches the alpha from lw_alpha_and_target for the same target."""
        alpha_direct, target = lw_alpha_and_target(X)
        alpha_via_fn = lw_alpha_for_target(X, target)
        assert alpha_direct == pytest.approx(alpha_via_fn, rel=1e-6)

    def test_with_cc_target(self, X):  # noqa: N803
        """Works with a constant-correlation target."""
        target, _ = cc_target(X)
        alpha = lw_alpha_for_target(X, target)
        assert 0.0 <= alpha <= 1.0


# ---------------------------------------------------------------------------
# rmt_target_and_alpha
# ---------------------------------------------------------------------------


class TestRmtTargetAndAlpha:
    """Tests for rmt_target_and_alpha (Marchenko-Pastur eigenvalue cleaning)."""

    def test_return_shapes(self, X):  # noqa: N803
        """Return shapes are consistent with n assets and k signal factors."""
        target, lr_factors, k, _alpha = rmt_target_and_alpha(X)
        n = X.shape[1]
        _bar_lam, U_k, delta_k = lr_factors  # noqa: N806
        assert target.shape == (n, n)
        assert U_k.shape == (n, k)
        assert delta_k.shape == (k,)

    def test_k_non_negative(self, X):  # noqa: N803
        """Number of signal factors is non-negative."""
        _, _, k, _ = rmt_target_and_alpha(X)
        assert k >= 0

    def test_alpha_in_unit_interval(self, X):  # noqa: N803
        """LW oracle alpha is in [0, 1]."""
        _, _, _, alpha = rmt_target_and_alpha(X)
        assert 0.0 <= alpha <= 1.0

    def test_target_is_symmetric(self, X):  # noqa: N803
        """RMT target is symmetric."""
        target, *_ = rmt_target_and_alpha(X)
        np.testing.assert_allclose(target, target.T, atol=1e-10)

    def test_target_is_psd(self, X):  # noqa: N803
        """RMT target is positive semi-definite."""
        target, *_ = rmt_target_and_alpha(X)
        eigs = np.linalg.eigvalsh(target)
        assert np.all(eigs >= -1e-10)

    def test_lr_factors_reconstruct_target(self, X):  # noqa: N803
        """Low-rank factors reproduce the full target: bar_lam*I + U_k diag(delta_k) U_k^T."""
        target, lr_factors, _k, _ = rmt_target_and_alpha(X)
        bar_lam, U_k, delta_k = lr_factors  # noqa: N806
        n = X.shape[1]
        reconstructed = bar_lam * np.eye(n) + U_k @ np.diag(delta_k) @ U_k.T
        np.testing.assert_allclose(reconstructed, target, atol=1e-10)

    def test_rank_deficient(self):
        """Works on rank-deficient X (n > T) where many eigenvalues are zero."""
        X = np.random.default_rng(7).standard_normal((15, 50))  # noqa: N806
        target, lr_factors, k, alpha = rmt_target_and_alpha(X)
        bar_lam, U_k, delta_k = lr_factors  # noqa: N806
        assert target.shape == (50, 50)
        assert U_k.shape == (50, k)
        assert 0.0 <= alpha <= 1.0
        reconstructed = bar_lam * np.eye(50) + U_k @ np.diag(delta_k) @ U_k.T
        np.testing.assert_allclose(reconstructed, target, atol=1e-10)
