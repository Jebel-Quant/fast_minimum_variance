"""Cross-validation: CG solver vs an independent SLSQP reference for _MinVarProblem."""

import numpy as np
import pytest

from fast_minimum_variance import Problem
from fast_minimum_variance.shrinkage.util import rmt_target_and_alpha


def make_returns(T, N, seed=0):  # noqa: N803
    """Generate a T x N matrix of i.i.d. standard normal returns."""
    return np.random.default_rng(seed).standard_normal((T, N))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 5) with a fixed seed."""
    return make_returns(T=100, N=5, seed=7)


# ---------------------------------------------------------------------------
# CG vs reference oracle
# ---------------------------------------------------------------------------


class TestCgVsReference:
    """CG and the independent SLSQP oracle must return the same portfolio."""

    def test_plain_minvar(self, X, reference_weights):  # noqa: N803
        """Plain minimum variance (alpha=0, rho=0)."""
        prob = Problem(X)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_shrinkage(self, X, reference_weights):  # noqa: N803
        """Ledoit-Wolf shrinkage (alpha > 0)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T))
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_return_tilt(self, X, reference_weights):  # noqa: N803
        """Return tilt (rho != 0, mu given)."""
        mu = np.random.default_rng(1).standard_normal(X.shape[1])
        prob = Problem(X, rho=0.5, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_small_problem(self, X_small, reference_weights):  # noqa: N803
        """Small problem (T=100, N=5)."""
        prob = Problem(X_small)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_shrinkage_and_tilt(self, X, reference_weights):  # noqa: N803
        """Shrinkage and return tilt combined."""
        T, N = X.shape  # noqa: N806
        mu = np.ones(N) / N
        prob = Problem(X, alpha=N / (N + T), rho=0.3, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N, reference_weights):  # noqa: N803
        """Agreement holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        prob = Problem(X)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)


# ---------------------------------------------------------------------------
# Low-rank RMT target (alpha=1)
# ---------------------------------------------------------------------------


def _build_rmt_problem(T=300, N=50, seed=99, rho=0.0, mu=None):  # noqa: N803
    """Return (X, Problem) with alpha=1 and RMT low-rank target."""
    X = make_returns(T=T, N=N, seed=seed)  # noqa: N806
    target, lr_factors, _k, alpha = rmt_target_and_alpha(X)
    assert alpha == 1.0
    return X, Problem(X, alpha=alpha, target=target, target_lr=lr_factors, rho=rho, mu=mu)


class TestLowRank:
    """The alpha=1 low-rank factor path must agree with the reference oracle."""

    def test_minvar_agrees_with_reference(self, reference_weights):
        """alpha=1, RMT target: CG matches the oracle."""
        _, prob = _build_rmt_problem()
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_return_tilt_agrees_with_reference(self, reference_weights):
        """alpha=1, RMT target, return tilt: CG matches the oracle."""
        mu = np.random.default_rng(5).standard_normal(50)
        _, prob = _build_rmt_problem(rho=0.5, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_weights_are_valid(self):
        """Low-rank solution sums to 1 and is non-negative."""
        _, prob = _build_rmt_problem()
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-6).all()

    def test_dense_target_without_target_lr(self):
        """alpha=1 with only a dense target (no target_lr) solves via the Gram path."""
        X = make_returns(T=300, N=20, seed=7)  # noqa: N806
        target, _, _k, _ = rmt_target_and_alpha(X)
        prob = Problem(X, alpha=1.0, target=target)  # no target_lr
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6
