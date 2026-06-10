"""Cross-validation: KKT solver vs CVXPY reference for _MinVarProblem."""

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
# KKT vs CVXPY
# ---------------------------------------------------------------------------


class TestKktVsCvxpy:
    """KKT and CVXPY must return the same portfolio up to solver tolerance."""

    def test_plain_minvar(self, X):  # noqa: N803
        """Plain minimum variance (alpha=0, rho=0)."""
        w_kkt, _ = Problem(X).solve_kkt()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_with_shrinkage(self, X):  # noqa: N803
        """Ledoit-Wolf shrinkage (alpha > 0)."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w_kkt, _ = Problem(X, alpha=alpha).solve_kkt()
        w_cvx, _ = Problem(X, alpha=alpha).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_with_return_tilt(self, X):  # noqa: N803
        """Return tilt (rho != 0, mu given)."""
        rng = np.random.default_rng(1)
        mu = rng.standard_normal(X.shape[1])
        w_kkt, _ = Problem(X, rho=0.5, mu=mu).solve_kkt()
        w_cvx, _ = Problem(X, rho=0.5, mu=mu).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_small_problem(self, X_small):  # noqa: N803
        """Small problem (T=100, N=5)."""
        w_kkt, _ = Problem(X_small).solve_kkt()
        w_cvx, _ = Problem(X_small).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_shrinkage_and_tilt(self, X):  # noqa: N803
        """Shrinkage and return tilt combined."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        mu = np.ones(N) / N
        w_kkt, _ = Problem(X, alpha=alpha, rho=0.3, mu=mu).solve_kkt()
        w_cvx, _ = Problem(X, alpha=alpha, rho=0.3, mu=mu).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N):  # noqa: N803
        """Agreement holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        w_kkt, _ = Problem(X).solve_kkt()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)


# ---------------------------------------------------------------------------
# Woodbury path (alpha=1, RMT target)
# ---------------------------------------------------------------------------


def _build_rmt_problem(T=300, N=50, seed=99, rho=0.0, mu=None):  # noqa: N803
    """Return (X, Problem) with alpha=1 and RMT low-rank target."""
    X = make_returns(T=T, N=N, seed=seed)  # noqa: N806
    target, lr_factors, _k, alpha = rmt_target_and_alpha(X)
    assert alpha == 1.0
    return X, Problem(X, alpha=alpha, target=target, target_lr=lr_factors, rho=rho, mu=mu)


class TestWoodbury:
    """Woodbury path must agree with CVXPY and with the CG path."""

    def test_minvar_agrees_with_cvxpy(self):
        """alpha=1, RMT target: KKT (Woodbury) matches CVXPY."""
        _, prob = _build_rmt_problem()
        w_kkt, _ = prob.solve_kkt()
        w_cvx, _ = prob.solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_minvar_agrees_with_cg(self):
        """alpha=1, RMT target: KKT (Woodbury) matches CG."""
        _, prob = _build_rmt_problem()
        w_kkt, _ = prob.solve_kkt()
        w_cg, _, _ = prob.solve_cg()
        np.testing.assert_allclose(w_kkt, w_cg, atol=1e-4)

    def test_return_tilt_agrees_with_cvxpy(self):
        """alpha=1, RMT target, return tilt: KKT (Woodbury) matches CVXPY."""
        rng = np.random.default_rng(5)
        N = 50  # noqa: N806
        mu = rng.standard_normal(N)
        _, prob = _build_rmt_problem(rho=0.5, mu=mu)
        w_kkt, _ = prob.solve_kkt()
        w_cvx, _ = prob.solve_cvxpy()
        np.testing.assert_allclose(w_kkt, w_cvx, atol=1e-4)

    def test_weights_are_valid(self):
        """Woodbury solution sums to 1 and is non-negative."""
        _, prob = _build_rmt_problem()
        w, _ = prob.solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= -1e-8).all()

    def test_woodbury_not_triggered_without_target_lr(self):
        """alpha=1 without target_lr falls back to dense solve (no crash)."""
        X = make_returns(T=300, N=20, seed=7)  # noqa: N806
        target, _, _k, _ = rmt_target_and_alpha(X)
        prob = Problem(X, alpha=1.0, target=target)  # no target_lr
        w, _ = prob.solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8
