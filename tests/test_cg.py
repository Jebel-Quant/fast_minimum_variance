"""Cross-validation: CG solver vs an independent SLSQP reference for _MinVarProblem."""

import numpy as np
import pytest

from fast_minimum_variance import Problem


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

    def test_with_explicit_target(self, X, reference_weights):  # noqa: N803
        """CG with an explicit target matrix agrees with the oracle (target matvec branch)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N))
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)
