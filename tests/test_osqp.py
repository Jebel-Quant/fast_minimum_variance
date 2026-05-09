"""Tests for solve_osqp across _MinVarProblem and _Problem."""

import numpy as np
import pytest

from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.problem import _Problem as Problem


def _make_returns(T, N, seed=42):  # noqa: N803
    return np.random.default_rng(seed).standard_normal((T, N))


@pytest.fixture(scope="module")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return _make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="module")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 3) for fast solver tests."""
    return _make_returns(T=100, N=3, seed=0)


@pytest.fixture(scope="module")
def mvp(X):  # noqa: N803
    """MinVarProblem wrapping the session-scoped (200, 10) return matrix."""
    return MinVarProblem(X)


@pytest.fixture(scope="module")
def prob(X):  # noqa: N803
    """Problem wrapping the session-scoped (200, 10) return matrix."""
    return Problem(X)


# ---------------------------------------------------------------------------
# MinVarProblem
# ---------------------------------------------------------------------------


class TestMinVarSolveOsqp:
    """solve_osqp on _MinVarProblem (long-only min-var constraints)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_osqp()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_osqp()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_osqp()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, mvp):
        """OSQP solution agrees with KKT to solver tolerance."""
        w_osqp, _ = mvp.solve_osqp()
        w_kkt, _ = mvp.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = MinVarProblem(X_small, alpha=N / (N + T))
        w_osqp, _ = p.solve_osqp()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_with_explicit_target(self, X_small):  # noqa: N803
        """Explicit target exercises the target-aware P-matrix branch in solve_osqp."""
        T, N = X_small.shape  # noqa: N806
        alpha = N / (N + T)
        p = MinVarProblem(X_small, alpha=alpha, target=np.eye(N))
        w_osqp, _ = p.solve_osqp()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) sets q = -rho*mu."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = MinVarProblem(X_small, rho=0.5, mu=mu)
        w_osqp, _ = p.solve_osqp()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_project_false(self, mvp):
        """project=False returns raw OSQP solution without clipping."""
        w, iters = mvp.solve_osqp(project=False)
        assert w.shape == (mvp.n,)
        assert iters >= 1

    def test_iters_positive(self, mvp):
        """OSQP always takes at least one ADMM iteration."""
        _, iters = mvp.solve_osqp()
        assert iters > 0


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class TestProblemSolveOsqp:
    """solve_osqp on _Problem (arbitrary linear constraints)."""

    def test_shape(self, prob):
        """Output weight vector has shape (N,)."""
        w, _ = prob.solve_osqp()
        assert w.shape == (prob.n,)

    def test_weights_sum_to_one(self, prob):
        """Weights sum to 1."""
        w, _ = prob.solve_osqp()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, prob):
        """All weights are non-negative."""
        w, _ = prob.solve_osqp()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, X_small):  # noqa: N803
        """OSQP solution agrees with KKT to solver tolerance."""
        w_osqp, _ = Problem(X_small).solve_osqp()
        w_kkt, _ = Problem(X_small).solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_agrees_with_minvar(self, X_small):  # noqa: N803
        """_Problem and _MinVarProblem should give the same OSQP solution."""
        w_prob, _ = Problem(X_small).solve_osqp()
        w_mvp, _ = MinVarProblem(X_small).solve_osqp()
        np.testing.assert_allclose(w_prob, w_mvp, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = Problem(X_small, alpha=N / (N + T))
        w_osqp, _ = p.solve_osqp()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) sets q = -rho*mu."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = Problem(X_small, rho=0.5, mu=mu)
        w_osqp, _ = p.solve_osqp()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_osqp, w_kkt, atol=1e-4)

    def test_project_false(self, prob):
        """project=False returns raw OSQP solution without clipping."""
        w, iters = prob.solve_osqp(project=False)
        assert w.shape == (prob.n,)
        assert iters >= 1

    def test_iters_positive(self, prob):
        """OSQP always takes at least one ADMM iteration."""
        _, iters = prob.solve_osqp()
        assert iters > 0
