"""Tests for solve_clarabel across _MinVarProblem and _Problem."""

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
    return MinVarProblem(X, target=np.eye(X.shape[1]))


@pytest.fixture(scope="module")
def prob(X):  # noqa: N803
    """Problem wrapping the session-scoped (200, 10) return matrix."""
    return Problem(X)


# ---------------------------------------------------------------------------
# MinVarProblem
# ---------------------------------------------------------------------------


class TestMinVarSolveClarabel:
    """solve_clarabel on _MinVarProblem (long-only min-var constraints)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_clarabel()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_clarabel()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_clarabel()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, mvp):
        """Clarabel solution agrees with KKT to solver tolerance."""
        w_clar, _ = mvp.solve_clarabel()
        w_kkt, _ = mvp.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-5)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = MinVarProblem(X_small, alpha=N / (N + T))
        w_clar, _ = p.solve_clarabel()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-5)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) sets q = -rho*mu."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = MinVarProblem(X_small, rho=0.5, mu=mu)
        w_clar, _ = p.solve_clarabel()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-4)

    def test_project_false(self, mvp):
        """project=False returns raw Clarabel solution without clipping."""
        w, iters = mvp.solve_clarabel(project=False)
        assert w.shape == (mvp.n,)
        assert iters >= 1

    def test_boyd_experiment_orthogonal_factors(self):
        """Orthogonal-factor MinVarProblem matches the simplex-constrained optimum."""
        rng = np.random.default_rng(0)
        x_mat, _ = np.linalg.qr(rng.standard_normal((5000, 50)))
        gram = x_mat.T @ x_mat
        np.testing.assert_allclose(gram, np.eye(50), atol=1e-10)

        p = MinVarProblem(x_mat)
        w, iters = p.solve_clarabel(project=False)

        expected = np.full(50, 1.0 / 50.0)
        assert w.shape == (50,)
        assert iters >= 1
        assert np.all(w >= -1e-8)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(w, expected, atol=1e-5)
        assert w @ gram @ w == pytest.approx(1.0 / 50.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class TestProblemSolveClarabel:
    """solve_clarabel on _Problem (arbitrary linear constraints)."""

    def test_shape(self, prob):
        """Output weight vector has shape (N,)."""
        w, _ = prob.solve_clarabel()
        assert w.shape == (prob.n,)

    def test_weights_sum_to_one(self, prob):
        """Weights sum to 1."""
        w, _ = prob.solve_clarabel()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, prob):
        """All weights are non-negative."""
        w, _ = prob.solve_clarabel()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, X_small):  # noqa: N803
        """Clarabel solution agrees with KKT to solver tolerance."""
        w_clar, _ = Problem(X_small).solve_clarabel()
        w_kkt, _ = Problem(X_small).solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-4)

    def test_agrees_with_minvar(self, X_small):  # noqa: N803
        """_Problem and _MinVarProblem should give the same clarabel solution."""
        w_prob, _ = Problem(X_small).solve_clarabel()
        w_mvp, _ = MinVarProblem(X_small).solve_clarabel()
        np.testing.assert_allclose(w_prob, w_mvp, atol=1e-5)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = Problem(X_small, alpha=N / (N + T))
        w_clar, _ = p.solve_clarabel()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-4)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) sets q = -rho*mu."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = Problem(X_small, rho=0.5, mu=mu)
        w_clar, _ = p.solve_clarabel()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-4)

    def test_project_false(self, prob):
        """project=False returns raw Clarabel solution without clipping."""
        w, iters = prob.solve_clarabel(project=False)
        assert w.shape == (prob.n,)
        assert iters >= 1
