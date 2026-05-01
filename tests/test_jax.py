"""Tests for Problem.solve_cg and Problem.solve_minres with backend='jax'.

The entire module is skipped when JAX is not installed, so CI without the JAX
extra will not fail.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from fast_minimum_variance.api import Problem  # noqa: E402


@pytest.fixture(scope="module")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return np.random.default_rng(42).standard_normal((200, 10))


@pytest.fixture(scope="module")
def problem_jax(X):  # noqa: N803
    """Problem instance with backend='jax'."""
    return Problem(X, backend="jax")


class TestSolveCgJax:
    """Tests for Problem.solve_cg with backend='jax'."""

    def test_jax_cg_weights_sum_to_one(self, problem_jax):
        """Budget constraint: weights sum to 1."""
        w, _ = problem_jax.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-4

    def test_jax_cg_weights_non_negative(self, problem_jax):
        """Long-only constraint: all weights are non-negative."""
        w, _ = problem_jax.solve_cg()
        assert np.all(w >= -1e-4)

    def test_jax_cg_agrees_with_numpy_cg(self, X):  # noqa: N803
        """JAX and NumPy backends agree to within float32 tolerance."""
        w_np, _ = Problem(X, backend="numpy").solve_cg()
        w_jax, _ = Problem(X, backend="jax").solve_cg()
        np.testing.assert_allclose(w_jax, w_np, atol=1e-4)

    def test_jax_cg_returns_shape(self, problem_jax):
        """Output weight vector has shape (N,)."""
        w, _ = problem_jax.solve_cg()
        assert w.shape == (problem_jax.n,)

    def test_jax_cg_returns_numpy_array(self, problem_jax):
        """Output is a plain NumPy array, not a JAX array."""
        w, _ = problem_jax.solve_cg()
        assert isinstance(w, np.ndarray)


class TestSolveMinresJax:
    """Tests for Problem.solve_minres with backend='jax'."""

    def test_jax_minres_weights_sum_to_one(self, problem_jax):
        """Budget constraint: weights sum to 1."""
        w, _ = problem_jax.solve_minres()
        assert abs(w.sum() - 1.0) < 1e-4

    def test_jax_minres_weights_non_negative(self, problem_jax):
        """Long-only constraint: all weights are non-negative."""
        w, _ = problem_jax.solve_minres()
        assert np.all(w >= -1e-4)

    def test_jax_minres_agrees_with_numpy_minres(self, X):  # noqa: N803
        """JAX and NumPy MINRES backends agree to within float32 tolerance."""
        w_np, _ = Problem(X, backend="numpy").solve_minres()
        w_jax, _ = Problem(X, backend="jax").solve_minres()
        np.testing.assert_allclose(w_jax, w_np, atol=1e-4)

    def test_jax_minres_returns_shape(self, problem_jax):
        """Output weight vector has shape (N,)."""
        w, _ = problem_jax.solve_minres()
        assert w.shape == (problem_jax.n,)

    def test_jax_minres_returns_numpy_array(self, problem_jax):
        """Output is a plain NumPy array, not a JAX array."""
        w, _ = problem_jax.solve_minres()
        assert isinstance(w, np.ndarray)

    def test_jax_minres_returns_positive_iters(self, problem_jax):
        """Iteration count is a positive integer."""
        _, iters = problem_jax.solve_minres()
        assert isinstance(iters, int)
        assert iters > 0


class TestJaxBackendValidation:
    """Tests for backend validation in Problem.__post_init__."""

    def test_jax_backend_invalid_raises(self):
        """Unknown backend string raises ValueError."""
        X = np.random.default_rng(0).standard_normal((50, 3))  # noqa: N806
        with pytest.raises(ValueError, match="Unknown backend"):
            Problem(X, backend="gpu")

    def test_jax_backend_unknown_raises(self):
        """Any unrecognised backend string raises ValueError."""
        X = np.random.default_rng(0).standard_normal((50, 3))  # noqa: N806
        with pytest.raises(ValueError, match="Unknown backend"):
            Problem(X, backend="cupy")


if __name__ == "__main__":
    pytest.main()
