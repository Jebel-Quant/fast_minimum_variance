"""Tests for the proximal module."""

from __future__ import annotations

import numpy as np
import pytest

from fast_minimum_variance import Problem
from fast_minimum_variance.proximal import _lipschitz, fista_gradient, proj_simplex, prox_gradient


def make_returns(T, N, seed=0):  # noqa: N803
    """Generate a (T, N) matrix of standard-normal returns."""
    return np.random.default_rng(seed).standard_normal((T, N))


class TestProjSimplex:
    """Test suite for proj_simplex function."""

    @pytest.mark.parametrize("n", [2, 5, 10, 100, 1000])
    def test_output_sums_to_one(self, n: int) -> None:
        """Verify projected vector sums to 1 (default radius)."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(n)
        result = proj_simplex(vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("rad", [0.5, 1.0, 2.0, 10.0])
    def test_output_sums_to_radius(self, rad: float) -> None:
        """Verify projected vector sums to specified radius."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(10)
        result = proj_simplex(vec, rad=rad)
        np.testing.assert_allclose(result.sum(), rad, rtol=1e-10)

    def test_output_non_negative(self) -> None:
        """Verify all elements of projected vector are non-negative."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(100)
        result = proj_simplex(vec)
        assert np.all(result >= 0)

    def test_idempotent(self) -> None:
        """Projecting a valid simplex point should return the same point."""
        vec = np.array([0.2, 0.3, 0.5])
        result = proj_simplex(vec)
        np.testing.assert_allclose(result, vec, rtol=1e-10)

    def test_already_on_simplex(self) -> None:
        """Vector already on simplex should be unchanged."""
        vec = np.array([0.25, 0.25, 0.25, 0.25])
        result = proj_simplex(vec)
        np.testing.assert_allclose(result, vec, rtol=1e-10)

    def test_single_element(self) -> None:
        """Single element vector projects to [rad]."""
        result = proj_simplex(np.array([5.0]))
        np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-10)

    def test_all_negative_input(self) -> None:
        """All negative inputs should project to corner of simplex."""
        vec = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        result = proj_simplex(vec)
        # Should project to [1, 0, 0, 0, 0] since -1 is largest
        assert result[0] == pytest.approx(1.0, rel=1e-10)
        assert np.sum(result[1:]) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_input(self) -> None:
        """Uniform input should project to uniform simplex point."""
        vec = np.array([3.0, 3.0, 3.0, 3.0])
        result = proj_simplex(vec)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_projection_is_closest_point(self, n: int) -> None:
        """Verify projection is the closest point on simplex."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(n)
        proj = proj_simplex(vec)

        # Any point on simplex should be farther from vec than proj
        # Test with some random simplex points
        for _ in range(10):
            random_simplex = rng.dirichlet(np.ones(n))
            dist_to_proj = np.linalg.norm(vec - proj)
            dist_to_random = np.linalg.norm(vec - random_simplex)
            assert dist_to_proj <= dist_to_random + 1e-10

    def test_two_elements(self) -> None:
        """Test projection with only 2 elements."""
        vec = np.array([2.0, -1.0])
        result = proj_simplex(vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
        assert np.all(result >= 0)

    def test_large_values(self) -> None:
        """Test projection with very large values."""
        vec = np.array([1e10, 1e10, 1e10])
        result = proj_simplex(vec)
        # Numerical precision degrades with very large values
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(result, np.array([1 / 3, 1 / 3, 1 / 3]), rtol=1e-5)

    def test_small_values(self) -> None:
        """Test projection with very small values."""
        vec = np.array([1e-10, 2e-10, 3e-10])
        result = proj_simplex(vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)

    def test_mixed_scales(self) -> None:
        """Test projection with mixed scale values."""
        vec = np.array([1e8, 1e-8, 1e4, 1e-4])
        result = proj_simplex(vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
        assert np.all(result >= 0)


class TestProxGradient:
    """Test suite for prox_gradient function."""

    def test_output_on_simplex(self) -> None:
        """Verify output satisfies simplex constraints."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((5, 3))
        vec = rng.standard_normal(5)
        result, _ = prox_gradient(mat, vec)

        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-6)
        assert np.all(result >= -1e-10)

    @pytest.mark.parametrize("shape", [(3, 2), (5, 5), (10, 3), (20, 10)])
    def test_various_shapes(self, shape: tuple[int, int]) -> None:
        """Test with various matrix shapes."""
        rng = np.random.default_rng(42)
        m, n = shape
        mat = rng.standard_normal((m, n))
        vec = rng.standard_normal(m)
        result, _ = prox_gradient(mat, vec)

        assert result.shape == (n,)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        assert np.all(result >= -1e-10)

    def test_identity_matrix(self) -> None:
        """Test with identity matrix - solution should be projection of vec."""
        n = 5
        mat = np.eye(n)
        vec = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # Already on simplex
        result, _ = prox_gradient(mat, vec)
        np.testing.assert_allclose(result, vec, rtol=1e-5)

    def test_convergence_tolerance(self) -> None:
        """Test that tighter tolerance gives valid results."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((10, 5))
        vec = rng.standard_normal(10)

        result_loose, _ = prox_gradient(mat, vec, eps_rel=1e-3)
        result_tight, _ = prox_gradient(mat, vec, eps_rel=1e-10)

        # Both should satisfy constraints
        np.testing.assert_allclose(result_loose.sum(), 1.0, rtol=1e-3)
        np.testing.assert_allclose(result_tight.sum(), 1.0, rtol=1e-10)

    def test_max_iter_respected(self) -> None:
        """Test that max_iter parameter is respected."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((100, 50))
        vec = rng.standard_normal(100)

        result, iters = prox_gradient(mat, vec, max_iter=10)
        assert result.shape == (50,)
        assert iters <= 10

    def test_objective_convergence(self) -> None:
        """Test that algorithm converges to similar objective values."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((5, 3))
        vec = rng.standard_normal(5)

        # Run multiple times - results may vary due to random init
        # but objective values should be similar
        objectives = []
        for _ in range(5):
            result, _ = prox_gradient(mat, vec, eps_rel=1e-10)
            obj = 0.5 * np.linalg.norm(mat @ result - vec) ** 2
            objectives.append(obj)

        # All objectives should be close to the minimum
        np.testing.assert_allclose(objectives, objectives[0], rtol=1e-3)

    def test_near_zero_matrix(self) -> None:
        """Test with near-zero matrix (low Lipschitz constant)."""
        mat = np.array([[1e-10, 0], [0, 1e-10]])
        vec = np.array([1.0, 1.0])
        result, _ = prox_gradient(mat, vec)
        assert result.shape == (2,)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_rank_deficient(self) -> None:
        """Test with rank-deficient matrix."""
        mat = np.array([[1, 2, 3], [2, 4, 6]])  # Rank 1
        vec = np.array([1.0, 2.0])
        result, _ = prox_gradient(mat, vec)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_ill_conditioned(self) -> None:
        """Test with ill-conditioned matrix."""
        # Create an ill-conditioned matrix
        u = np.array([[1, 0], [0, 1], [0, 0]])
        s = np.array([1e6, 1e-6])
        vt = np.array([[1, 0], [0, 1]])
        mat = u @ np.diag(s) @ vt
        vec = np.array([1.0, 1.0, 1.0])

        result, _ = prox_gradient(mat, vec, eps_rel=1e-6, max_iter=5000)
        assert result.shape == (2,)
        # May not converge perfectly but should satisfy constraints approximately
        assert abs(result.sum() - 1.0) < 0.01

    @pytest.mark.parametrize("seed", range(5))
    def test_random_problems(self, seed: int) -> None:
        """Test with various random problem instances."""
        rng = np.random.default_rng(seed)
        m, n = rng.integers(5, 20), rng.integers(2, 10)
        mat = rng.standard_normal((m, n))
        vec = rng.standard_normal(m)

        result, _ = prox_gradient(mat, vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        assert np.all(result >= -1e-10)


class TestKKTConditions:
    """Test mathematical properties via KKT conditions."""

    @pytest.mark.parametrize("seed", range(5))
    def test_proj_simplex_kkt(self, seed: int) -> None:
        """Verify KKT optimality conditions for simplex projection.

        For the projection problem:
            min 0.5 ||x - v||^2
            s.t. sum(x) = 1, x >= 0

        KKT conditions require:
            - Primal feasibility: sum(x) = 1, x >= 0
            - For active constraints (x_i > 0): x_i = v_i - lambda
        """
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(10)
        x = proj_simplex(v)

        # Primal feasibility
        np.testing.assert_allclose(x.sum(), 1.0, rtol=1e-10)
        assert np.all(x >= -1e-12)

        # For x_i > 0, we have x_i = v_i - lambda
        # So lambda should be the same for all active components
        active_mask = x > 1e-10
        if np.any(active_mask):
            lambdas = v[active_mask] - x[active_mask]
            # All should be approximately equal
            np.testing.assert_allclose(lambdas, lambdas[0], rtol=1e-8)


class TestSolveProximal:
    """Cross-validation: solve_proximal vs CVXPY reference."""

    @pytest.fixture(scope="class")
    def X(self):  # noqa: N802
        """Return a (200, 10) return matrix."""
        return make_returns(T=200, N=10, seed=42)

    @pytest.fixture(scope="class")
    def X_small(self):  # noqa: N802
        """Return a (100, 5) return matrix."""
        return make_returns(T=100, N=5, seed=7)

    def test_return_shape(self, X) -> None:  # noqa: N803
        """solve_proximal returns (w, iters) with w of shape (N,)."""
        w, iters = Problem(X).solve_proximal()
        assert w.shape == (X.shape[1],)
        assert iters >= 1

    def test_weights_sum_to_one(self, X) -> None:  # noqa: N803
        """Weights must sum to 1."""
        w, _ = Problem(X).solve_proximal()
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6)

    def test_weights_non_negative(self, X) -> None:  # noqa: N803
        """Weights must be non-negative after projection."""
        w, _ = Problem(X).solve_proximal()
        assert np.all(w >= 0)

    def test_plain_minvar_vs_cvxpy(self, X) -> None:  # noqa: N803
        """Plain minimum variance agrees with CVXPY reference."""
        w_prox, _ = Problem(X).solve_proximal()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_prox, w_cvx, atol=1e-3)

    def test_with_shrinkage_vs_cvxpy(self, X) -> None:  # noqa: N803
        """Shrinkage (alpha > 0, target = I) agrees with CVXPY reference."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w_prox, _ = Problem(X, alpha=alpha, target=np.eye(N)).solve_proximal()
        w_cvx, _ = Problem(X, alpha=alpha, target=np.eye(N)).solve_cvxpy()
        np.testing.assert_allclose(w_prox, w_cvx, atol=1e-3)

    def test_small_problem_vs_cvxpy(self, X_small) -> None:  # noqa: N803
        """Small problem (T=100, N=5) agrees with CVXPY reference."""
        w_prox, _ = Problem(X_small).solve_proximal()
        w_cvx, _ = Problem(X_small).solve_cvxpy()
        np.testing.assert_allclose(w_prox, w_cvx, atol=1e-3)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N) -> None:  # noqa: N803
        """Agreement with CVXPY holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        w_prox, _ = Problem(X).solve_proximal()
        w_cvx, _ = Problem(X).solve_cvxpy()
        np.testing.assert_allclose(w_prox, w_cvx, atol=1e-3)

    def test_project_false(self, X) -> None:  # noqa: N803
        """project=False skips clip-and-renormalize; result still on simplex from prox_gradient."""
        _w_proj, _ = Problem(X).solve_proximal(project=True)
        w_raw, _ = Problem(X).solve_proximal(project=False)
        # Both should satisfy constraints (prox_gradient enforces simplex internally)
        np.testing.assert_allclose(w_raw.sum(), 1.0, rtol=1e-6)
        assert np.all(w_raw >= -1e-10)


class TestLipschitz:
    """Tests for the _lipschitz power-iteration helper."""

    def test_returns_positive(self) -> None:
        """Lipschitz estimate is strictly positive for a non-zero matrix."""
        mat = np.eye(3)
        assert _lipschitz(mat) > 0

    def test_rng_none_branch(self) -> None:
        """Passing rng=None triggers internal default_rng creation."""
        mat = np.random.default_rng(0).standard_normal((5, 3))
        lip = _lipschitz(mat, rng=None)
        assert lip > 0


class TestFistaGradient:
    """Tests for fista_gradient (Nesterov-accelerated proximal gradient)."""

    def test_output_on_simplex(self) -> None:
        """Result lies on the probability simplex."""
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((5, 3))
        vec = rng.standard_normal(5)
        result, _ = fista_gradient(mat, vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        assert np.all(result >= -1e-10)

    def test_iters_positive(self) -> None:
        """Iteration count is at least 1."""
        mat = np.eye(3)
        _, iters = fista_gradient(mat, np.zeros(3))
        assert iters >= 1

    def test_extra_grad_branch(self) -> None:
        """extra_grad callback is exercised and result still lies on simplex."""
        rng = np.random.default_rng(7)
        mat = rng.standard_normal((5, 3))
        vec = rng.standard_normal(5)
        alpha = 0.3
        target = np.eye(3)

        def extra_grad(v):
            """Return alpha * target @ v."""
            return alpha * (target @ v)

        result, _ = fista_gradient(mat, vec, extra_grad=extra_grad)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        assert np.all(result >= -1e-10)

    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_various_sizes(self, N: int) -> None:  # noqa: N803
        """Agreement with prox_gradient for several problem sizes."""
        rng = np.random.default_rng(N)
        mat = rng.standard_normal((3 * N, N))
        vec = rng.standard_normal(3 * N)
        w_fista, _ = fista_gradient(mat, vec, eps_rel=1e-8)
        w_prox, _ = prox_gradient(mat, vec, eps_rel=1e-8)
        np.testing.assert_allclose(w_fista, w_prox, atol=1e-4)
