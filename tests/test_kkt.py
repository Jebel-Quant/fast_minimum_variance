"""Tests for Problem.solve_kkt."""

import numpy as np
import pytest

from fast_minimum_variance.api import Problem


class TestBuildKkt:
    """Tests for Problem.kkt."""

    def test_shape(self, problem_small):
        """KKT matrix has shape (N+1, N+1) and rhs has shape (N+1,)."""
        K, rhs = problem_small.kkt()  # noqa: N806
        assert K.shape == (problem_small.n + 1, problem_small.n + 1)
        assert rhs.shape == (problem_small.n + 1,)

    def test_rhs(self, problem_small):
        """RHS is zero everywhere except the last entry which equals 1."""
        _, rhs = problem_small.kkt()
        np.testing.assert_array_equal(rhs[: problem_small.n], 0.0)
        assert rhs[problem_small.n] == 1.0

    def test_constraint_row_col(self, problem_small):
        """Last row and column (excluding corner) are all ones."""
        K, _ = problem_small.kkt()  # noqa: N806
        n = problem_small.n
        np.testing.assert_array_equal(K[:n, n], 1.0)
        np.testing.assert_array_equal(K[n, :n], 1.0)
        assert K[n, n] == 0.0

    def test_hessian_block_symmetry(self, problem_small):
        """The (N, N) Hessian block 2 R^T R is symmetric."""
        K, _ = problem_small.kkt()  # noqa: N806
        n = problem_small.n
        np.testing.assert_allclose(K[:n, :n], K[:n, :n].T)

    def test_hessian_block_positive_semidefinite(self, problem_small):
        """The (N, N) Hessian block is positive semi-definite."""
        K, _ = problem_small.kkt()  # noqa: N806
        n = problem_small.n
        eigenvalues = np.linalg.eigvalsh(K[:n, :n])
        assert np.all(eigenvalues >= -1e-10)

    def test_rhs_with_return_term(self):
        """With rho > 0 the first N entries of rhs equal rho * mu."""
        N = 3  # noqa: N806
        X = np.eye(N)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = Problem(X, rho=0.5, mu=mu).kkt()
        np.testing.assert_allclose(rhs[:N], 0.5 * mu)
        assert rhs[N] == 1.0


class TestSolveKkt:
    """Tests for Problem.solve_kkt."""

    def test_iters(self, problem):
        """Direct KKT solver always returns iters==1."""
        _, iters = problem.solve_kkt()
        assert iters == 1

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_kkt()
        assert w.shape == (problem.n,)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_kkt()
        assert np.all(w >= -1e-10)

    def test_trivial_case(self):
        """With a single asset the full weight goes to it."""
        R = np.ones((10, 1))  # noqa: N806
        w, _ = Problem(X=R).solve_kkt()
        np.testing.assert_allclose(w, [1.0], atol=1e-10)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(R).solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = Problem(X).solve_kkt()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_kkt()
        assert w_mk[4] > w_mv[4]

    def test_all_inequalities_active_break(self):
        """Active-set loop terminates when every inequality constraint is binding."""
        X = np.eye(2)  # noqa: N806
        C = np.array([[1.0], [0.0]])  # w1 <= 0.3  # noqa: N806
        d = np.array([0.3])
        w, _ = Problem(X, C=C, d=d).solve_kkt()
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
        w, _ = Problem(X).solve_kkt()
        assert w[2] == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-8)


if __name__ == "__main__":
    pytest.main()
