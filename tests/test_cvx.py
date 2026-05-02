"""Tests for Problem.solve_cvxpy."""

import numpy as np
import pytest

from fast_minimum_variance.problem import Problem


class TestSolveCvxpy:
    """Tests for Problem.solve_cvxpy."""

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_cvxpy()
        assert w.shape == (problem.X.shape[1],)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_cvxpy()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, problem_small):
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt, _ = problem_small.solve_kkt()
        w_cvxpy, _ = problem_small.solve_cvxpy()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X, problem):  # noqa: N803
        """Optimal portfolio has variance ≤ equal-weight portfolio."""
        w_opt, _ = problem.solve_cvxpy()
        N = X.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_term_tilts_weights(self, X, problem):  # noqa: N803
        """With rho > 0 the solution tilts toward the highest-return asset."""
        mu = np.zeros(X.shape[1])
        mu[-1] = 1.0
        w_mv, _ = problem.solve_cvxpy()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_cvxpy()
        assert w_mk[-1] > w_mv[-1]

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        X = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(X).solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)

    def test_alpha_branch(self):
        """Alpha != 0 enters the Ledoit-Wolf ridge regularisation branch."""
        X = np.eye(3)  # noqa: N806
        w, _ = Problem(X, alpha=0.1).solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_import_error_without_cvxpy(self, monkeypatch):
        """solve_cvxpy raises ImportError when cvxpy is unavailable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("no cvxpy")  # noqa: TRY003
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="cvxpy is required"):
            Problem(np.eye(3)).solve_cvxpy()

    def test_runtime_error_on_infeasible(self):
        """solve_cvxpy raises RuntimeError when the problem is infeasible."""
        X = np.eye(2)  # noqa: N806
        # C^T w <= d with d = [-1, -1] forces w[i] <= -1, which contradicts sum(w)=1.
        C = np.eye(2)  # noqa: N806
        d = np.array([-1.0, -1.0])
        with pytest.raises(RuntimeError, match="CVXPY solver failed"):
            Problem(X, C=C, d=d).solve_cvxpy()
