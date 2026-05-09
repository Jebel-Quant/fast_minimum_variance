"""Tests for MinVarProblem — primal-dual outer loop solver for long-only min-var."""

import numpy as np
import pytest

from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem

# we have some cross-validation tests that need to access the _Problem class
from fast_minimum_variance.problem import _Problem as Problem


def _sigma(p, active):
    """Compute the n_a × n_a SPD covariance matrix for the active assets."""
    x_a = p.X[:, active]
    int(active.sum())
    return (1.0 - p.alpha) * (x_a.T @ x_a) + p.alpha * p.target[np.ix_(active, active)]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


def _make_returns(T, N, seed=42):  # noqa: N803
    return np.random.default_rng(seed).standard_normal((T, N))


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return _make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 3) for fast solver tests."""
    return _make_returns(T=100, N=3, seed=0)


@pytest.fixture(scope="session")
def mvp(X):  # noqa: N803
    """MinVarProblem wrapping the session-scoped (200, 10) return matrix."""
    return MinVarProblem(X)


@pytest.fixture(scope="session")
def mvp_small(X_small):  # noqa: N803
    """MinVarProblem wrapping the session-scoped (100, 3) return matrix."""
    return MinVarProblem(X_small)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestMinVarProblemDefaults:
    """Tests for default field values in MinVarProblem."""

    def test_n_equals_columns(self, mvp):
        """N equals the number of columns in X."""
        assert mvp.n == mvp.X.shape[1]

    def test_alpha_default(self, mvp):
        """Default alpha is 0.0."""
        assert mvp.alpha == 0.0

    def test_rho_default(self, mvp):
        """Default rho is 0.0."""
        assert mvp.rho == 0.0

    def test_mu_default(self, mvp):
        """Default mu is None."""
        assert mvp.mu is None


# ---------------------------------------------------------------------------
# _constraint_active_set
# ---------------------------------------------------------------------------


class TestConstraintActiveSet:
    """Tests for MinVarProblem._constraint_active_set."""

    def test_weak_negative_single_drop(self):
        """A weakly negative weight (between -tol and -10*tol) uses the single-drop path."""
        X = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # noqa: N806
        p = MinVarProblem(X, target=np.eye(3))
        call_count = [0]

        def solve_fn(mask):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-5e-6, 0.6, 0.5 + 5e-6]), 1
            return p._kkt_step(mask)

        w, _ = p._constraint_active_set(solve_fn)
        assert w[0] == pytest.approx(0.0)
        assert w.shape == (3,)

    def test_return_tilt_gradient(self):
        """With rho != 0 the gradient is adjusted by -rho*mu in the dual check."""
        X = _make_returns(100, 5, seed=7)  # noqa: N806
        mu = np.ones(5) / 5
        w, _ = MinVarProblem(X, rho=0.1, mu=mu).solve_kkt()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-10)

    def test_starts_all_active(self):
        """solve_fn receives all-True mask on the first call."""
        X = np.eye(3)  # noqa: N806
        p = MinVarProblem(X)
        first_mask = []

        def solve_fn(mask):
            if not first_mask:
                first_mask.append(mask.copy())
            return np.array([1 / 3, 1 / 3, 1 / 3]), 1

        p._constraint_active_set(solve_fn)
        assert first_mask[0].all()
        assert first_mask[0].shape == (3,)

    def test_single_call_when_feasible(self):
        """solve_fn is called once when the first solution has no negative weights."""
        X = np.eye(3)  # noqa: N806
        p = MinVarProblem(X)
        calls = []

        def solve_fn(mask):
            calls.append(mask.copy())
            return np.array([0.5, 0.3, 0.2]), 1

        p._constraint_active_set(solve_fn)
        assert len(calls) == 1

    def test_iters_accumulated(self):
        """Total iters is the sum across all solver calls (primal + dual steps)."""
        # X chosen so that excluding asset 0 at w=[0,0.5,0.5] is dual-feasible:
        # X.T@X = [[4,2,2],[2,2,1],[2,1,2]]; grad=[4,3,3], lambda=3 -> grad[0]>=lambda.
        X = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # noqa: N806
        p = MinVarProblem(X)
        call_count = [0]

        def solve_fn(mask):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 0.6, 0.5]), 3
            return np.array([0.5, 0.5], dtype=float), 2

        _, total = p._constraint_active_set(solve_fn)
        assert total == 5

    def test_negative_asset_removed(self):
        """An asset with negative weight is excluded from the second call."""
        # Same X as test_iters_accumulated: excluding asset 0 is dual-feasible.
        X = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # noqa: N806
        p = MinVarProblem(X)
        masks = []

        def solve_fn(mask):
            masks.append(mask.copy())
            if len(masks) == 1:
                return np.array([-0.1, 0.6, 0.5]), 1
            return np.array([0.5, 0.5], dtype=float), 1

        p._constraint_active_set(solve_fn)
        assert len(masks) == 2
        assert not masks[1][0]  # asset 0 dropped
        assert masks[1][1]  # asset 1 retained
        assert masks[1][2]  # asset 2 retained

    def test_zero_weight_padded_correctly(self):
        """Assets excluded from the sub-problem receive weight 0 in the output."""
        X = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # noqa: N806
        p = MinVarProblem(X)
        call_count = [0]

        def solve_fn(mask):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 0.6, 0.5]), 1
            return np.array([0.5, 0.5], dtype=float), 1

        w, _ = p._constraint_active_set(solve_fn)
        assert w[0] == pytest.approx(0.0)
        assert w.shape == (3,)

    def test_dual_step_readds_asset(self):
        """An excluded asset is re-added when the dual feasibility condition fails."""
        # X = np.eye(3): optimal is equal-weight; excluding any asset fails dual check
        # (grad[i]=0 < lambda=2/3 for any excluded asset i with w_i=0).
        X = np.eye(3)  # noqa: N806
        p = MinVarProblem(X)

        w, _ = p._constraint_active_set(p._kkt_step)
        # All assets should be in the final portfolio (equal-weight is optimal).
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# Solver end-to-end tests
# ---------------------------------------------------------------------------


class TestKktStep:
    """Tests for MinVarProblem._kkt_step."""

    def test_rho_nonzero_two_solves(self):
        """With rho != 0 and mu given, _kkt_step performs two SPD solves."""
        X = _make_returns(50, 4, seed=3)  # noqa: N806
        mu = np.array([0.1, 0.2, 0.15, 0.05])
        p = MinVarProblem(X, rho=0.5, mu=mu)
        active = np.ones(4, dtype=bool)
        w_a, iters = p._kkt_step(active)
        assert w_a.shape == (4,)
        assert iters == 1


class TestSolveKkt:
    """Tests for MinVarProblem.solve_kkt."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_kkt()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_kkt()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_kkt()
        assert np.all(w >= -1e-10)

    def test_project_false_preserves_raw(self):
        """project=False skips clip-and-renormalize; result may not sum to 1."""
        X = np.eye(3)  # noqa: N806
        w, _ = MinVarProblem(X).solve_kkt(project=False)
        assert w.shape == (3,)


# ---------------------------------------------------------------------------
# Cross-validation: MinVarProblem agrees with Problem
# ---------------------------------------------------------------------------


class TestSolveClarabel:
    """Tests for _MinVarProblem.solve_clarabel (direct Clarabel API)."""

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
        """Clarabel direct agrees with KKT direct to solver tolerance."""
        w_clar, _ = mvp.solve_clarabel()
        w_kkt, _ = mvp.solve_kkt()
        np.testing.assert_allclose(w_clar, w_kkt, atol=1e-5)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (gamma > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        alpha = N / (N + T)
        p = MinVarProblem(X_small, alpha=alpha)
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


class TestSolveCg:
    """Tests for MinVarProblem.solve_cg (matrix-free CG on reduced SPD system)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_cg()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_cg()
        assert np.all(w >= -1e-4)

    def test_close_to_kkt(self, mvp_small):
        """CG solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_cg, _ = mvp_small.solve_cg()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = MinVarProblem(X_small, alpha=N / (N + T))
        w_cg, _ = p.solve_cg()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) runs two CG solves."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = MinVarProblem(X_small, rho=0.5, mu=mu)
        w_cg, _ = p.solve_cg()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_project_false(self, mvp):
        """project=False returns raw CG solution without clipping."""
        w, _ = mvp.solve_cg(project=False)
        assert w.shape == (mvp.n,)


class TestSolveCvxpy:
    """Tests for MinVarProblem.solve_cvxpy (CVXPY / Clarabel reference solver)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_cvxpy()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_cvxpy()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, mvp_small):
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_cvxpy, _ = mvp_small.solve_cvxpy()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X, mvp):  # noqa: N803
        """Optimal portfolio has variance <= equal-weight portfolio."""
        w_opt, _ = mvp.solve_cvxpy()
        w_eq = np.ones(mvp.n) / mvp.n
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) tilts weights toward high-return assets."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = MinVarProblem(X_small, rho=0.5, mu=mu)
        w_cvxpy, _ = p.solve_cvxpy()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Alpha != 0 enters the Ledoit-Wolf ridge branch."""
        T, N = X_small.shape  # noqa: N806
        p = MinVarProblem(X_small, alpha=N / (N + T))
        w_cvxpy, _ = p.solve_cvxpy()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_project_false(self, mvp):
        """project=False returns raw CVXPY solution without clipping."""
        w, iters = mvp.solve_cvxpy(project=False)
        assert w.shape == (mvp.n,)
        assert iters > 0


class TestSolveNnls:
    """Tests for MinVarProblem.solve_nnls (NNLS on augmented return matrix)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_nnls()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_nnls()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_nnls()
        assert np.all(w >= 0)

    def test_iters_always_one(self, mvp):
        """NNLS is a single solve; iter count is always 1."""
        _, iters = mvp.solve_nnls()
        assert iters == 1

    def test_close_to_kkt(self, mvp_small):
        """NNLS solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_nnls, _ = mvp_small.solve_nnls()
        np.testing.assert_allclose(w_nnls, w_kkt, atol=1e-3)

    def test_project_false_returns_raw(self, mvp_small):
        """project=False returns unnormalized weights that are still non-negative."""
        w_raw, _ = mvp_small.solve_nnls(project=False)
        w_proj, _ = mvp_small.solve_nnls(project=True)
        assert np.all(w_raw >= 0)
        np.testing.assert_allclose(w_proj, w_raw / w_raw.sum(), atol=1e-10)

    def test_with_shrinkage(self, X):  # noqa: N803
        """Shrinkage (alpha > 0) exercises the ridge augmentation rows."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w_nnls, _ = MinVarProblem(X, alpha=alpha).solve_nnls()
        w_kkt, _ = MinVarProblem(X, alpha=alpha).solve_kkt()
        np.testing.assert_allclose(w_nnls, w_kkt, atol=1e-3)


class TestSolveOsqp:
    """Tests for MinVarProblem.solve_osqp (direct OSQP API)."""

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
        """OSQP solution agrees with KKT direct to solver tolerance."""
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
# Cross-validation: MinVarProblem agrees with Problem
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """MinVarProblem and Problem should produce identical portfolios."""

    def test_kkt_agrees_with_problem(self, X_small):  # noqa: N803
        """KKT solutions from both classes are identical."""
        w_mvp, _ = MinVarProblem(X_small).solve_kkt()
        w_prob, _ = Problem(X_small).solve_kkt()
        np.testing.assert_allclose(w_mvp, w_prob, atol=1e-6)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Agreement holds with Ledoit-Wolf ridge."""
        T, N = X_small.shape  # noqa: N806
        alpha = N / (N + T)
        w_mvp, _ = MinVarProblem(X_small, alpha=alpha).solve_kkt()
        w_prob, _ = Problem(X_small, alpha=alpha).solve_kkt()
        np.testing.assert_allclose(w_mvp, w_prob, atol=1e-6)
