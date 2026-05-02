"""Tests for MinVarProblem — primal-dual outer loop solver for long-only min-var."""

import numpy as np
import pytest

from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem

# we have some cross-validation tests that need to access the _Problem class
from fast_minimum_variance.problem import _Problem as Problem

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
# _kkt_reduced
# ---------------------------------------------------------------------------


class TestKktReduced:
    """Tests for MinVarProblem._kkt_reduced."""

    def test_full_active_shape(self, mvp_small):
        """KKT matrix is (n+1) × (n+1) when all assets are active."""
        mask = np.ones(mvp_small.n, dtype=bool)
        K, rhs = mvp_small._kkt_reduced(mask)  # noqa: N806
        assert K.shape == (mvp_small.n + 1, mvp_small.n + 1)
        assert rhs.shape == (mvp_small.n + 1,)

    def test_reduced_shape(self):
        """Excluding one asset shrinks the system by one row/col."""
        X = np.eye(4)  # noqa: N806
        p = MinVarProblem(X)
        mask = np.array([True, True, False, True])
        K, _ = p._kkt_reduced(mask)  # noqa: N806
        assert K.shape == (4, 4)  # n_a=3, so (3+1)x(3+1)

    def test_symmetry(self, mvp_small):
        """KKT matrix is symmetric."""
        mask = np.ones(mvp_small.n, dtype=bool)
        K, _ = mvp_small._kkt_reduced(mask)  # noqa: N806
        np.testing.assert_allclose(K, K.T)

    def test_hessian_block_psd(self, mvp_small):
        """Top-left (n_a, n_a) Hessian block is positive semi-definite."""
        mask = np.ones(mvp_small.n, dtype=bool)
        K, _ = mvp_small._kkt_reduced(mask)  # noqa: N806
        n_a = mvp_small.n
        assert np.all(np.linalg.eigvalsh(K[:n_a, :n_a]) >= -1e-10)

    def test_constraint_block_is_ones(self, mvp_small):
        """Off-diagonal block is a ones vector (budget constraint)."""
        mask = np.ones(mvp_small.n, dtype=bool)
        K, _ = mvp_small._kkt_reduced(mask)  # noqa: N806
        n_a = mvp_small.n
        np.testing.assert_array_equal(K[:n_a, n_a], 1.0)
        np.testing.assert_array_equal(K[n_a, :n_a], 1.0)

    def test_rhs_last_entry_is_one(self, mvp_small):
        """The dual RHS entry (budget constraint) equals 1."""
        mask = np.ones(mvp_small.n, dtype=bool)
        _, rhs = mvp_small._kkt_reduced(mask)
        assert rhs[-1] == pytest.approx(1.0)

    def test_rhs_primal_zeros_no_return(self, mvp_small):
        """Primal RHS is zero when rho=0."""
        mask = np.ones(mvp_small.n, dtype=bool)
        _, rhs = mvp_small._kkt_reduced(mask)
        np.testing.assert_array_equal(rhs[: mvp_small.n], 0.0)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS equals rho * mu[active]."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        p = MinVarProblem(X, rho=0.5, mu=mu)
        mask = np.ones(3, dtype=bool)
        _, rhs = p._kkt_reduced(mask)
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)


# ---------------------------------------------------------------------------
# _kkt_operator_reduced
# ---------------------------------------------------------------------------


class TestKktOperatorReduced:
    """Tests for MinVarProblem._kkt_operator_reduced."""

    def test_shape_matches_explicit(self, mvp_small):
        """Operator shape equals the explicit KKT matrix shape."""
        mask = np.ones(mvp_small.n, dtype=bool)
        op, _ = mvp_small._kkt_operator_reduced(mask)
        K, _ = mvp_small._kkt_reduced(mask)  # noqa: N806
        assert op.shape == K.shape

    def test_matvec_matches_explicit(self, mvp_small):
        """Matrix-free matvec equals the explicit KKT matrix-vector product."""
        mask = np.ones(mvp_small.n, dtype=bool)
        op, _ = mvp_small._kkt_operator_reduced(mask)
        K, _ = mvp_small._kkt_reduced(mask)  # noqa: N806
        x = np.ones(mvp_small.n + 1)
        np.testing.assert_allclose(op @ x, K @ x, atol=1e-10)

    def test_rhs_matches_kkt_rhs(self, mvp_small):
        """RHS from operator builder matches explicit KKT RHS."""
        mask = np.ones(mvp_small.n, dtype=bool)
        _, rhs_op = mvp_small._kkt_operator_reduced(mask)
        _, rhs_kkt = mvp_small._kkt_reduced(mask)
        np.testing.assert_allclose(rhs_op, rhs_kkt)

    def test_reduced_mask_shape(self):
        """Operator correctly handles a sub-problem with fewer active assets."""
        X = np.eye(4)  # noqa: N806
        p = MinVarProblem(X)
        mask = np.array([True, True, False, True])  # n_a = 3
        op, rhs = p._kkt_operator_reduced(mask)
        assert op.shape == (4, 4)
        assert rhs.shape == (4,)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS equals rho * mu[active]."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        p = MinVarProblem(X, rho=0.5, mu=mu)
        mask = np.ones(3, dtype=bool)
        _, rhs = p._kkt_operator_reduced(mask)
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)


# ---------------------------------------------------------------------------
# _null_space_operator_reduced
# ---------------------------------------------------------------------------


class TestNullSpaceOperatorReduced:
    """Tests for MinVarProblem._null_space_operator_reduced."""

    def test_particular_solution_sums_to_one(self, mvp_small):
        """w0 satisfies the budget constraint: sum(w0) = 1."""
        mask = np.ones(mvp_small.n, dtype=bool)
        _, _, w0, _ = mvp_small._null_space_operator_reduced(mask)
        assert w0.sum() == pytest.approx(1.0, abs=1e-10)

    def test_reconstruct_satisfies_budget(self, mvp_small):
        """reconstruct(v) satisfies sum(w) = 1 for any v."""
        mask = np.ones(mvp_small.n, dtype=bool)
        op, _rhs, _w0, reconstruct = mvp_small._null_space_operator_reduced(mask)
        assert op is not None
        n_free = mvp_small.n - 1
        v = np.random.default_rng(0).standard_normal(n_free)
        w = reconstruct(v)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_fully_determined_returns_none_op(self):
        """Returns (None, None, w0, None) when n_a <= 1."""
        X = np.eye(1)  # noqa: N806
        p = MinVarProblem(X)
        mask = np.ones(1, dtype=bool)
        op, rhs, w0, reconstruct = p._null_space_operator_reduced(mask)
        assert op is None
        assert rhs is None
        assert reconstruct is None
        assert w0.shape == (1,)
        assert w0[0] == pytest.approx(1.0)

    def test_op_shape(self, mvp_small):
        """Null-space operator has shape (n_a-1, n_a-1)."""
        mask = np.ones(mvp_small.n, dtype=bool)
        op, _, _, _ = mvp_small._null_space_operator_reduced(mask)
        n_a = mvp_small.n
        assert op.shape == (n_a - 1, n_a - 1)

    def test_op_symmetric_matvec(self, mvp_small):
        """Null-space operator is symmetric: <u, A v> == <A u, v>."""
        mask = np.ones(mvp_small.n, dtype=bool)
        op, _, _, _ = mvp_small._null_space_operator_reduced(mask)
        rng = np.random.default_rng(7)
        n_free = mvp_small.n - 1
        u, v = rng.standard_normal(n_free), rng.standard_normal(n_free)
        assert u @ (op @ v) == pytest.approx(v @ (op @ u), abs=1e-10)


# ---------------------------------------------------------------------------
# Solver end-to-end tests
# ---------------------------------------------------------------------------


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


class TestSolveMinres:
    """Tests for MinVarProblem.solve_minres."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_minres()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_minres()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_minres()
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, mvp):
        """Iteration count is a positive integer."""
        _, iters = mvp.solve_minres()
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, mvp_small):
        """MINRES solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_minres, _ = mvp_small.solve_minres()
        np.testing.assert_allclose(w_minres, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = MinVarProblem(R).solve_minres()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-10)

    def test_active_set_drops_high_variance_asset(self):
        """The shrinking active-set removes an asset with outsized variance."""
        R = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w, _ = MinVarProblem(R).solve_minres()
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = MinVarProblem(X).solve_minres()
        w_mk, _ = MinVarProblem(X, rho=1.0, mu=mu).solve_minres()
        assert w_mk[4] > w_mv[4]


class TestSolveCg:
    """Tests for MinVarProblem.solve_cg."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, _ = mvp.solve_cg()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, _ = mvp.solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, _ = mvp.solve_cg()
        assert np.all(w >= -1e-10)

    def test_returns_positive_iters(self, mvp):
        """Iteration count is a positive integer."""
        _, iters = mvp.solve_cg()
        assert isinstance(iters, int)
        assert iters > 0

    def test_close_to_kkt(self, mvp_small):
        """CG solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_cg, _ = mvp_small.solve_cg()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = MinVarProblem(R).solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-10)

    def test_single_asset_fast_path(self):
        """n_a == 1 triggers the fast path; sole active asset gets weight 1."""
        rng = np.random.default_rng(42)
        R = rng.standard_normal((50, 1))  # noqa: N806
        w, iters = MinVarProblem(R).solve_cg()
        assert w.shape == (1,)
        assert w[0] == pytest.approx(1.0)
        assert iters == 0

    def test_active_set_drops_high_variance_asset(self):
        """The shrinking active-set removes an asset with outsized variance."""
        R = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w, _ = MinVarProblem(R).solve_cg()
        assert w[2] == pytest.approx(0.0, abs=1e-4)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-4)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the CG solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = MinVarProblem(X).solve_cg()
        w_mk, _ = MinVarProblem(X, rho=1.0, mu=mu).solve_cg()
        assert w_mk[4] > w_mv[4]


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

    def test_minres_agrees_with_problem(self, X_small):  # noqa: N803
        """MINRES solutions from both classes are identical."""
        w_mvp, _ = MinVarProblem(X_small).solve_minres()
        w_prob, _ = Problem(X_small).solve_minres()
        np.testing.assert_allclose(w_mvp, w_prob, atol=1e-4)

    def test_cg_agrees_with_problem(self, X_small):  # noqa: N803
        """CG solutions from both classes are identical."""
        w_mvp, _ = MinVarProblem(X_small).solve_cg()
        w_prob, _ = Problem(X_small).solve_cg()
        np.testing.assert_allclose(w_mvp, w_prob, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Agreement holds with Ledoit-Wolf ridge."""
        T, N = X_small.shape  # noqa: N806
        alpha = N / (N + T)
        w_mvp, _ = MinVarProblem(X_small, alpha=alpha).solve_kkt()
        w_prob, _ = Problem(X_small, alpha=alpha).solve_kkt()
        np.testing.assert_allclose(w_mvp, w_prob, atol=1e-6)
