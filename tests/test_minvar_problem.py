"""Tests for MinVarProblem — primal-dual outer loop solver for long-only min-var."""

import numpy as np
import pytest

from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


def _sigma(p, active):
    """Compute the n_a × n_a SPD covariance matrix for the active assets."""
    x_a = p.X[:, active]
    int(active.sum())
    return (1.0 - p.alpha) * (x_a.T @ x_a) + p.alpha * p.target[np.ix_(active, active)]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


def _make_returns(T, N, seed=42):  # noqa: N803
    """Build a (T, N) standard-normal returns matrix from a seeded RNG."""
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
            """Return a weakly negative weight first, then defer to _kkt_step."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-5e-6, 0.6, 0.5 + 5e-6]), 1
            return p._kkt_step(mask)

        w, *_ = p._constraint_active_set(solve_fn)
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
            """Record the first mask seen and always return equal weights."""
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
            """Record each mask seen and always return a feasible solution."""
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
            """Return 3 iters then 2 iters across the primal and dual steps."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 0.6, 0.5]), 3
            return np.array([0.5, 0.5], dtype=float), 2

        _, _, total = p._constraint_active_set(solve_fn)
        assert total == 5

    def test_negative_asset_removed(self):
        """An asset with negative weight is excluded from the second call."""
        # Same X as test_iters_accumulated: excluding asset 0 is dual-feasible.
        X = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # noqa: N806
        p = MinVarProblem(X)
        masks = []

        def solve_fn(mask):
            """Record each mask, returning a negative-weight solution first."""
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
            """Return a negative-weight solution first, then a feasible pair."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 0.6, 0.5]), 1
            return np.array([0.5, 0.5], dtype=float), 1

        w, *_ = p._constraint_active_set(solve_fn)
        assert w[0] == pytest.approx(0.0)
        assert w.shape == (3,)

    def test_dual_step_readds_asset(self):
        """An excluded asset is re-added when the dual feasibility condition fails."""
        # X = np.eye(3): optimal is equal-weight; excluding any asset fails dual check
        # (grad[i]=0 < lambda=2/3 for any excluded asset i with w_i=0).
        X = np.eye(3)  # noqa: N806
        p = MinVarProblem(X)

        w, *_ = p._constraint_active_set(p._kkt_step)
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


class TestSolveCg:
    """Tests for MinVarProblem.solve_cg (matrix-free CG on reduced SPD system)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, *_ = mvp.solve_cg()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights sum to 1."""
        w, *_ = mvp.solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_weights_non_negative(self, mvp):
        """All weights are non-negative."""
        w, *_ = mvp.solve_cg()
        assert np.all(w >= -1e-4)

    def test_close_to_kkt(self, mvp_small):
        """CG solution is close to the exact KKT solution."""
        w_kkt, _ = mvp_small.solve_kkt()
        w_cg, *_ = mvp_small.solve_cg()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_with_shrinkage(self, X_small):  # noqa: N803
        """Shrinkage branch (alpha > 0) agrees with KKT."""
        T, N = X_small.shape  # noqa: N806
        p = MinVarProblem(X_small, alpha=N / (N + T))
        w_cg, *_ = p.solve_cg()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_return_tilt_branch(self, X_small):  # noqa: N803
        """Return-tilt (rho != 0) runs two CG solves."""
        _T, N = X_small.shape  # noqa: N806
        mu = np.random.default_rng(7).standard_normal(N)
        p = MinVarProblem(X_small, rho=0.5, mu=mu)
        w_cg, *_ = p.solve_cg()
        w_kkt, _ = p.solve_kkt()
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-4)

    def test_project_false(self, mvp):
        """project=False returns raw CG solution without clipping."""
        w, *_ = mvp.solve_cg(project=False)
        assert w.shape == (mvp.n,)


# ---------------------------------------------------------------------------
# target_lr (low-rank shrinkage target)
# ---------------------------------------------------------------------------


def _make_target_lr(n, k=2, seed=0):
    """Build a valid (bar_lam, U_k, delta_k) low-rank target triple."""
    rng = np.random.default_rng(seed)
    U_k = rng.standard_normal((n, k))  # noqa: N806
    delta_k = np.abs(rng.standard_normal(k)) + 0.1
    bar_lam = 0.5
    return bar_lam, U_k, delta_k


class TestTargetLr:
    """target_lr (low-rank target) exercises distinct code branches in CG and KKT steps."""

    @pytest.fixture(scope="class")
    @staticmethod
    def X():  # noqa: N802
        """Return a (100, 8) return matrix."""
        return np.random.default_rng(3).standard_normal((100, 8))

    def test_cg_with_target_lr(self, X):  # noqa: N803
        """solve_cg with target_lr returns a valid portfolio."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr).solve_cg()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)

    def test_kkt_with_target_lr(self, X):  # noqa: N803
        """solve_kkt with target_lr returns a valid portfolio."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w, _ = MinVarProblem(X, alpha=alpha, target_lr=target_lr).solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)

    def test_cg_with_target_lr_and_return_tilt(self, X):  # noqa: N803
        """target_lr + rho != 0 exercises the matvec2 c_lr branch."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        mu = np.random.default_rng(5).standard_normal(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr, rho=0.5, mu=mu).solve_cg()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)


# ---------------------------------------------------------------------------
# solve_pcg (PCG with RMT preconditioner)
# ---------------------------------------------------------------------------


def _make_pcg_lr(X, k=2, seed=1):  # noqa: N803
    """Build a valid (bar_lam, U_k, delta_k) RMT preconditioner triple."""
    rng = np.random.default_rng(seed)
    T, N = X.shape  # noqa: N806
    bar_lam = float(np.trace(X.T @ X / T) / N)
    U_k, _ = np.linalg.qr(rng.standard_normal((N, k)))  # noqa: N806
    U_k = U_k[:, :k]  # noqa: N806
    delta_k = np.abs(rng.standard_normal(k)) + 0.1
    return bar_lam, U_k, delta_k


class TestSolvePcg:
    """Tests for MinVarProblem.solve_pcg (PCG with RMT preconditioner)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def X():  # noqa: N802
        """Return a (100, 5) return matrix."""
        return np.random.default_rng(5).standard_normal((100, 5))

    @pytest.fixture(scope="class")
    @staticmethod
    def pcg_lr(X):  # noqa: N803
        """Valid (bar_lam, U_k, delta_k) preconditioner triple for X."""
        return _make_pcg_lr(X)

    def test_raises_without_pcg_lr(self, X):  # noqa: N803
        """solve_pcg raises ValueError when pcg_lr is not set."""
        with pytest.raises(ValueError, match="pcg_lr"):
            MinVarProblem(X).solve_pcg()

    def test_returns_valid_portfolio(self, X, pcg_lr):  # noqa: N803
        """solve_pcg returns weights that sum to 1 and are non-negative."""
        w, outer, inner = MinVarProblem(X, pcg_lr=pcg_lr).solve_pcg()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)
        assert outer >= 1
        assert inner >= 1

    def test_project_false(self, X, pcg_lr):  # noqa: N803
        """project=False skips clip-and-renormalize."""
        w, *_ = MinVarProblem(X, pcg_lr=pcg_lr).solve_pcg(project=False)
        assert w.shape == (X.shape[1],)

    def test_close_to_kkt(self, X, pcg_lr):  # noqa: N803
        """PCG solution agrees with the direct KKT solution."""
        w_pcg, *_ = MinVarProblem(X, pcg_lr=pcg_lr).solve_pcg()
        w_kkt, _ = MinVarProblem(X).solve_kkt()
        np.testing.assert_allclose(w_pcg, w_kkt, atol=1e-4)

    def test_with_target_lr(self, X, pcg_lr):  # noqa: N803
        """target_lr exercises the low-rank system-matvec branch in _pcg_step."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr, pcg_lr=pcg_lr).solve_pcg()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)

    def test_with_dense_target(self, X, pcg_lr):  # noqa: N803
        """Dense target exercises the target_sub branch and _apply_system call in _pcg_step."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w, *_ = MinVarProblem(X, alpha=alpha, target=np.eye(N), pcg_lr=pcg_lr).solve_pcg()
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)
