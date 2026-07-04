"""Tests for balance systems (B w = c) on the shrinking active-set solver."""

import numpy as np
import pytest

from fast_minimum_variance import Problem, simulate_equity_returns
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


def _objective(prob, w):
    """Shrunk portfolio variance ``(1-alpha)/T ||Xw||^2 + alpha w^T target w``."""
    data = (1 - prob.alpha) * w @ (prob.X.T @ (prob.X @ w)) / prob.t
    return float(data + prob.alpha * w @ (prob.target @ w))


def _sleeve_system(n, p, rng):
    """Partition the universe into p sleeves; each holds its proportional share."""
    perm = rng.permutation(n)
    groups = np.array_split(perm, p)
    b_eq = np.zeros((p, n))
    c_eq = np.zeros(p)
    for g, idx in enumerate(groups):
        b_eq[g, idx] = 1.0
        c_eq[g] = len(idx) / n
    return b_eq, c_eq


@pytest.fixture(scope="module")
def X():  # noqa: N802
    """Factor-model return matrix (500, 60) so the long-only constraint binds."""
    x = simulate_equity_returns(60, 500, rng=42)
    return x - x.mean(axis=0)


@pytest.fixture(scope="module")
def sleeves(X):  # noqa: N803
    """A p=4 sleeve system on the 60-asset universe."""
    return _sleeve_system(X.shape[1], 4, np.random.default_rng(0))


@pytest.fixture(scope="module")
def lw(X):  # noqa: N803
    """LW alpha=0.5 with scaled-identity target."""
    t, n = X.shape
    bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (n * t)
    return 0.5, bar_lam * np.eye(n)


@pytest.fixture(scope="module")
def rmt(X):  # noqa: N803
    """A rank-2 RMT-style target (bar_lam, U_k, delta_k) from the top eigenpairs."""
    t = X.shape[0]
    sigma = X.T @ X / t
    lam, u = np.linalg.eigh(sigma)
    bar_lam = float(lam.mean())
    u_k = u[:, -2:]
    delta_k = lam[-2:] - bar_lam
    return bar_lam, u_k, delta_k


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Shape and pairing checks for (B, c)."""

    def test_b_without_c_raises(self, X):  # noqa: N803
        """Supplying B without c is rejected."""
        with pytest.raises(ValueError, match="together"):
            MinVarProblem(X, B=np.ones((1, X.shape[1])))

    def test_c_without_b_raises(self, X):  # noqa: N803
        """Supplying c without B is rejected."""
        with pytest.raises(ValueError, match="together"):
            MinVarProblem(X, c=np.ones(1))

    def test_bad_b_shape_raises(self, X):  # noqa: N803
        """B with the wrong number of columns is rejected."""
        with pytest.raises(ValueError, match="B must have shape"):
            MinVarProblem(X, B=np.ones((2, 3)), c=np.ones(2))

    def test_bad_c_shape_raises(self, X):  # noqa: N803
        """``c`` whose length differs from B's row count is rejected."""
        with pytest.raises(ValueError, match="c must have shape"):
            MinVarProblem(X, B=np.ones((2, X.shape[1])), c=np.ones(3))

    def test_factory_rejects_b_with_custom_constraints(self, X):  # noqa: N803
        """The factory refuses to mix (B, c) with A/b/C/d."""
        n = X.shape[1]
        with pytest.raises(ValueError, match="cannot be combined"):
            Problem(X, A=np.ones((n, 1)), b=np.ones(1), B=np.ones((1, n)), c=np.ones(1))

    def test_factory_routes_balance_to_minvar(self, X):  # noqa: N803
        """The factory returns the shrinking active-set solver for (B, c)."""
        n = X.shape[1]
        prob = Problem(X, B=np.ones((1, n)), c=np.array([1.0]))
        assert isinstance(prob, MinVarProblem)


# ---------------------------------------------------------------------------
# Budget equivalence: B = ones row reproduces the default exactly
# ---------------------------------------------------------------------------


class TestBudgetEquivalence:
    """An explicit ones-row budget matches the default budget path."""

    def test_kkt_identical(self, X):  # noqa: N803
        """solve_kkt with B=1^T, c=[1] equals the default budget solution."""
        n = X.shape[1]
        w0, _ = MinVarProblem(X).solve_kkt()
        w1, _ = MinVarProblem(X, B=np.ones((1, n)), c=np.array([1.0])).solve_kkt()
        np.testing.assert_allclose(w1, w0, atol=1e-12)

    def test_cg_same_iteration_counts(self, X):  # noqa: N803
        """The single-constraint CG path takes the same outer/inner counts."""
        n = X.shape[1]
        w0, outer0, inner0 = MinVarProblem(X).solve_cg()
        w1, outer1, inner1 = MinVarProblem(X, B=np.ones((1, n)), c=np.array([1.0])).solve_cg()
        assert (outer1, inner1) == (outer0, inner0)
        np.testing.assert_allclose(w1, w0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sleeve systems against the CVXPY reference
# ---------------------------------------------------------------------------


class TestSleeves:
    """p=4 sleeve systems solved by every production path."""

    def test_kkt_matches_cvxpy(self, X, sleeves, lw):  # noqa: N803
        """solve_kkt reaches the CVXPY objective and is exactly feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w, _ = prob.solve_kkt()
        w_ref, _ = prob.solve_cvxpy(project=False)

        assert np.abs(b_eq @ w - c_eq).max() < 1e-12
        assert w.min() > -1e-6
        assert _objective(prob, w) <= _objective(prob, w_ref) + 1e-9

    def test_cg_matches_kkt(self, X, sleeves, lw):  # noqa: N803
        """solve_cg agrees with solve_kkt on the sleeve problem."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_kkt, _ = prob.solve_kkt()
        w_cg, _outer, inner = prob.solve_cg()
        assert inner > 0
        np.testing.assert_allclose(w_cg, w_kkt, atol=1e-5)
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8

    def test_no_shrinkage_active_set_shrinks(self, X, sleeves):  # noqa: N803
        """Without shrinkage some assets are eliminated and feasibility holds."""
        b_eq, c_eq = sleeves
        w, outer = MinVarProblem(X, B=b_eq, c=c_eq).solve_kkt()
        assert outer > 1
        assert (w > 1e-8).sum() < X.shape[1]
        assert np.abs(b_eq @ w - c_eq).max() < 1e-12

    def test_clarabel_and_osqp_reach_optimum(self, X, sleeves, lw):  # noqa: N803
        """Direct Clarabel and OSQP reach the active-set objective and stay feasible.

        Interior-point/ADMM iterates keep tiny positive weights where the
        active-set solver gives exact zeros, so the meaningful comparison is on
        objective value and feasibility, not pointwise weights.
        """
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_kkt, _ = prob.solve_kkt()
        opt = _objective(prob, w_kkt)
        for solver in (prob.solve_clarabel, prob.solve_osqp):
            w, _ = solver(project=False)
            assert _objective(prob, w) == pytest.approx(opt, rel=1e-3)
            assert np.abs(b_eq @ w - c_eq).max() < 1e-6

    def test_nnls_reaches_optimum(self, X, sleeves, lw):  # noqa: N803
        """NNLS with weighted balance rows reaches the optimum and stays feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_nn, _ = prob.solve_nnls()
        w_kkt, _ = prob.solve_kkt()
        assert _objective(prob, w_nn) == pytest.approx(_objective(prob, w_kkt), rel=1e-4)
        assert np.abs(b_eq @ w_nn - c_eq).max() < 1e-8

    def test_projection_is_identity_for_balance(self, X, sleeves, lw):  # noqa: N803
        """project=True must not renormalise a balance-system solution."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_proj, _ = prob.solve_kkt(project=True)
        w_raw, _ = prob.solve_kkt(project=False)
        np.testing.assert_array_equal(w_proj, w_raw)


# ---------------------------------------------------------------------------
# Return tilt (rho > 0) with sleeves
# ---------------------------------------------------------------------------


class TestFreeMatvec:
    """The free-block matvec pre-slices via ``restricted`` with an ``apply_free`` fallback."""

    def test_uses_restricted_when_available(self, X):  # noqa: N803
        """A backend exposing ``restricted`` is pre-sliced once, not via apply_free."""
        idx = np.array([0, 2, 4])
        calls = {"restricted": 0, "apply_free": 0}

        class _Restrictable:
            def restricted(self, free):
                calls["restricted"] += 1
                sub = np.diag([1.0, 2.0, 3.0])
                return type("_Sub", (), {"matvec": staticmethod(lambda v: sub @ v)})()

            def apply_free(self, free, v):  # pragma: no cover - must not be reached
                calls["apply_free"] += 1
                raise AssertionError

        f = MinVarProblem._free_matvec(_Restrictable(), idx)
        np.testing.assert_allclose(f(np.ones(3)), [1.0, 2.0, 3.0])
        assert calls == {"restricted": 1, "apply_free": 0}

    def test_falls_back_to_apply_free(self, X):  # noqa: N803
        """A backend without ``restricted`` falls back to per-call ``apply_free``."""

        class _Legacy:
            def apply_free(self, free, v):
                return 2.0 * v

        f = MinVarProblem._free_matvec(_Legacy(), np.array([0, 1]))
        np.testing.assert_allclose(f(np.array([1.0, 3.0])), [2.0, 6.0])

    def test_falls_back_when_restricted_not_implemented(self, X):  # noqa: N803
        """A backend whose ``restricted`` raises NotImplementedError falls back."""

        class _Partial:
            def restricted(self, free):
                raise NotImplementedError

            def apply_free(self, free, v):
                return 3.0 * v

        f = MinVarProblem._free_matvec(_Partial(), np.array([0]))
        np.testing.assert_allclose(f(np.array([2.0])), [6.0])


class TestSleevesWithTilt:
    """Markowitz tilt combined with a sleeve system."""

    def test_kkt_and_cg_match_cvxpy(self, X, sleeves, lw):  # noqa: N803
        """Tilted sleeve solves agree with the CVXPY reference."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        mu = np.random.default_rng(1).standard_normal(X.shape[1]) * 0.01
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target, rho=0.5, mu=mu)
        w_kkt, _ = prob.solve_kkt()
        w_cg, _, _ = prob.solve_cg()
        w_ref, _ = prob.solve_cvxpy(project=False)
        np.testing.assert_allclose(w_kkt, w_ref, atol=1e-5)
        np.testing.assert_allclose(w_cg, w_ref, atol=1e-5)
        assert np.abs(b_eq @ w_kkt - c_eq).max() < 1e-12


# ---------------------------------------------------------------------------
# RMT low-rank target (alpha=1) and PCG with sleeves
# ---------------------------------------------------------------------------


class TestSleevesLowRank:
    """Balance systems through the Woodbury and PCG paths."""

    def test_woodbury_kkt_matches_dense(self, X, sleeves, rmt):  # noqa: N803
        """alpha=1 with target_lr equals the dense-target solve on sleeves."""
        b_eq, c_eq = sleeves
        bar_lam, u_k, delta_k = rmt
        dense = bar_lam * np.eye(X.shape[1]) + (u_k * delta_k) @ u_k.T
        w_lr, _ = MinVarProblem(X, B=b_eq, c=c_eq, alpha=1.0, target_lr=rmt).solve_kkt()
        w_dense, _ = MinVarProblem(X, B=b_eq, c=c_eq, alpha=1.0, target=dense).solve_kkt()
        np.testing.assert_allclose(w_lr, w_dense, atol=1e-10)
        assert np.abs(b_eq @ w_lr - c_eq).max() < 1e-12

    def test_pcg_matches_cg(self, X, sleeves, lw, rmt):  # noqa: N803
        """solve_pcg with an RMT preconditioner agrees with plain CG on sleeves."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target, pcg_lr=rmt)
        w_pcg, _, inner_pcg = prob.solve_pcg()
        w_cg, _, _ = prob.solve_cg()
        assert inner_pcg > 0
        np.testing.assert_allclose(w_pcg, w_cg, atol=1e-5)
