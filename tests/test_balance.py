"""Tests for balance systems (B w = c) on the shrinking active-set solver."""

import numpy as np
import pytest

from fast_minimum_variance import Problem
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


def _simulate_equity_returns(n, T, *, rng=None):  # noqa: N803
    """Demeaned (T, n) return matrix from a market + sparse-style factor model."""
    rng = np.random.default_rng(rng)
    k = max(3, n // 10)
    factor_vols = np.concatenate([[0.01], np.full(k - 1, 0.005)])
    f = rng.standard_normal((T, k)) * factor_vols
    b = np.zeros((n, k))
    b[:, 0] = rng.uniform(0.4, 0.8, size=n)
    for j in range(1, k):
        mask = rng.random(n) < 0.5
        b[mask, j] = rng.standard_normal(int(mask.sum())) * 0.2
    idio_vols = rng.uniform(0.005, 0.015, size=n)
    e = rng.standard_normal((T, n)) * idio_vols
    x = f @ b.T + e
    return x - x.mean(axis=0)


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
    return _simulate_equity_returns(60, 500, rng=42)


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

    def test_cg_same_iteration_counts(self, X):  # noqa: N803
        """The single-constraint CG path takes the same outer/inner counts."""
        n = X.shape[1]
        w0, outer0, inner0 = MinVarProblem(X).solve_cg()
        w1, outer1, inner1 = MinVarProblem(X, B=np.ones((1, n)), c=np.array([1.0])).solve_cg()
        assert (outer1, inner1) == (outer0, inner0)
        np.testing.assert_allclose(w1, w0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sleeve systems against the reference oracle
# ---------------------------------------------------------------------------


class TestSleeves:
    """p=4 sleeve systems solved by every production path."""

    def test_cg_matches_reference(self, X, sleeves, lw, reference_weights):  # noqa: N803
        """solve_cg reaches the reference-oracle objective and is exactly feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_cg, _outer, inner = prob.solve_cg()
        w_ref = reference_weights(prob)

        assert inner > 0
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8
        assert w_cg.min() > -1e-6
        # CG is at least as good as the independent oracle (which is itself only
        # approximately optimal near the long-only boundary on this universe).
        assert _objective(prob, w_cg) <= _objective(prob, w_ref) + 1e-9

    def test_no_shrinkage_active_set_shrinks(self, X, sleeves):  # noqa: N803
        """Without shrinkage some assets are eliminated and feasibility holds."""
        b_eq, c_eq = sleeves
        w, outer, _inner = MinVarProblem(X, B=b_eq, c=c_eq).solve_cg()
        assert outer > 1
        assert (w > 1e-8).sum() < X.shape[1]
        assert np.abs(b_eq @ w - c_eq).max() < 1e-8

    def test_projection_is_identity_for_balance(self, X, sleeves, lw):  # noqa: N803
        """project=True must not renormalise a balance-system solution."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_proj, *_ = prob.solve_cg(project=True)
        w_raw, *_ = prob.solve_cg(project=False)
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
            """Backend exposing ``restricted`` so it should never touch ``apply_free``."""

            def restricted(self, free):
                """Return a sub-operator over ``free``, counting the call."""
                calls["restricted"] += 1
                sub = np.diag([1.0, 2.0, 3.0])
                return type("_Sub", (), {"matvec": staticmethod(lambda v: sub @ v)})()

            def apply_free(self, free, v):  # pragma: no cover - must not be reached
                """Fail loudly: ``restricted`` should be preferred over this path."""
                calls["apply_free"] += 1
                raise AssertionError

        f = MinVarProblem._free_matvec(_Restrictable(), idx)
        np.testing.assert_allclose(f(np.ones(3)), [1.0, 2.0, 3.0])
        assert calls == {"restricted": 1, "apply_free": 0}

    def test_falls_back_to_apply_free(self, X):  # noqa: N803
        """A backend without ``restricted`` falls back to per-call ``apply_free``."""

        class _Legacy:
            """Backend without ``restricted``; only the ``apply_free`` path exists."""

            def apply_free(self, free, v):
                """Scale the free sub-vector by two."""
                return 2.0 * v

        f = MinVarProblem._free_matvec(_Legacy(), np.array([0, 1]))
        np.testing.assert_allclose(f(np.array([1.0, 3.0])), [2.0, 6.0])

    def test_falls_back_when_restricted_not_implemented(self, X):  # noqa: N803
        """A backend whose ``restricted`` raises NotImplementedError falls back."""

        class _Partial:
            """Backend whose ``restricted`` is declared but not implemented."""

            def restricted(self, free):
                """Signal that restriction is unsupported so the fallback is used."""
                raise NotImplementedError

            def apply_free(self, free, v):
                """Scale the free sub-vector by three."""
                return 3.0 * v

        f = MinVarProblem._free_matvec(_Partial(), np.array([0]))
        np.testing.assert_allclose(f(np.array([2.0])), [6.0])


class TestSleevesWithTilt:
    """Markowitz tilt combined with a sleeve system."""

    def test_cg_matches_reference(self, X, sleeves, lw, reference_weights):  # noqa: N803
        """Tilted sleeve solve agrees with the reference oracle."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        mu = np.random.default_rng(1).standard_normal(X.shape[1]) * 0.01
        prob = MinVarProblem(X, B=b_eq, c=c_eq, alpha=alpha, target=target, rho=0.5, mu=mu)
        w_cg, _, _ = prob.solve_cg()
        w_ref = reference_weights(prob)
        np.testing.assert_allclose(w_cg, w_ref, atol=1e-5)
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8


# ---------------------------------------------------------------------------
# RMT low-rank target (alpha=1) with sleeves
# ---------------------------------------------------------------------------


class TestSleevesLowRank:
    """Balance systems through the Woodbury low-rank path."""

    def test_lowrank_matches_dense(self, X, sleeves, rmt):  # noqa: N803
        """alpha=1 with target_lr equals the dense-target solve on sleeves."""
        b_eq, c_eq = sleeves
        bar_lam, u_k, delta_k = rmt
        dense = bar_lam * np.eye(X.shape[1]) + (u_k * delta_k) @ u_k.T
        w_lr, *_ = MinVarProblem(X, B=b_eq, c=c_eq, alpha=1.0, target_lr=rmt).solve_cg()
        w_dense, *_ = MinVarProblem(X, B=b_eq, c=c_eq, alpha=1.0, target=dense).solve_cg()
        np.testing.assert_allclose(w_lr, w_dense, atol=1e-6)
        assert np.abs(b_eq @ w_lr - c_eq).max() < 1e-8
