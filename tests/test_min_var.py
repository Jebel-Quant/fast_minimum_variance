"""Test suite for the global (equality-constrained) minimum-variance solver.

Covers:

* shared fixtures (``resource_dir``, ``reference_weights``) and the
  ``make_returns`` helper;
* a small hand-verifiable three-asset worked example;
* unit tests for ``_MinVarProblem`` (defaults, validation, ``solve_cg``,
  low-rank ``target_lr``);
* cross-validation of the CG solver against an independent SLSQP oracle
  (plain / shrinkage / return-tilt / sizes / dense- and low-rank targets);
* balance-system (``B w = c``) tests across every production path;
* the matrix-free operator layer and property-based invariants.

Weights are sign-unconstrained (global minimum variance): the only hard
invariant is feasibility (``B w = c``, or the budget ``1^T w = 1``); shorts are
allowed, so no non-negativity is asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from fast_minimum_variance import Problem
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.operators import restricted_matvec

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------


def make_returns(T, N, seed=0):  # noqa: N803
    """Generate a T x N matrix of i.i.d. standard normal returns from a seeded RNG."""
    return np.random.default_rng(seed).standard_normal((T, N))


@pytest.fixture(scope="session")
def resource_dir() -> Path:
    """Return the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def reference_weights() -> Callable[[object], np.ndarray]:
    """Return an independent equality-constrained min-var oracle (SLSQP), for cross-validation.

    Solves the same objective as ``_MinVarProblem`` — ``(1-alpha)||Xw||^2/T +
    alpha*w^T T0 w - rho*mu^T w`` subject to ``Bw = c`` (or the budget) with no
    sign constraint — using SciPy's SLSQP, sharing no code with the library's CG
    solver. It agrees with the direct KKT solve to ~1e-7 on the covered cases.
    """

    def _reference(prob: object) -> np.ndarray:
        """Return the SLSQP-optimal equality-constrained weights for ``prob``."""
        x = prob.X  # ty:ignore[unresolved-attribute]
        t, n = x.shape
        alpha = prob.alpha  # ty:ignore[unresolved-attribute]

        if prob.target_lr is not None:  # ty:ignore[unresolved-attribute]
            bar_lam, u_k, delta_k = prob.target_lr  # ty:ignore[unresolved-attribute]

            def target_quad(w: np.ndarray) -> float:
                """Quadratic form ``w^T T0 w`` for the low-rank RMT target."""
                return float(w @ (bar_lam * w + u_k @ (delta_k * (u_k.T @ w))))

            has_target = True
        elif prob.target is not None:  # ty:ignore[unresolved-attribute]
            target = prob.target  # ty:ignore[unresolved-attribute]

            def target_quad(w: np.ndarray) -> float:
                """Quadratic form ``w^T target w`` for the dense target."""
                return float(w @ (target @ w))

            has_target = True
        else:
            has_target = False

        rho = prob.rho  # ty:ignore[unresolved-attribute]
        mu = prob.mu  # ty:ignore[unresolved-attribute]

        def objective(w: np.ndarray) -> float:
            """Portfolio objective: variance (+ shrinkage) minus the return tilt."""
            data = float((x @ w) @ (x @ w)) / t
            value = (1.0 - alpha) * data + alpha * target_quad(w) if has_target else data
            if rho != 0.0 and mu is not None:
                value = value - rho * float(mu @ w)
            return value

        if prob.B is not None:  # ty:ignore[unresolved-attribute]
            b_mat, c_vec = prob.B, prob.c  # ty:ignore[unresolved-attribute]
            constraints = [
                {"type": "eq", "fun": (lambda w, i=i: float(b_mat[i] @ w - c_vec[i]))} for i in range(b_mat.shape[0])
            ]
        else:
            constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

        res = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result: np.ndarray = res.x
        return result

    return _reference


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 5) with a fixed seed."""
    return make_returns(T=100, N=5, seed=7)


@pytest.fixture(scope="session")
def mvp(X):  # noqa: N803
    """MinVarProblem wrapping the session-scoped (200, 10) return matrix."""
    return MinVarProblem(X)


# ===========================================================================
# Small worked example
# ===========================================================================
#
# Three-asset problem designed for hand-verification:
#
#     X = [[1, 0, 1],    X^T X = [[1, 0, 1],
#          [0, 1, 1],              [0, 1, 1],
#          [0, 0, 1]]              [1, 1, 3]]
#
# With (X^T X)^{-1} 1 = [2, 2, -1] and 1^T (X^T X)^{-1} 1 = 3, the global
# minimum-variance portfolio (budget 1^T w = 1, no sign constraint) is
#     w* = [2, 2, -1] / 3 = [2/3, 2/3, -1/3].

# X s.t. X^T X = [[1,0,1],[0,1,1],[1,1,3]] (Cholesky factor transposed).
X3 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
W_GMV = np.array([2 / 3, 2 / 3, -1 / 3])


def test_covariance():
    """X^T X equals the intended matrix."""
    np.testing.assert_array_equal(X3.T @ X3, [[1, 0, 1], [0, 1, 1], [1, 1, 3]])


def test_known_optimum():
    """CG solver recovers the known GMV optimum [2/3, 2/3, -1/3] (asset 2 is short)."""
    w, *_ = MinVarProblem(X3).solve_cg()
    np.testing.assert_allclose(w, W_GMV, atol=1e-6)


def test_short_position_allowed():
    """The unconstrained solution keeps the negative weight instead of clipping it."""
    w, *_ = MinVarProblem(X3).solve_cg()
    assert w[2] < 0
    assert w.sum() == pytest.approx(1.0)


# ===========================================================================
# _MinVarProblem unit tests
# ===========================================================================


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

    def test_n_rectangular(self):
        """N equals the column count for a non-square matrix."""
        assert MinVarProblem(np.ones((20, 7))).n == 7


class TestTargetValidation:
    """__post_init__ rejects mis-shaped target / target_lr arguments."""

    def test_wrong_target_shape_raises(self):
        """A target with the wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="target must be"):
            MinVarProblem(np.eye(3), target=np.eye(4))

    def test_wrong_target_lr_shape_raises(self):
        """A target_lr with mismatched U_k / delta_k shapes raises ValueError."""
        U_k = np.ones((4, 2))  # noqa: N806  # wrong: 4 rows but n=3
        delta_k = np.ones(2)
        with pytest.raises(ValueError, match="target_lr"):
            MinVarProblem(np.eye(3), target_lr=(0.5, U_k, delta_k))


class TestSolveCg:
    """Tests for MinVarProblem.solve_cg (matrix-free CG on the KKT system)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w, *_ = mvp.solve_cg()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights satisfy the budget exactly (sum to 1)."""
        w, *_ = mvp.solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_outer_steps_is_one(self, mvp):
        """There is no outer loop: solve_cg reports a single step."""
        _, outer, inner = mvp.solve_cg()
        assert outer == 1
        assert inner > 0

    # CG-vs-oracle cross-validation (plain / shrinkage / tilt / sizes / low-rank)
    # lives in TestCgVsReference / TestLowRank below.


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
    """target_lr (low-rank target) exercises distinct code branches in the CG step."""

    @pytest.fixture(scope="class")
    @staticmethod
    def X():  # noqa: N802
        """Return a (100, 8) return matrix."""
        return np.random.default_rng(3).standard_normal((100, 8))

    def test_cg_with_target_lr(self, X):  # noqa: N803
        """solve_cg with target_lr returns a budget-feasible portfolio."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr).solve_cg()
        assert abs(w.sum() - 1.0) < 1e-8

    def test_cg_with_target_lr_and_return_tilt(self, X):  # noqa: N803
        """target_lr + rho != 0 exercises the low-rank matvec plus return-tilt branch."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        mu = np.random.default_rng(5).standard_normal(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr, rho=0.5, mu=mu).solve_cg()
        assert abs(w.sum() - 1.0) < 1e-8


# ===========================================================================
# Cross-validation: CG solver vs an independent SLSQP reference
# ===========================================================================


def rmt_target_and_alpha(X):  # noqa: N803
    """Build an RMT-clipped shrinkage target (alpha=1) as solver input.

    Eigenvalues of the sample covariance above the Marchenko-Pastur bulk edge
    are kept; the rest are clipped to bar_lambda. Returns
    ``(target, lr_factors, k, 1.0)`` where ``lr_factors = (bar_lam, U_k, delta_k)``
    feeds the library's low-rank ``target_lr`` matvec path. Pure numpy; kept as a
    test helper so the solver's low-rank branch stays exercised.
    """
    T, n = X.shape  # noqa: N806
    cov = (X.T @ X) / T
    bar_lam = np.trace(cov) / n
    mp_upper = bar_lam * (1.0 + np.sqrt(n / T)) ** 2

    eigs, vecs = np.linalg.eigh(cov)  # ascending order
    signal = eigs > mp_upper
    k = int(signal.sum())
    vecs_k = vecs[:, signal]
    delta_k = eigs[signal] - bar_lam  # (k,) eigenvalue excesses

    target = bar_lam * np.eye(n) + vecs_k @ np.diag(delta_k) @ vecs_k.T
    lr_factors = (float(bar_lam), vecs_k, delta_k)
    return target, lr_factors, k, 1.0


class TestCgVsReference:
    """CG and the independent SLSQP oracle must return the same portfolio."""

    def test_plain_minvar(self, X, reference_weights):  # noqa: N803
        """Plain minimum variance (alpha=0, rho=0)."""
        prob = Problem(X)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_shrinkage(self, X, reference_weights):  # noqa: N803
        """Ledoit-Wolf shrinkage (alpha > 0)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N))
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_return_tilt(self, X, reference_weights):  # noqa: N803
        """Return tilt (rho != 0, mu given)."""
        mu = np.random.default_rng(1).standard_normal(X.shape[1])
        prob = Problem(X, rho=0.5, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_small_problem(self, X_small, reference_weights):  # noqa: N803
        """Small problem (T=100, N=5)."""
        prob = Problem(X_small)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_shrinkage_and_tilt(self, X, reference_weights):  # noqa: N803
        """Shrinkage and return tilt combined."""
        T, N = X.shape  # noqa: N806
        mu = np.ones(N) / N
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N), rho=0.3, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N, reference_weights):  # noqa: N803
        """Agreement holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        prob = Problem(X)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_explicit_target(self, X, reference_weights):  # noqa: N803
        """CG with an explicit target matrix agrees with the oracle (target matvec branch)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N))
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)


def _build_rmt_problem(T=300, N=50, seed=99, rho=0.0, mu=None):  # noqa: N803
    """Return (X, Problem) with alpha=1 and RMT low-rank target."""
    X = make_returns(T=T, N=N, seed=seed)  # noqa: N806
    target, lr_factors, _k, alpha = rmt_target_and_alpha(X)
    assert alpha == 1.0
    return X, Problem(X, alpha=alpha, target=target, target_lr=lr_factors, rho=rho, mu=mu)


class TestLowRank:
    """The alpha=1 low-rank factor path must agree with the reference oracle."""

    def test_minvar_agrees_with_reference(self, reference_weights):
        """alpha=1, RMT target: CG matches the oracle."""
        _, prob = _build_rmt_problem()
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_return_tilt_agrees_with_reference(self, reference_weights):
        """alpha=1, RMT target, return tilt: CG matches the oracle."""
        mu = np.random.default_rng(5).standard_normal(50)
        _, prob = _build_rmt_problem(rho=0.5, mu=mu)
        w_cg, *_ = prob.solve_cg()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_weights_are_budget_feasible(self):
        """Low-rank solution satisfies the budget exactly."""
        _, prob = _build_rmt_problem()
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_dense_target_without_target_lr(self):
        """alpha=1 with only a dense target (no target_lr) solves via the Gram path."""
        X = make_returns(T=300, N=20, seed=7)  # noqa: N806
        target, _, _k, _ = rmt_target_and_alpha(X)
        prob = Problem(X, alpha=1.0, target=target)  # no target_lr
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6


# ===========================================================================
# Balance systems (B w = c)
# ===========================================================================


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


def _portfolio_objective(prob, w):
    """Full solver objective at ``w``: variance (+ shrinkage) minus the return tilt."""
    val = w @ (prob.X.T @ (prob.X @ w)) / prob.t
    if prob.target_lr is not None:
        bar_lam, u_k, delta_k = prob.target_lr
        tq = float(w @ (bar_lam * w + u_k @ (delta_k * (u_k.T @ w))))
        val = (1 - prob.alpha) * val + prob.alpha * tq
    elif prob.target is not None:
        val = (1 - prob.alpha) * val + prob.alpha * float(w @ (prob.target @ w))
    if prob.rho != 0.0 and prob.mu is not None:
        val = val - prob.rho * float(prob.mu @ w)
    return float(val)


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
def X_bal():  # noqa: N802
    """Factor-model return matrix (500, 60)."""
    return _simulate_equity_returns(60, 500, rng=42)


@pytest.fixture(scope="module")
def sleeves(X_bal):  # noqa: N803
    """A p=4 sleeve system on the 60-asset universe."""
    return _sleeve_system(X_bal.shape[1], 4, np.random.default_rng(0))


@pytest.fixture(scope="module")
def lw(X_bal):  # noqa: N803
    """LW alpha=0.5 with scaled-identity target."""
    t, n = X_bal.shape
    bar_lam = float(np.linalg.norm(X_bal, "fro") ** 2) / (n * t)
    return 0.5, bar_lam * np.eye(n)


@pytest.fixture(scope="module")
def rmt(X_bal):  # noqa: N803
    """A rank-2 RMT-style target (bar_lam, U_k, delta_k) from the top eigenpairs."""
    t = X_bal.shape[0]
    sigma = X_bal.T @ X_bal / t
    lam, u = np.linalg.eigh(sigma)
    bar_lam = float(lam.mean())
    u_k = u[:, -2:]
    delta_k = lam[-2:] - bar_lam
    return bar_lam, u_k, delta_k


class TestBalanceValidation:
    """Shape and pairing checks for (B, c)."""

    def test_b_without_c_raises(self, X_bal):  # noqa: N803
        """Supplying B without c is rejected."""
        with pytest.raises(ValueError, match="together"):
            MinVarProblem(X_bal, B=np.ones((1, X_bal.shape[1])))

    def test_c_without_b_raises(self, X_bal):  # noqa: N803
        """Supplying c without B is rejected."""
        with pytest.raises(ValueError, match="together"):
            MinVarProblem(X_bal, c=np.ones(1))

    def test_bad_b_shape_raises(self, X_bal):  # noqa: N803
        """B with the wrong number of columns is rejected."""
        with pytest.raises(ValueError, match="B must have shape"):
            MinVarProblem(X_bal, B=np.ones((2, 3)), c=np.ones(2))

    def test_bad_c_shape_raises(self, X_bal):  # noqa: N803
        """``c`` whose length differs from B's row count is rejected."""
        with pytest.raises(ValueError, match="c must have shape"):
            MinVarProblem(X_bal, B=np.ones((2, X_bal.shape[1])), c=np.ones(3))

    def test_factory_routes_balance_to_minvar(self, X_bal):  # noqa: N803
        """The factory returns the equality-constrained solver for (B, c)."""
        n = X_bal.shape[1]
        prob = Problem(X_bal, B=np.ones((1, n)), c=np.array([1.0]))
        assert isinstance(prob, MinVarProblem)


class TestBudgetEquivalence:
    """An explicit ones-row budget matches the default budget path."""

    def test_cg_same_iteration_counts(self, X_bal):  # noqa: N803
        """The default budget and an explicit ones-row B give the same solve."""
        n = X_bal.shape[1]
        w0, outer0, inner0 = MinVarProblem(X_bal).solve_cg()
        w1, outer1, inner1 = MinVarProblem(X_bal, B=np.ones((1, n)), c=np.array([1.0])).solve_cg()
        assert (outer1, inner1) == (outer0, inner0)
        np.testing.assert_allclose(w1, w0, atol=1e-12)


class TestSleeves:
    """p=4 sleeve systems solved against the reference oracle."""

    def test_cg_matches_reference(self, X_bal, sleeves, lw, reference_weights):  # noqa: N803
        """solve_cg reaches the reference-oracle objective and is exactly feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_cg, _outer, inner = prob.solve_cg()
        w_ref = reference_weights(prob)

        assert inner > 0
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8
        # Unique convex minimum: CG solves it to higher accuracy than the SLSQP
        # oracle (whose large-leverage solution is only approximate), so CG's
        # objective is at least as good.
        assert _portfolio_objective(prob, w_cg) <= _portfolio_objective(prob, w_ref) + 1e-9


class TestFreeMatvec:
    """The free-block matvec pre-slices via ``restricted`` with an ``apply_free`` fallback."""

    def test_uses_restricted_when_available(self):
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

        f = restricted_matvec(_Restrictable(), idx)
        np.testing.assert_allclose(f(np.ones(3)), [1.0, 2.0, 3.0])
        assert calls == {"restricted": 1, "apply_free": 0}

    def test_falls_back_to_apply_free(self):
        """A backend without ``restricted`` falls back to per-call ``apply_free``."""

        class _Legacy:
            """Backend without ``restricted``; only the ``apply_free`` path exists."""

            def apply_free(self, free, v):
                """Scale the free sub-vector by two."""
                return 2.0 * v

        f = restricted_matvec(_Legacy(), np.array([0, 1]))
        np.testing.assert_allclose(f(np.array([1.0, 3.0])), [2.0, 6.0])

    def test_falls_back_when_restricted_not_implemented(self):
        """A backend whose ``restricted`` raises NotImplementedError falls back."""

        class _Partial:
            """Backend whose ``restricted`` is declared but not implemented."""

            def restricted(self, free):
                """Signal that restriction is unsupported so the fallback is used."""
                raise NotImplementedError

            def apply_free(self, free, v):
                """Scale the free sub-vector by three."""
                return 3.0 * v

        f = restricted_matvec(_Partial(), np.array([0]))
        np.testing.assert_allclose(f(np.array([2.0])), [6.0])


class TestSleevesWithTilt:
    """Markowitz tilt combined with a sleeve system."""

    def test_cg_matches_reference(self, X_bal, sleeves, lw, reference_weights):  # noqa: N803
        """Tilted sleeve solve agrees with the reference oracle."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        mu = np.random.default_rng(1).standard_normal(X_bal.shape[1]) * 0.01
        prob = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target, rho=0.5, mu=mu)
        w_cg, _, _ = prob.solve_cg()
        w_ref = reference_weights(prob)
        assert _portfolio_objective(prob, w_cg) <= _portfolio_objective(prob, w_ref) + 1e-9
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8


class TestSleevesLowRank:
    """Balance systems through the Woodbury low-rank path."""

    def test_lowrank_matches_dense(self, X_bal, sleeves, rmt):  # noqa: N803
        """alpha=1 with target_lr equals the dense-target solve on sleeves."""
        b_eq, c_eq = sleeves
        bar_lam, u_k, delta_k = rmt
        dense = bar_lam * np.eye(X_bal.shape[1]) + (u_k * delta_k) @ u_k.T
        w_lr, *_ = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=1.0, target_lr=rmt).solve_cg()
        w_dense, *_ = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=1.0, target=dense).solve_cg()
        np.testing.assert_allclose(w_lr, w_dense, atol=1e-6)
        assert np.abs(b_eq @ w_lr - c_eq).max() < 1e-8


# ===========================================================================
# Property-based invariants (hypothesis)
# ===========================================================================


class TestSolveCgProperties:
    """solve_cg must satisfy its equality constraints for arbitrary well-posed inputs."""

    @pytest.mark.property
    @given(
        n=st.integers(min_value=2, max_value=12),
        t_mult=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_budget_solution_is_feasible(self, n, t_mult, seed):
        """For any random returns matrix the weights satisfy the budget exactly."""
        x = make_returns(T=n * t_mult, N=n, seed=seed)
        w, *_ = Problem(x).solve_cg()
        assert w.shape == (n,)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.property
    @given(
        n=st.integers(min_value=4, max_value=12),
        p=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=40, deadline=None)
    def test_balance_solution_is_feasible(self, n, p, seed):
        """A sleeve balance system stays exactly feasible (B w = c)."""
        rng = np.random.default_rng(seed)
        x = make_returns(T=5 * n, N=n, seed=seed)
        b_eq, c_eq = _sleeve_system(n, min(p, n), rng)
        w, *_ = Problem(x, B=b_eq, c=c_eq).solve_cg()
        assert np.abs(b_eq @ w - c_eq).max() < 1e-6
