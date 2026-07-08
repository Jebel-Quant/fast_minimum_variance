"""Test suite for the global (equality-constrained) minimum-variance solver.

Covers:

* shared fixtures (``resource_dir``, ``reference_weights``) and the
  ``make_returns`` helper;
* a small hand-verifiable three-asset worked example;
* unit tests for ``Problem`` (defaults, validation, ``solve``,
  low-rank ``target_lr``);
* cross-validation of the solver against an independent augmented-KKT reference
  (plain / shrinkage / return-tilt / sizes / dense- and low-rank targets);
* balance-system (``B w = c``) tests across every production path;
* property-based invariants.

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

from fast_minimum_variance import Problem

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
    """Return an independent equality-constrained min-var reference, for cross-validation.

    Solves the same problem as ``Problem`` but via a different linear-algebra
    path: it assembles the full ``(n+p) x (n+p)`` augmented KKT saddle-point
    system

        [[2*Sigma, -B^T], [B, 0]] @ [w; lambda] = [rho*mu; c]

    and solves it with ``np.linalg.solve``, sharing no code with the solver's
    Schur reduction. Pure NumPy — no SciPy.
    """

    def _reference(prob: object) -> np.ndarray:
        """Return the augmented-KKT weights for ``prob``."""
        x = prob.X  # ty:ignore[unresolved-attribute]
        t, n = x.shape
        alpha = prob.alpha  # ty:ignore[unresolved-attribute]

        sigma = (x.T @ x) / t
        if prob.target_lr is not None:  # ty:ignore[unresolved-attribute]
            bar_lam, u_k, delta_k = prob.target_lr  # ty:ignore[unresolved-attribute]
            t0 = bar_lam * np.eye(n) + (u_k * delta_k) @ u_k.T
            sigma = (1.0 - alpha) * sigma + alpha * t0
        elif prob.target is not None:  # ty:ignore[unresolved-attribute]
            sigma = (1.0 - alpha) * sigma + alpha * prob.target  # ty:ignore[unresolved-attribute]

        b_mat = np.ones((1, n)) if prob.B is None else prob.B  # ty:ignore[unresolved-attribute]
        c_vec = np.ones(1) if prob.c is None else prob.c  # ty:ignore[unresolved-attribute]
        p = b_mat.shape[0]

        rho, mu = prob.rho, prob.mu  # ty:ignore[unresolved-attribute]
        grad_rhs = rho * mu if rho != 0.0 and mu is not None else np.zeros(n)

        kkt = np.zeros((n + p, n + p))
        kkt[:n, :n] = 2.0 * sigma
        kkt[:n, n:] = -b_mat.T
        kkt[n:, :n] = b_mat
        sol = np.linalg.solve(kkt, np.concatenate([grad_rhs, c_vec]))
        result: np.ndarray = sol[:n]
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
    """Problem wrapping the session-scoped (200, 10) return matrix."""
    return Problem(X)


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
    """Solver recovers the known GMV optimum [2/3, 2/3, -1/3] (asset 2 is short)."""
    w = Problem(X3).solve()
    np.testing.assert_allclose(w, W_GMV, atol=1e-6)


def test_short_position_allowed():
    """The unconstrained solution keeps the negative weight instead of clipping it."""
    w = Problem(X3).solve()
    assert w[2] < 0
    assert w.sum() == pytest.approx(1.0)


# ===========================================================================
# Problem unit tests
# ===========================================================================


class TestProblemDefaults:
    """Tests for default field values in Problem."""

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
        assert Problem(np.ones((20, 7))).n == 7


class TestTargetValidation:
    """__post_init__ rejects mis-shaped target / target_lr arguments."""

    def test_wrong_target_shape_raises(self):
        """A target with the wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="target must be"):
            Problem(np.eye(3), target=np.eye(4))

    def test_wrong_target_lr_shape_raises(self):
        """A target_lr with mismatched U_k / delta_k shapes raises ValueError."""
        U_k = np.ones((4, 2))  # noqa: N806  # wrong: 4 rows but n=3
        delta_k = np.ones(2)
        with pytest.raises(ValueError, match="target_lr"):
            Problem(np.eye(3), target_lr=(0.5, U_k, delta_k))


class TestSolve:
    """Tests for Problem.solve (dense NumPy solve of the KKT system)."""

    def test_shape(self, mvp):
        """Output weight vector has shape (N,)."""
        w = mvp.solve()
        assert w.shape == (mvp.n,)

    def test_weights_sum_to_one(self, mvp):
        """Weights satisfy the budget exactly (sum to 1)."""
        w = mvp.solve()
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    # solver-vs-oracle cross-validation (plain / shrinkage / tilt / sizes / low-rank)
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
    """target_lr (low-rank target) exercises distinct code branches in the solve."""

    @pytest.fixture(scope="class")
    @staticmethod
    def X():  # noqa: N802
        """Return a (100, 8) return matrix."""
        return np.random.default_rng(3).standard_normal((100, 8))

    def test_cg_with_target_lr(self, X):  # noqa: N803
        """Solve with target_lr returns a budget-feasible portfolio."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w = Problem(X, alpha=alpha, target_lr=target_lr).solve()
        assert abs(w.sum() - 1.0) < 1e-8

    def test_cg_with_target_lr_and_return_tilt(self, X):  # noqa: N803
        """target_lr + rho != 0 exercises the low-rank matvec plus return-tilt branch."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        mu = np.random.default_rng(5).standard_normal(N)
        w = Problem(X, alpha=alpha, target_lr=target_lr, rho=0.5, mu=mu).solve()
        assert abs(w.sum() - 1.0) < 1e-8


# ===========================================================================
# Cross-validation: solver vs an independent augmented-KKT reference
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
    """The solver and the independent augmented-KKT reference must return the same portfolio."""

    def test_plain_minvar(self, X, reference_weights):  # noqa: N803
        """Plain minimum variance (alpha=0, rho=0)."""
        prob = Problem(X)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_shrinkage(self, X, reference_weights):  # noqa: N803
        """Ledoit-Wolf shrinkage (alpha > 0)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N))
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_return_tilt(self, X, reference_weights):  # noqa: N803
        """Return tilt (rho != 0, mu given)."""
        mu = np.random.default_rng(1).standard_normal(X.shape[1])
        prob = Problem(X, rho=0.5, mu=mu)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_small_problem(self, X_small, reference_weights):  # noqa: N803
        """Small problem (T=100, N=5)."""
        prob = Problem(X_small)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_shrinkage_and_tilt(self, X, reference_weights):  # noqa: N803
        """Shrinkage and return tilt combined."""
        T, N = X.shape  # noqa: N806
        mu = np.ones(N) / N
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N), rho=0.3, mu=mu)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    @pytest.mark.parametrize("N", [2, 5, 20])
    def test_various_sizes(self, N, reference_weights):  # noqa: N803
        """Agreement holds for several problem sizes."""
        X = make_returns(T=5 * N, N=N, seed=N)  # noqa: N806
        prob = Problem(X)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_with_explicit_target(self, X, reference_weights):  # noqa: N803
        """The solve with an explicit target matrix agrees with the oracle (target matvec branch)."""
        T, N = X.shape  # noqa: N806
        prob = Problem(X, alpha=N / (N + T), target=np.eye(N))
        w_cg = prob.solve()
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
        """alpha=1, RMT target: the solve matches the oracle."""
        _, prob = _build_rmt_problem()
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_return_tilt_agrees_with_reference(self, reference_weights):
        """alpha=1, RMT target, return tilt: the solve matches the oracle."""
        mu = np.random.default_rng(5).standard_normal(50)
        _, prob = _build_rmt_problem(rho=0.5, mu=mu)
        w_cg = prob.solve()
        np.testing.assert_allclose(w_cg, reference_weights(prob), atol=1e-4)

    def test_weights_are_budget_feasible(self):
        """Low-rank solution satisfies the budget exactly."""
        _, prob = _build_rmt_problem()
        w = prob.solve()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_dense_target_without_target_lr(self):
        """alpha=1 with only a dense target (no target_lr) solves via the Gram path."""
        X = make_returns(T=300, N=20, seed=7)  # noqa: N806
        target, _, _k, _ = rmt_target_and_alpha(X)
        prob = Problem(X, alpha=1.0, target=target)  # no target_lr
        w = prob.solve()
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
            Problem(X_bal, B=np.ones((1, X_bal.shape[1])))

    def test_c_without_b_raises(self, X_bal):  # noqa: N803
        """Supplying c without B is rejected."""
        with pytest.raises(ValueError, match="together"):
            Problem(X_bal, c=np.ones(1))

    def test_bad_b_shape_raises(self, X_bal):  # noqa: N803
        """B with the wrong number of columns is rejected."""
        with pytest.raises(ValueError, match="B must have shape"):
            Problem(X_bal, B=np.ones((2, 3)), c=np.ones(2))

    def test_bad_c_shape_raises(self, X_bal):  # noqa: N803
        """``c`` whose length differs from B's row count is rejected."""
        with pytest.raises(ValueError, match="c must have shape"):
            Problem(X_bal, B=np.ones((2, X_bal.shape[1])), c=np.ones(3))


class TestBudgetEquivalence:
    """An explicit ones-row budget matches the default budget path."""

    def test_ones_row_matches_default(self, X_bal):  # noqa: N803
        """The default budget and an explicit ones-row B give the same weights."""
        n = X_bal.shape[1]
        w0 = Problem(X_bal).solve()
        w1 = Problem(X_bal, B=np.ones((1, n)), c=np.array([1.0])).solve()
        np.testing.assert_allclose(w1, w0, atol=1e-12)


class TestSleeves:
    """p=4 sleeve systems solved against the reference oracle."""

    def test_matches_reference(self, X_bal, sleeves, lw, reference_weights):  # noqa: N803
        """Solve matches the independent augmented-KKT reference and is exactly feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = Problem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_cg = prob.solve()
        w_ref = reference_weights(prob)

        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8
        np.testing.assert_allclose(w_cg, w_ref, rtol=1e-6, atol=1e-8)


class TestSleevesWithTilt:
    """Markowitz tilt combined with a sleeve system."""

    def test_matches_reference(self, X_bal, sleeves, lw, reference_weights):  # noqa: N803
        """Tilted sleeve solve agrees with the reference oracle."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        mu = np.random.default_rng(1).standard_normal(X_bal.shape[1]) * 0.01
        prob = Problem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target, rho=0.5, mu=mu)
        w_cg = prob.solve()
        w_ref = reference_weights(prob)
        np.testing.assert_allclose(w_cg, w_ref, rtol=1e-6, atol=1e-8)
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8


class TestSleevesLowRank:
    """Balance systems through the Woodbury low-rank path."""

    def test_lowrank_matches_dense(self, X_bal, sleeves, rmt):  # noqa: N803
        """alpha=1 with target_lr equals the dense-target solve on sleeves."""
        b_eq, c_eq = sleeves
        bar_lam, u_k, delta_k = rmt
        dense = bar_lam * np.eye(X_bal.shape[1]) + (u_k * delta_k) @ u_k.T
        w_lr = Problem(X_bal, B=b_eq, c=c_eq, alpha=1.0, target_lr=rmt).solve()
        w_dense = Problem(X_bal, B=b_eq, c=c_eq, alpha=1.0, target=dense).solve()
        np.testing.assert_allclose(w_lr, w_dense, atol=1e-6)
        assert np.abs(b_eq @ w_lr - c_eq).max() < 1e-8


# ===========================================================================
# Property-based invariants (hypothesis)
# ===========================================================================


class TestSolveCgProperties:
    """solve must satisfy its equality constraints for arbitrary well-posed inputs."""

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
        w = Problem(x).solve()
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
        w = Problem(x, B=b_eq, c=c_eq).solve()
        assert np.abs(b_eq @ w - c_eq).max() < 1e-6
