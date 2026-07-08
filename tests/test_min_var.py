"""Consolidated test suite for the long-only minimum-variance solver.

Merges the former ``conftest.py``, ``test_small.py``, ``test_minvar_problem.py``,
``test_cg.py`` and ``test_balance.py`` into a single module:

* shared fixtures (``resource_dir``, ``reference_weights``) and the
  ``make_returns`` helper;
* a small hand-verifiable three-asset worked example for the primal-dual loop;
* unit tests for ``_MinVarProblem`` (defaults, validation, projection,
  active-set loop, ``solve_cg``, low-rank ``target_lr``);
* cross-validation of the CG solver against an independent SLSQP oracle
  (plain / shrinkage / return-tilt / sizes / dense- and low-rank targets);
* balance-system (``B w = c``) tests across every production path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import minimize

from fast_minimum_variance import Problem
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem

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
    """Return an independent long-only min-var oracle (SLSQP), for cross-validation.

    Solves the same objective as ``_MinVarProblem`` — ``(1-alpha)||Xw||^2/T +
    alpha*w^T T0 w - rho*mu^T w`` subject to ``Bw = c`` (or the budget) and
    ``w >= 0`` — with SciPy's SLSQP, sharing no code with the library's
    active-set solvers. It replaces the former CVXPY reference and agrees with
    the direct KKT solve to ~1e-7 on the covered cases.
    """

    def _reference(prob: object) -> np.ndarray:
        """Return the SLSQP-optimal long-only weights for ``prob``."""
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
            bounds=[(0.0, None)] * n,
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
# Small worked example for the primal-dual loop in _MinVarProblem
# ===========================================================================
#
# Three-asset problem designed for hand-verification:
#
#     X = [[1, 0, 1],    X^T X = [[1, 0, 1],
#          [0, 1, 1],              [0, 1, 1],
#          [0, 0, 1]]              [1, 1, 3]]
#
# Equality-constrained optimum (no long-only): w = [2/3, 2/3, -1/3].
# Long-only optimum:                           w* = [1/2, 1/2,   0].
#
# Primal-dual trace:
#   Iteration 1 — solve on {0,1,2}: w = [2/3, 2/3, -1/3]; w[2] < 0 → drop.
#   Iteration 2 — solve on {0,1}:   w = [1/2, 1/2]; all non-negative.
#   Dual check:   grad = [1, 1, 2], lambda_ = 1, nu = [0, 0, 1] >= 0 → done.

# X s.t. X^T X = [[1,0,1],[0,1,1],[1,1,3]] (Cholesky factor transposed).
X3 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
W_OPT = np.array([0.5, 0.5, 0.0])


# ---------------------------------------------------------------------------
# Analytic checks on the problem structure
# ---------------------------------------------------------------------------


def test_covariance():
    """X^T X equals the intended matrix."""
    np.testing.assert_array_equal(X3.T @ X3, [[1, 0, 1], [0, 1, 1], [1, 1, 3]])


def test_dual_variable_at_optimum():
    """At w*=[1/2,1/2,0]: nu_2 = grad_2 - lambda_ = 2 - 1 = 1 > 0."""
    grad = 2 * (X3.T @ X3) @ W_OPT  # [1, 1, 2]
    active = W_OPT > 0
    lambda_ = np.median(grad[active])  # 1.0
    nu = grad - lambda_  # [0, 0, 1]
    assert nu[2] == pytest.approx(1.0)
    assert np.all(nu[~active] >= 0)


# ---------------------------------------------------------------------------
# Primal-dual loop behaviour
# ---------------------------------------------------------------------------


def test_known_optimum():
    """CG solver recovers the known long-only optimum [1/2, 1/2, 0]."""
    w, *_ = MinVarProblem(X3).solve_cg()
    np.testing.assert_allclose(w, W_OPT, atol=1e-6)


def test_two_outer_iterations():
    """Primal step fires once (asset 2 dropped); dual check passes immediately."""
    p = MinVarProblem(X3)
    calls = []

    def counting_cg(active):
        """Record each active-set mask, then delegate to the real _cg_step."""
        calls.append(active.copy())
        return p._cg_step(active)

    p._constraint_active_set(counting_cg)

    assert len(calls) == 2
    assert calls[0].all()  # iteration 1: full active set
    assert not calls[1][2]  # iteration 2: asset 2 excluded


def test_dual_readd():
    """Dual step re-adds an excluded asset when nu_i < 0.

    X = I_3, optimal = [1/3, 1/3, 1/3].  A mock solve_fn forces asset 2 to be
    dropped in the primal step, then returns [1/2, 1/2] on the reduced active
    set {0, 1}.  The dual check computes nu_2 = 0 - 1 = -1 < 0 and re-adds
    asset 2.  The final solve on the full active set returns [1/3, 1/3, 1/3].
    """
    p = MinVarProblem(np.eye(3))
    call_no = [0]

    def solve_fn(active):
        """Force a primal drop of asset 2, then a dual re-add, per the trace."""
        call_no[0] += 1
        if call_no[0] == 1:
            return np.array([0.45, 0.45, -0.1]), 1  # w[2] < 0 → primal drop
        if call_no[0] == 2:
            return np.array([0.5, 0.5]), 1  # nu_2 = 0-1 = -1 → re-add
        return np.ones(active.sum()) / active.sum(), 1

    w, *_ = p._constraint_active_set(solve_fn)

    assert call_no[0] == 3
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


# ===========================================================================
# _MinVarProblem unit tests
# ===========================================================================


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

    def test_n_rectangular(self):
        """N equals the column count for a non-square matrix."""
        assert MinVarProblem(np.ones((20, 7))).n == 7


# ---------------------------------------------------------------------------
# Shape validation (target / target_lr)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _clip_and_renormalize
# ---------------------------------------------------------------------------


class TestClipAndRenormalize:
    """The budget-simplex projection applied by solve_cg(project=True)."""

    def test_clips_negatives_and_sums_to_one(self):
        """Negative weights are clipped to zero and the result sums to 1."""
        w = MinVarProblem(np.eye(3))._clip_and_renormalize(np.array([-0.2, 0.5, 0.7]))
        assert w[0] == 0.0
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)

    def test_already_valid_unchanged(self):
        """A valid weight vector is returned unchanged."""
        w_in = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(MinVarProblem(np.eye(3))._clip_and_renormalize(w_in.copy()), w_in)


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
            """Return a weakly negative weight first, then defer to _cg_step."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-5e-6, 0.6, 0.5 + 5e-6]), 1
            return p._cg_step(mask)

        w, *_ = p._constraint_active_set(solve_fn)
        assert w[0] == pytest.approx(0.0)
        assert w.shape == (3,)

    def test_return_tilt_gradient(self):
        """With rho != 0 the gradient is adjusted by -rho*mu in the dual check."""
        X = make_returns(100, 5, seed=7)  # noqa: N806
        mu = np.ones(5) / 5
        w, *_ = MinVarProblem(X, rho=0.1, mu=mu).solve_cg()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-6)

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

    # Dual re-add (excluded asset comes back when nu_i < 0) is covered by the
    # explicit white-box trace in test_dual_readd.


# ---------------------------------------------------------------------------
# Solver end-to-end tests
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

    def test_project_false(self, mvp):
        """project=False returns raw CG solution without clipping."""
        w, *_ = mvp.solve_cg(project=False)
        assert w.shape == (mvp.n,)

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
        """solve_cg with target_lr returns a valid portfolio."""
        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        target_lr = _make_target_lr(N)
        w, *_ = MinVarProblem(X, alpha=alpha, target_lr=target_lr).solve_cg()
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
        prob = Problem(X, alpha=N / (N + T))
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
        prob = Problem(X, alpha=N / (N + T), rho=0.3, mu=mu)
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

    def test_weights_are_valid(self):
        """Low-rank solution sums to 1 and is non-negative."""
        _, prob = _build_rmt_problem()
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-6).all()

    def test_dense_target_without_target_lr(self):
        """alpha=1 with only a dense target (no target_lr) solves via the Gram path."""
        X = make_returns(T=300, N=20, seed=7)  # noqa: N806
        target, _, _k, _ = rmt_target_and_alpha(X)
        prob = Problem(X, alpha=1.0, target=target)  # no target_lr
        w, *_ = prob.solve_cg()
        assert abs(w.sum() - 1.0) < 1e-6


# ===========================================================================
# Balance systems (B w = c) on the shrinking active-set solver
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
def X_bal():  # noqa: N802
    """Factor-model return matrix (500, 60) so the long-only constraint binds."""
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


# ---------------------------------------------------------------------------
# Validation (B, c)
# ---------------------------------------------------------------------------


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
        """The factory returns the shrinking active-set solver for (B, c)."""
        n = X_bal.shape[1]
        prob = Problem(X_bal, B=np.ones((1, n)), c=np.array([1.0]))
        assert isinstance(prob, MinVarProblem)


# ---------------------------------------------------------------------------
# Budget equivalence: B = ones row reproduces the default exactly
# ---------------------------------------------------------------------------


class TestBudgetEquivalence:
    """An explicit ones-row budget matches the default budget path."""

    def test_cg_same_iteration_counts(self, X_bal):  # noqa: N803
        """The single-constraint CG path takes the same outer/inner counts."""
        n = X_bal.shape[1]
        w0, outer0, inner0 = MinVarProblem(X_bal).solve_cg()
        w1, outer1, inner1 = MinVarProblem(X_bal, B=np.ones((1, n)), c=np.array([1.0])).solve_cg()
        assert (outer1, inner1) == (outer0, inner0)
        np.testing.assert_allclose(w1, w0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sleeve systems against the reference oracle
# ---------------------------------------------------------------------------


class TestSleeves:
    """p=4 sleeve systems solved by every production path."""

    def test_cg_matches_reference(self, X_bal, sleeves, lw, reference_weights):  # noqa: N803
        """solve_cg reaches the reference-oracle objective and is exactly feasible."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_cg, _outer, inner = prob.solve_cg()
        w_ref = reference_weights(prob)

        assert inner > 0
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8
        assert w_cg.min() > -1e-6
        # CG is at least as good as the independent oracle (which is itself only
        # approximately optimal near the long-only boundary on this universe).
        assert _objective(prob, w_cg) <= _objective(prob, w_ref) + 1e-9

    def test_no_shrinkage_active_set_shrinks(self, X_bal, sleeves):  # noqa: N803
        """Without shrinkage some assets are eliminated and feasibility holds."""
        b_eq, c_eq = sleeves
        w, outer, _inner = MinVarProblem(X_bal, B=b_eq, c=c_eq).solve_cg()
        assert outer > 1
        assert (w > 1e-8).sum() < X_bal.shape[1]
        assert np.abs(b_eq @ w - c_eq).max() < 1e-8

    def test_projection_is_identity_for_balance(self, X_bal, sleeves, lw):  # noqa: N803
        """project=True must not renormalise a balance-system solution."""
        b_eq, c_eq = sleeves
        alpha, target = lw
        prob = MinVarProblem(X_bal, B=b_eq, c=c_eq, alpha=alpha, target=target)
        w_proj, *_ = prob.solve_cg(project=True)
        w_raw, *_ = prob.solve_cg(project=False)
        np.testing.assert_array_equal(w_proj, w_raw)


# ---------------------------------------------------------------------------
# Free-block matvec
# ---------------------------------------------------------------------------


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

        f = MinVarProblem._free_matvec(_Restrictable(), idx)
        np.testing.assert_allclose(f(np.ones(3)), [1.0, 2.0, 3.0])
        assert calls == {"restricted": 1, "apply_free": 0}

    def test_falls_back_to_apply_free(self):
        """A backend without ``restricted`` falls back to per-call ``apply_free``."""

        class _Legacy:
            """Backend without ``restricted``; only the ``apply_free`` path exists."""

            def apply_free(self, free, v):
                """Scale the free sub-vector by two."""
                return 2.0 * v

        f = MinVarProblem._free_matvec(_Legacy(), np.array([0, 1]))
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

        f = MinVarProblem._free_matvec(_Partial(), np.array([0]))
        np.testing.assert_allclose(f(np.array([2.0])), [6.0])


# ---------------------------------------------------------------------------
# Return tilt (rho > 0) with sleeves
# ---------------------------------------------------------------------------


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
        np.testing.assert_allclose(w_cg, w_ref, atol=1e-5)
        assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8


# ---------------------------------------------------------------------------
# RMT low-rank target (alpha=1) with sleeves
# ---------------------------------------------------------------------------


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
