"""Tests for _BaseProblem shared fields, utilities, and template solvers."""

import sys
from dataclasses import dataclass

import numpy as np
import pytest

from fast_minimum_variance._base import _BaseProblem

# ---------------------------------------------------------------------------
# Minimal concrete stub — implements all abstract hooks with predictable output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stub(_BaseProblem):
    """_BaseProblem subclass that returns fixed, distinguishable values.

    Each ``_XXX_step`` returns a unique iter count (1 / 3 / 5) so tests can
    verify that each ``solve_*`` template calls exactly the right step method.
    """

    def _constraint_active_set(self, solve_fn):
        """Call solve_fn once and report a fixed outer-iteration count of 1."""
        w, step_iters = solve_fn(None)
        return w, 1, step_iters

    def _kkt_step(self, mask):
        """Return canned weights and iter count 1 for the KKT step."""
        return np.array([0.5, -0.1, 0.6]), 1

    def _cg_step(self, mask):
        """Return canned weights and inner iter count 5 for the CG step."""
        return np.array([0.5, -0.1, 0.6]), 5

    def _nnls_solve(self):
        """Return canned weights and iter count 1 for the NNLS solve."""
        return np.array([0.5, -0.1, 0.6]), 1

    def _cvxpy_constraints(self, w, cp):
        """Return the long-only, sum-to-one CVXPY constraints for the stub."""
        return [cp.sum(w) == 1, w >= 0]

    def _clarabel_constraints(self):
        """Return empty Clarabel constraint data for the stub."""
        return None, None, []

    def _osqp_constraints(self):
        """Return empty OSQP constraint data for the stub."""
        return None, None, None


_X3 = np.eye(3)  # minimal 3x3 return matrix for most tests


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestAbstractInterface:
    """_BaseProblem cannot be instantiated; incomplete subclasses are rejected."""

    def test_cannot_instantiate_base_directly(self):
        """Instantiating _BaseProblem directly raises TypeError."""
        with pytest.raises(TypeError):
            _BaseProblem(_X3)  # type: ignore[abstract]

    def test_missing_cvxpy_constraints_raises(self):
        """A subclass missing _cvxpy_constraints cannot be instantiated."""

        @dataclass(frozen=True)
        class _Partial(_BaseProblem):
            """_BaseProblem subclass missing _cvxpy_constraints (still abstract)."""

            def _constraint_active_set(self, fn):
                """Call fn once and return its result."""
                return fn(None)

            def _kkt_step(self, mask):
                """Return zero weights and iter count 1."""
                return np.zeros(3), 1

            # _cvxpy_constraints intentionally omitted

        with pytest.raises(TypeError):
            _Partial(_X3)

    def test_complete_subclass_instantiates(self):
        """A fully-implemented subclass can be instantiated."""
        stub = _Stub(_X3)
        assert stub is not None

    def test_wrong_target_shape_raises(self):
        """A target with the wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="target must be"):
            _Stub(_X3, target=np.eye(4))

    def test_wrong_target_lr_shape_raises(self):
        """A target_lr with mismatched U_k / delta_k shapes raises ValueError."""
        U_k = np.ones((4, 2))  # noqa: N806  # wrong: 4 rows but n=3
        delta_k = np.ones(2)
        with pytest.raises(ValueError, match="target_lr"):
            _Stub(_X3, target_lr=(0.5, U_k, delta_k))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


class TestN:
    """Tests for _BaseProblem.n property."""

    def test_n_equals_columns(self):
        """N equals the number of columns in X."""
        assert _Stub(_X3).n == 3

    def test_n_rectangular(self):
        """N equals the column count for a non-square matrix."""
        assert _Stub(np.ones((20, 7))).n == 7


class TestClipAndRenormalize:
    """Tests for _BaseProblem._clip_and_renormalize."""

    def test_clips_negatives_to_zero(self):
        """Negative weights are clipped to zero."""
        w = _BaseProblem._clip_and_renormalize(np.array([-0.2, 0.5, 0.7]))
        assert w[0] == 0.0

    def test_result_sums_to_one(self):
        """Output sums to 1 after clipping and renormalizing."""
        w = _BaseProblem._clip_and_renormalize(np.array([-0.1, 0.4, 0.7]))
        assert w.sum() == pytest.approx(1.0)

    def test_all_non_negative(self):
        """All output weights are non-negative."""
        w = _BaseProblem._clip_and_renormalize(np.array([-0.5, 0.3, 0.9, -0.1]))
        assert np.all(w >= 0)

    def test_already_valid_unchanged(self):
        """A valid weight vector is returned unchanged."""
        w_in = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(_BaseProblem._clip_and_renormalize(w_in.copy()), w_in)


# ---------------------------------------------------------------------------
# Template solver delegation and project behaviour
# ---------------------------------------------------------------------------


class TestTemplateDelegation:
    """solve_* passes the correct _XXX_step to _constraint_active_set."""

    def test_solve_kkt_uses_kkt_step(self):
        """solve_kkt delegates to _kkt_step (iters==1)."""
        _, iters = _Stub(_X3).solve_kkt()
        assert iters == 1

    def test_solve_cg_uses_cg_step(self):
        """solve_cg delegates to _cg_step (inner iters==5)."""
        _, _, iters = _Stub(_X3).solve_cg()
        assert iters == 5

    def test_solve_nnls_uses_nnls_solve(self):
        """solve_nnls calls _nnls_solve directly (iters==1)."""
        _, iters = _Stub(_X3).solve_nnls()
        assert iters == 1


class TestProjectParameter:
    """project=True clips and renormalizes; project=False returns raw weights."""

    def test_project_true_clips_negative(self):
        """project=True removes negative weights."""
        w, _ = _Stub(_X3).solve_kkt(project=True)
        assert np.all(w >= 0)

    def test_project_true_sums_to_one(self):
        """project=True ensures weights sum to 1."""
        w, _ = _Stub(_X3).solve_kkt(project=True)
        assert w.sum() == pytest.approx(1.0)

    def test_project_false_preserves_negative(self):
        """project=False returns the raw negative weight unchanged."""
        w, _ = _Stub(_X3).solve_kkt(project=False)
        assert w[1] == pytest.approx(-0.1)

    def test_project_default_is_true(self):
        """Default project behaviour matches project=True."""
        w_default, _ = _Stub(_X3).solve_kkt()
        w_explicit, _ = _Stub(_X3).solve_kkt(project=True)
        np.testing.assert_array_equal(w_default, w_explicit)


# ---------------------------------------------------------------------------
# solve_cvxpy
# ---------------------------------------------------------------------------


class TestSolveCvxpy:
    """Tests for _BaseProblem.solve_cvxpy template."""

    def test_raises_import_error_when_cvxpy_missing(self, monkeypatch):
        """solve_cvxpy raises ImportError when cvxpy is absent from sys.modules."""
        monkeypatch.setitem(sys.modules, "cvxpy", None)
        with pytest.raises(ImportError, match="cvxpy"):
            _Stub(_X3).solve_cvxpy()

    def test_calls_cvxpy_constraints(self):
        """_cvxpy_constraints is invoked during solve_cvxpy."""
        stub = _Stub(_X3)
        w, iters = stub.solve_cvxpy()
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        assert np.all(w >= -1e-4)
        assert iters > 0

    def test_solve_cvxpy_project_false(self):
        """project=False returns the raw CVXPY solution without clipping."""
        w, _ = _Stub(_X3).solve_cvxpy(project=False)
        assert w.shape == (3,)

    def test_cvxpy_constraints_called_with_correct_args(self):
        """_cvxpy_constraints receives (w: cp.Variable, cp: module)."""
        import cvxpy as cp

        received = {}

        @dataclass(frozen=True)
        class _SpyStub(_Stub):
            """Stub that records the arguments passed to _cvxpy_constraints."""

            def _cvxpy_constraints(self, w, cp_module):
                """Record the (w, cp) arguments, then return long-only constraints."""
                received["w_type"] = type(w).__name__
                received["cp"] = cp_module
                return [cp_module.sum(w) == 1, w >= 0]

        _SpyStub(_X3).solve_cvxpy()
        assert received["w_type"] == "Variable"
        assert received["cp"] is cp


# ---------------------------------------------------------------------------
# solve_proximal
# ---------------------------------------------------------------------------


class TestSolveProximal:
    """Tests for _BaseProblem.solve_proximal template."""

    def test_returns_tuple(self):
        """solve_proximal returns a (w, iters) tuple."""
        result = _Stub(_X3).solve_proximal()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_iters_positive(self):
        """Iteration count is at least 1."""
        _, iters = _Stub(_X3).solve_proximal()
        assert iters >= 1

    def test_weight_shape(self):
        """Returned weight vector has shape (N,)."""
        w, _ = _Stub(_X3).solve_proximal()
        assert w.shape == (_X3.shape[1],)

    def test_project_true_sums_to_one(self):
        """project=True ensures weights sum to 1."""
        w, _ = _Stub(_X3).solve_proximal(project=True)
        assert w.sum() == pytest.approx(1.0)

    def test_project_true_non_negative(self):
        """project=True ensures all weights are non-negative."""
        w, _ = _Stub(_X3).solve_proximal(project=True)
        assert np.all(w >= 0)

    def test_project_false_still_on_simplex(self):
        """project=False skips clip-and-renormalize; prox_gradient already enforces simplex."""
        w, _ = _Stub(_X3).solve_proximal(project=False)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6)
        assert np.all(w >= -1e-10)

    def test_project_default_clips_and_renormalizes(self):
        """Default (project=True) clips and renormalizes like other template solvers."""
        w, _ = _Stub(_X3).solve_proximal()
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# solve_fista
# ---------------------------------------------------------------------------


class TestSolveFista:
    """Tests for _BaseProblem.solve_fista template."""

    @pytest.fixture(scope="class")
    @staticmethod
    def X():  # noqa: N802
        """Return a (100, 5) return matrix."""
        return np.random.default_rng(0).standard_normal((100, 5))

    def test_returns_tuple(self, X):  # noqa: N803
        """solve_fista returns a (w, iters) tuple."""
        from fast_minimum_variance import Problem

        result = Problem(X).solve_fista()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_weights_sum_to_one(self, X):  # noqa: N803
        """Weights sum to 1."""
        from fast_minimum_variance import Problem

        w, _ = Problem(X).solve_fista()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self, X):  # noqa: N803
        """All weights are non-negative."""
        from fast_minimum_variance import Problem

        w, _ = Problem(X).solve_fista()
        assert np.all(w >= -1e-8)

    def test_with_shrinkage_and_target(self, X):  # noqa: N803
        """Shrinkage branch (alpha > 0, target supplied) exercises the extra_grad path."""
        from fast_minimum_variance import Problem

        T, N = X.shape  # noqa: N806
        alpha = N / (N + T)
        w, iters = Problem(X, alpha=alpha, target=np.eye(N)).solve_fista()
        assert w.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(w >= -1e-8)
        assert iters >= 1
