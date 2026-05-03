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
        w, step_iters = solve_fn(None)
        return w, step_iters

    def _kkt_step(self, mask):
        return np.array([0.5, -0.1, 0.6]), 1

    def _cvxpy_constraints(self, w, cp):
        return [cp.sum(w) == 1, w >= 0]


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
            def _constraint_active_set(self, fn):
                return fn(None)

            def _kkt_step(self, mask):
                return np.zeros(3), 1

            # _cvxpy_constraints intentionally omitted

        with pytest.raises(TypeError):
            _Partial(_X3)

    def test_complete_subclass_instantiates(self):
        """A fully-implemented subclass can be instantiated."""
        stub = _Stub(_X3)
        assert stub is not None


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


class TestRidge:
    """Tests for _BaseProblem._ridge."""

    def test_zero_alpha_gives_zero_ridge(self):
        """Ridge is 0 when alpha=0."""
        assert _Stub(_X3, alpha=0.0)._ridge() == 0.0

    def test_nonzero_alpha(self):
        """Ridge equals alpha * ||X||_F^2 / n."""
        X = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # noqa: N806
        assert _Stub(X, alpha=0.4)._ridge() == pytest.approx(0.4 * 5 / 2)

    def test_identity_matrix(self):
        """For identity X, ridge equals alpha."""
        assert _Stub(np.eye(5), alpha=0.3)._ridge() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Template solver delegation and project behaviour
# ---------------------------------------------------------------------------


class TestTemplateDelegation:
    """solve_* passes the correct _XXX_step to _constraint_active_set."""

    def test_solve_kkt_uses_kkt_step(self):
        """solve_kkt delegates to _kkt_step (iters==1)."""
        _, iters = _Stub(_X3).solve_kkt()
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
            def _cvxpy_constraints(self, w, cp_module):
                received["w_type"] = type(w).__name__
                received["cp"] = cp_module
                return [cp_module.sum(w) == 1, w >= 0]

        _SpyStub(_X3).solve_cvxpy()
        assert received["w_type"] == "Variable"
        assert received["cp"] is cp
