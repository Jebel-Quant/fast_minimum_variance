"""Tests for the Problem factory in fast_minimum_variance.__init__."""

import numpy as np
import pytest

from fast_minimum_variance import Problem
from fast_minimum_variance.minvar_problem import _MinVarProblem
from fast_minimum_variance.problem import _Problem


@pytest.fixture(scope="module")
def X():  # noqa: N802
    """Return matrix of shape (100, 5) with a fixed seed."""
    return np.random.default_rng(0).standard_normal((100, 5))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    """Problem() routes to the right implementation class."""

    def test_no_constraints_returns_minvar(self, X):  # noqa: N803
        """No constraint args → shrinking active-set implementation."""
        assert isinstance(Problem(X), _MinVarProblem)

    def test_a_triggers_general(self, X):  # noqa: N803
        """Passing A alone → growing active-set implementation."""
        assert isinstance(Problem(X, A=np.ones((5, 1))), _Problem)

    def test_b_triggers_general(self, X):  # noqa: N803
        """Passing b alone → growing active-set implementation."""
        assert isinstance(Problem(X, b=np.ones(1)), _Problem)

    def test_c_triggers_general(self, X):  # noqa: N803
        """Passing C alone → growing active-set implementation."""
        assert isinstance(Problem(X, C=-np.eye(5)), _Problem)

    def test_d_triggers_general(self, X):  # noqa: N803
        """Passing d alone → growing active-set implementation."""
        assert isinstance(Problem(X, d=np.zeros(5)), _Problem)


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestParameterForwarding:
    """Keyword arguments are forwarded correctly to the underlying class."""

    def test_alpha_forwarded_to_minvar(self, X):  # noqa: N803
        """Alpha is forwarded when dispatching to _MinVarProblem."""
        assert Problem(X, alpha=0.1).alpha == pytest.approx(0.1)

    def test_rho_forwarded_to_minvar(self, X):  # noqa: N803
        """Rho is forwarded when dispatching to _MinVarProblem."""
        assert Problem(X, rho=0.5).rho == pytest.approx(0.5)

    def test_mu_forwarded_to_minvar(self, X):  # noqa: N803
        """Mu is forwarded when dispatching to _MinVarProblem."""
        mu = np.ones(5)
        np.testing.assert_array_equal(Problem(X, mu=mu).mu, mu)

    def test_alpha_forwarded_to_general(self, X):  # noqa: N803
        """Alpha is forwarded when dispatching to _Problem."""
        assert Problem(X, A=np.ones((5, 1)), alpha=0.2).alpha == pytest.approx(0.2)

    def test_rho_forwarded_to_general(self, X):  # noqa: N803
        """Rho is forwarded when dispatching to _Problem."""
        assert Problem(X, A=np.ones((5, 1)), rho=0.3).rho == pytest.approx(0.3)

    def test_mu_forwarded_to_general(self, X):  # noqa: N803
        """Mu is forwarded when dispatching to _Problem."""
        mu = np.arange(5, dtype=float)
        np.testing.assert_array_equal(Problem(X, A=np.ones((5, 1)), mu=mu).mu, mu)

    def test_partial_constraints_fill_defaults(self, X):  # noqa: N803
        """Unset constraint args default to None, triggering __post_init__ defaults."""
        p = Problem(X, A=np.ones((5, 1)))
        assert isinstance(p, _Problem)
        np.testing.assert_array_equal(p.b, np.ones(1))
        np.testing.assert_array_equal(p.C, -np.eye(5))
        np.testing.assert_array_equal(p.d, np.zeros(5))


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Only Problem is exported from the package."""

    def test_all_contains_problem(self):
        """Problem appears in __all__."""
        import fast_minimum_variance

        assert "Problem" in fast_minimum_variance.__all__

    def test_private_classes_not_in_all(self):
        """Private implementation classes are not re-exported."""
        import fast_minimum_variance

        assert "_Problem" not in fast_minimum_variance.__all__
        assert "_MinVarProblem" not in fast_minimum_variance.__all__
