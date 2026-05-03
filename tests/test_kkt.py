"""Tests for kkt.kkt_minres.

Three-asset problem:

    X = [[1, 0, 1],    X^T X = [[1, 0, 1],
         [0, 1, 1],              [0, 1, 1],
         [0, 0, 1]]              [1, 1, 3]]

Long-only optimum: w* = [1/2, 1/2, 0].
"""

import numpy as np
import pytest

from fast_minimum_variance.kkt import pd_projected_qp_solver as kkt_minres

X3 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])


def test_budget_constraint():
    """Weights sum to 1."""
    w, _, _ = kkt_minres(X3)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)


def test_long_only():
    """All weights are non-negative."""
    w, _, _ = kkt_minres(X3)
    assert np.all(w >= -1e-6)


def test_known_optimum_x3():
    """Recovers the analytic long-only optimum [1/2, 1/2, 0] for X3."""
    w, _, _ = kkt_minres(X3)
    np.testing.assert_allclose(w, [0.5, 0.5, 0.0], atol=1e-4)


def test_known_optimum_identity():
    """X = I_n → equal-weight portfolio is the unique long-only minimum."""
    n = 5
    w, _, _ = kkt_minres(np.eye(n))
    np.testing.assert_allclose(w, np.ones(n) / n, atol=1e-4)


def test_iters_respects_maxiter():
    """Returned iteration count never exceeds maxiter."""
    for m in (50, 100, 200):
        _, _, iters = kkt_minres(X3, maxiter=m)
        assert iters <= m


def test_return_types():
    """Return signature is (ndarray, float, int)."""
    w, lam, iters = kkt_minres(X3)
    assert w.shape == (3,)
    assert isinstance(float(lam), float)
    assert isinstance(iters, int)


def test_mu_none_equals_zero_rho():
    """mu=None (default) should give the same result as rho=0."""
    w1, _, _ = kkt_minres(X3)
    w2, _, _ = kkt_minres(X3, mu=np.zeros(3), rho=0.0)
    np.testing.assert_allclose(w1, w2, atol=1e-10)


def test_return_tilt_shifts_weight():
    """A large positive return on asset 0 should increase its weight."""
    mu = np.array([1.0, 0.0, 0.0])
    w_base, _, _ = kkt_minres(X3)
    w_tilt, _, _ = kkt_minres(X3, mu=mu, rho=1.0)
    assert w_tilt[0] > w_base[0]
