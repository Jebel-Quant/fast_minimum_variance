"""Small worked example for the primal-dual loop in _MinVarProblem.

Three-asset problem designed for hand-verification:

    X = [[1, 0, 1],    X^T X = [[1, 0, 1],
         [0, 1, 1],              [0, 1, 1],
         [0, 0, 1]]              [1, 1, 3]]

Equality-constrained optimum (no long-only): w = [2/3, 2/3, -1/3].
Long-only optimum:                           w* = [1/2, 1/2,   0].

Primal-dual trace:
  Iteration 1 — solve on {0,1,2}: w = [2/3, 2/3, -1/3]; w[2] < 0 → drop.
  Iteration 2 — solve on {0,1}:   w = [1/2, 1/2]; all non-negative.
  Dual check:   grad = [1, 1, 2], lambda_ = 1, nu = [0, 0, 1] >= 0 → done.
"""

import numpy as np
import pytest

from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem

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
    """KKT solver recovers the known long-only optimum [1/2, 1/2, 0]."""
    w, _ = MinVarProblem(X3).solve_kkt()
    np.testing.assert_allclose(w, W_OPT, atol=1e-10)


def test_two_outer_iterations():
    """Primal step fires once (asset 2 dropped); dual check passes immediately."""
    p = MinVarProblem(X3)
    calls = []

    def counting_kkt(active):
        calls.append(active.copy())
        return p._kkt_step(active)

    p._constraint_active_set(counting_kkt)

    assert len(calls) == 2
    assert calls[0].all()  # iteration 1: full active set
    assert not calls[1][2]  # iteration 2: asset 2 excluded


# ---------------------------------------------------------------------------
# Dual re-add
# ---------------------------------------------------------------------------


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
        call_no[0] += 1
        if call_no[0] == 1:
            return np.array([0.45, 0.45, -0.1]), 1  # w[2] < 0 → primal drop
        if call_no[0] == 2:
            return np.array([0.5, 0.5]), 1  # nu_2 = 0-1 = -1 → re-add
        return np.ones(active.sum()) / active.sum(), 1

    w, _ = p._constraint_active_set(solve_fn)

    assert call_no[0] == 3
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


if __name__ == "__main__":
    # run with pytest -s test_small.py
    test_covariance()
    test_two_outer_iterations()
    test_dual_variable_at_optimum()
    test_known_optimum()
    test_dual_readd()
