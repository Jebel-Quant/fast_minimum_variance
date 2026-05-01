"""Shared fixtures for fast_minimum_variance tests.

Security note: Test code uses pytest assertions (S101), which are intentional
and safe in the test context. No subprocess calls (S603/S607) are used here.
"""

import numpy as np
import pytest

from fast_minimum_variance.api import Problem


def make_returns(T, N, seed=42):  # noqa: N803
    """Generate a T x N matrix of i.i.d. standard normal returns."""
    return np.random.default_rng(seed).standard_normal((T, N))


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 3) for fast solver tests."""
    return make_returns(T=100, N=3, seed=0)


@pytest.fixture(scope="session")
def problem(X):  # noqa: N803
    """Problem instance wrapping the session-scoped return matrix."""
    return Problem(X)


@pytest.fixture(scope="session")
def problem_small(X_small):  # noqa: N803
    """Problem instance wrapping the small session-scoped return matrix."""
    return Problem(X_small)
