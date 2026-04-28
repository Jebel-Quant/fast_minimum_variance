"""Tests for fast_minimum_variance.random."""

import numpy as np
import pytest

from fast_minimum_variance.random import make_returns


def test_shape():
    """Output shape is (T, N)."""
    R = make_returns(T=50, N=5)  # noqa: N806
    assert R.shape == (50, 5)


def test_reproducible():
    """Same seed yields identical matrices."""
    R1 = make_returns(T=100, N=10, seed=7)  # noqa: N806
    R2 = make_returns(T=100, N=10, seed=7)  # noqa: N806
    np.testing.assert_array_equal(R1, R2)


def test_different_seeds_differ():
    """Different seeds yield different matrices."""
    R1 = make_returns(T=100, N=10, seed=1)  # noqa: N806
    R2 = make_returns(T=100, N=10, seed=2)  # noqa: N806
    assert not np.array_equal(R1, R2)


def test_standard_normal_statistics():
    """Mean ≈ 0 and std ≈ 1 for a large draw."""
    R = make_returns(T=10_000, N=1, seed=0)  # noqa: N806
    assert abs(R.mean()) < 0.05
    assert abs(R.std() - 1.0) < 0.05


@pytest.mark.parametrize(("T", "N"), [(1, 1), (1, 100), (100, 1), (500, 20)])
def test_various_shapes(T, N):  # noqa: N803
    """make_returns produces the requested shape for edge-case dimensions."""
    assert make_returns(T=T, N=N).shape == (T, N)
