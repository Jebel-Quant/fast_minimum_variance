"""Tests for fast_minimum_variance.data — simulate_equity_returns."""

import numpy as np
import pytest

from fast_minimum_variance.data import simulate_equity_returns


class TestSimulateEquityReturns:
    """Tests for simulate_equity_returns."""

    def test_output_shape(self) -> None:
        """Output has shape (T, n)."""
        X = simulate_equity_returns(100, 50, rng=0)  # noqa: N806
        assert X.shape == (50, 100)

    def test_columns_zero_mean(self) -> None:
        """Each column is demeaned (mean == 0)."""
        X = simulate_equity_returns(100, 200, k=5, rng=0)  # noqa: N806
        assert bool(abs(X.mean(axis=0)).max() < 1e-14)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical arrays."""
        X1 = simulate_equity_returns(50, 20, rng=42)  # noqa: N806
        X2 = simulate_equity_returns(50, 20, rng=42)  # noqa: N806
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different arrays."""
        X1 = simulate_equity_returns(50, 20, rng=1)  # noqa: N806
        X2 = simulate_equity_returns(50, 20, rng=2)  # noqa: N806
        assert not np.array_equal(X1, X2)

    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_various_k(self, k: int) -> None:
        """Output shape is correct for several factor counts."""
        X = simulate_equity_returns(20, 30, k=k, rng=0)  # noqa: N806
        assert X.shape == (30, 20)

    def test_default_k(self) -> None:
        """Default k (n // 10, min 3) produces correct shape."""
        X = simulate_equity_returns(100, 200, rng=0)  # noqa: N806
        assert X.shape == (200, 100)
