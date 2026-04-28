"""Return matrix generation utilities."""

import numpy as np


def make_returns(T, N, seed=42):  # noqa: N803
    """Generate a T x N matrix of standard normal returns."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))
