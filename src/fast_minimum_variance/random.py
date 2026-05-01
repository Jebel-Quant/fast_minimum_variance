"""Return matrix generation utilities."""

import numpy as np


def make_returns(T, N, seed=42):  # noqa: N803
    """Generate a T x N matrix of standard normal returns.

    Args:
        T: Number of time steps (rows).
        N: Number of assets (columns).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (T, N) with i.i.d. standard normal entries.

    Examples:
        >>> R = make_returns(100, 5, seed=0)
        >>> R.shape
        (100, 5)
        >>> import numpy as np
        >>> np.allclose(R.mean(axis=0), np.zeros(5), atol=0.3)
        True
    """
    # np.random.default_rng (Generator API) is used rather than the legacy
    # RandomState interface because it is faster, avoids global state, and
    # produces statistically better sequences.
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))
