"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

import numpy as np


def standard(X, A=None, b=None, C=None, d=None):  # noqa: N803
    """Fill in default values for A, b, C, d given the return matrix X."""
    n = X.shape[1]

    if A is None:
        A = np.ones((n, 1))  # noqa: N806
    if b is None:
        b = np.ones(1)
    if C is None:
        C = -np.eye(n)  # noqa: N806
    if d is None:
        d = np.zeros(n)

    return A, b, C, d
