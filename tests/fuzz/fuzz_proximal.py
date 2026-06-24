"""Fuzz the simplex projection in fast_minimum_variance against arbitrary vectors.

``proj_simplex`` computes the Euclidean projection of a vector onto the
probability simplex. It must never crash with an unexpected exception on
adversarial input — empty input raises ``ValueError``, and non-finite values
should produce a result or a documented error, not blow up unexpectedly. This
harness exercises that contract with coverage-guided input.

Run locally:
    pip install atheris numpy
    python tests/fuzz/fuzz_proximal.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Importing fast_minimum_variance pulls in cvxpy/clarabel/osqp/scipy via the
# package __init__; we let them load uninstrumented and instrument only the
# first-party package under test.
import clarabel  # noqa: F401  # pre-imported uninstrumented
import cvxpy  # noqa: F401  # pre-imported uninstrumented
import numpy as np
import osqp  # noqa: F401  # pre-imported uninstrumented
import scipy.sparse  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from fast_minimum_variance.proximal import proj_simplex

_ALLOWED = (ValueError, ZeroDivisionError, FloatingPointError)


def test_one_input(data: bytes) -> None:
    """Project a fuzzed vector onto the probability simplex."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(0, 16)
    vec = np.array([fdp.ConsumeFloat() for _ in range(n)], dtype=np.float64)
    rad = fdp.ConsumeFloat()

    with contextlib.suppress(_ALLOWED):
        proj_simplex(vec, rad)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
