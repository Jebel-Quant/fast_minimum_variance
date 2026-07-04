"""Fuzz the shrinking active-set minimum-variance solver against arbitrary inputs.

``Problem(X, ...).solve_kkt()`` / ``.solve_cg()`` run a primal-dual active-set
loop over a fuzzed returns matrix, optionally under a balance system ``B w = c``.
On adversarial input the solver must fail only through its documented contract
— shape/validation errors (``ValueError``) or numerical breakdown on singular
or non-finite data (``numpy.linalg.LinAlgError``) — never with an unexpected
exception type. This harness exercises that contract with coverage-guided input.

Run locally:
    pip install atheris numpy
    python tests/fuzz/fuzz_minvar.py -atheris_runs=20000

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
    from fast_minimum_variance import Problem

# Documented failure modes: input validation and numerical breakdown on
# singular / non-finite data. Any other exception type is a genuine crash.
_ALLOWED = (ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError)


def test_one_input(data: bytes) -> None:
    """Build a fuzzed problem (optionally with a balance system) and solve it."""
    fdp = atheris.FuzzedDataProvider(data)
    t = fdp.ConsumeIntInRange(1, 16)
    n = fdp.ConsumeIntInRange(1, 8)
    x = np.array([fdp.ConsumeFloat() for _ in range(t * n)], dtype=np.float64).reshape(t, n)

    # Optionally attach a balance system B w = c with p in [0, n] rows.
    p = fdp.ConsumeIntInRange(0, n)
    kwargs = {}
    if p > 0:
        b = np.array([fdp.ConsumeFloat() for _ in range(p * n)], dtype=np.float64).reshape(p, n)
        c = np.array([fdp.ConsumeFloat() for _ in range(p)], dtype=np.float64)
        kwargs = {"B": b, "c": c}

    with contextlib.suppress(_ALLOWED):
        problem = Problem(x, **kwargs)
        problem.solve_kkt()
        problem.solve_cg()


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
