"""NumPy vs JAX backend benchmark for fast-minimum-variance.

Compares wall-clock time of ``solve_cg`` and ``solve_minres`` across a range
of problem sizes using the ``'numpy'`` and ``'jax'`` backends.

Usage::

    pip install fast-minimum-variance[jax]
    pip install jax-metal          # Apple Silicon only
    python benchmarks/jax_backend.py

**CPU vs Metal**

JAX on a CPU backend is *slower* than NumPy because XLA adds per-operation
dispatch overhead that outweighs the benefit of compiled loops for these
problem sizes.  The JAX backend is designed for GPU/Metal, where the two
matrix-vector products per Krylov step (``X.T @ (X @ x)``) run as accelerated
GEMVs and the loop stays fully on-device via ``jax.lax.while_loop``.

To use the Metal GPU on Apple Silicon::

    pip install jax-metal

Once installed, ``jax.default_backend()`` will report ``'metal'`` and
speedups of 5-20x are typical at N >= 500 on M-series chips.

**JAX warmup**

The first call per problem size traces and compiles the XLA / Metal kernel.
This script runs one warmup solve before timing so that reported numbers
reflect steady-state throughput, not compilation cost.  The ``warmup_jax``
column shows the one-off JIT cost (paid once per process, not per solve in a
rolling-window loop).

The benchmark reports:

* ``time_np``   — NumPy backend wall time (``float64``)
* ``time_jax``  — JAX backend wall time after warmup (``float32``)
* ``warmup_jax``— first-call JIT / XLA compilation overhead
* ``speedup``   — ``time_np / time_jax``  (>1 means JAX is faster)
* ``err``       — ``max |w_jax - w_np|``  (float32 accuracy check)

Run without JAX installed to see NumPy-only timings (JAX columns show N/A).
"""

import sys
import time

import numpy as np

try:
    import jax

    JAX_AVAILABLE = True
    _JAX_VERSION = jax.__version__
    _JAX_BACKEND = jax.default_backend()
except ImportError:
    JAX_AVAILABLE = False
    _JAX_VERSION = None
    _JAX_BACKEND = None

from fast_minimum_variance.api import Problem

# ---------------------------------------------------------------------------
# Problem sizes to benchmark: (T, N) pairs
# ---------------------------------------------------------------------------
SIZES = [(250, 20), (500, 50), (500, 100), (1000, 200), (1000, 500), (2000, 1000), (5000, 2500), (10000, 5000)]

N_REPS = 5  # timed repetitions after warmup; min is reported

_INSTALL_MSG = """\
  JAX is not installed in this environment.  To enable the JAX columns:

    pip install fast-minimum-variance[jax]
    pip install jax-metal          # Apple Silicon / Metal GPU
    # or: pip install jax[cuda12]  # NVIDIA CUDA 12

  Then re-run this script.\
"""

_CPU_WARNING = """\
  Note: JAX backend is 'cpu' — XLA adds dispatch overhead that makes the JAX
  path slower than NumPy on CPU.  Install jax-metal (Apple Silicon) or a CUDA
  build of JAX to see GPU speedups.\
"""


def _make_problem(T, N, seed=42, backend="numpy"):  # noqa: N803
    """Generate a random return matrix and construct a Problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))  # noqa: N806
    return Problem(X, backend=backend)


def _time_solve(fn, n_reps):
    """Return (min_time_seconds, result) over n_reps calls."""
    best = float("inf")
    result = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def _run_size(T, N, solver):  # noqa: N803
    """Return a result dict for one (T, N, solver) combination."""
    p_np = _make_problem(T, N, backend="numpy")
    fn_np = getattr(p_np, solver)

    # NumPy timing
    t_np, (w_np, _) = _time_solve(fn_np, N_REPS)

    if not JAX_AVAILABLE:
        return {"t_np": t_np, "t_jax": None, "t_warmup": None, "err": None}

    p_jax = _make_problem(T, N, backend="jax")
    fn_jax = getattr(p_jax, solver)

    # Warmup: first call pays JIT / XLA compilation cost
    t0 = time.perf_counter()
    w_jax, _ = fn_jax()
    t_warmup = time.perf_counter() - t0

    # Steady-state timing
    t_jax, (w_jax, _) = _time_solve(fn_jax, N_REPS)

    err = float(np.max(np.abs(np.asarray(w_jax) - w_np)))
    return {"t_np": t_np, "t_jax": t_jax, "t_warmup": t_warmup, "err": err}


def _fmt(val, fmt=".4f", na_str="N/A"):
    """Format a value or return na_str if None."""
    return f"{val:{fmt}}" if val is not None else na_str


def _run_benchmark(solver):
    """Run the full benchmark for one solver and print results."""
    hdr = f"{'T':>6} {'N':>6}  {'time_np':>9}  {'time_jax':>9}  {'warmup_jax':>12}  {'speedup':>8}  {'err':>10}"
    print(f"\n{'─' * len(hdr)}")
    print(f"  {solver}")
    print(f"{'─' * len(hdr)}")
    print(hdr)
    print("─" * len(hdr))

    for T, N in SIZES:  # noqa: N806
        r = _run_size(T, N, solver)
        t_np = r["t_np"]
        t_jax = r["t_jax"]
        warmup = r["t_warmup"]
        err = r["err"]

        speedup = f"{t_np / t_jax:7.2f}x" if t_jax is not None and t_jax > 0 else "N/A"

        print(
            f"{T:>6} {N:>6}  {_fmt(t_np):>9}  {_fmt(t_jax):>9}  "
            f"{_fmt(warmup):>12}  {speedup:>8}  {_fmt(err, '.2e'):>10}"
        )

    print("─" * len(hdr))


def main():
    """Entry point."""
    print("fast-minimum-variance: NumPy vs JAX backend benchmark")
    print(f"Python:        {sys.version.split()[0]}  ({sys.executable})")
    if JAX_AVAILABLE:
        print(f"JAX:           {_JAX_VERSION}  (backend '{_JAX_BACKEND}')")
        if _JAX_BACKEND == "cpu":
            print(_CPU_WARNING)
    else:
        print("JAX:           not installed")
        print(_INSTALL_MSG)
    print(f"Repetitions:   {N_REPS}  (min of {N_REPS} runs after one warmup)")
    print("Times in seconds.  speedup = time_np / time_jax (>1 means JAX faster)")

    _run_benchmark("solve_cg")
    _run_benchmark("solve_minres")
    print()


if __name__ == "__main__":
    main()
