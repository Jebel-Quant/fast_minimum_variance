"""NumPy vs JAX backend benchmark for fast-minimum-variance.

Compares wall-clock time of ``solve_cg`` and ``solve_minres`` across a range
of problem sizes using the ``'numpy'`` and ``'jax'`` backends.

Usage::

    pip install fast-minimum-variance[jax]
    pip install jax-metal          # Apple Silicon only
    python benchmarks/jax_backend.py

JAX note: the first call to each solver traces and compiles the computation
graph (XLA / Metal).  This script runs one **warmup** solve before timing so
that reported numbers reflect steady-state throughput, not compilation cost.
The warmup time is printed separately so you can see the one-off JIT cost.

The benchmark reports:

* ``time_np``   — NumPy backend wall time (``float64``)
* ``time_jax``  — JAX backend wall time after warmup (``float32``)
* ``warmup_jax``— time for the first JAX call (JIT / XLA compilation)
* ``speedup``   — ``time_np / time_jax`` (>1 means JAX is faster)
* ``err``       — ``max |w_jax - w_np|`` (float32 accuracy check)

Run without JAX installed to see NumPy-only timings (JAX columns will show
``N/A``).
"""

import time

import numpy as np

try:
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from fast_minimum_variance.api import Problem

# ---------------------------------------------------------------------------
# Problem sizes to benchmark: (T, N) pairs
# ---------------------------------------------------------------------------
SIZES = [
    (250, 20),
    (500, 50),
    (500, 100),
    (1000, 200),
    (1000, 500),
    (2000, 1000),
]

N_REPS = 5  # timed repetitions after warmup; min is reported


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
    hdr = (
        f"{'T':>6} {'N':>6}  {'time_np':>9}  {'time_jax':>9}  "
        f"{'warmup_jax':>12}  {'speedup':>8}  {'err':>10}"
    )
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

        if t_jax is not None and t_jax > 0:
            speedup = f"{t_np / t_jax:7.2f}x"
        else:
            speedup = "N/A"

        print(
            f"{T:>6} {N:>6}  {_fmt(t_np):>9}  {_fmt(t_jax):>9}  "
            f"{_fmt(warmup):>12}  {speedup:>8}  {_fmt(err, '.2e'):>10}"
        )

    print("─" * len(hdr))


def main():
    """Entry point."""
    print("fast-minimum-variance: NumPy vs JAX backend benchmark")
    print(f"JAX available: {JAX_AVAILABLE}")
    if JAX_AVAILABLE:
        import jax

        print(f"JAX version:   {jax.__version__}")
        print(f"JAX backend:   {jax.default_backend()}")
    print(f"Repetitions:   {N_REPS}  (min of {N_REPS} runs after one warmup)")
    print("Times in seconds.  speedup = time_np / time_jax (>1 means JAX faster)")

    _run_benchmark("solve_cg")
    _run_benchmark("solve_minres")
    print()


if __name__ == "__main__":
    main()
