"""Timing and result-printing helpers for solver benchmarks."""

import time


def run_timed(fn, repeats=3):
    """Return (result, best_wall_time_s) over `repeats` calls."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def print_table(label, results, ref_key="cvxpy"):
    """Print a benchmark table with speedup relative to ref_key.

    Each entry in results maps to a dict with keys:
      time_s   float
      outer    int | None   (active-set outer steps; None if no outer loop)
      inner    int | None   (inner solver iterations; None for direct solvers)
    """
    ref = results[ref_key]["time_s"]
    print(f"\n{label}")
    print(f"{'Method':<30} {'Time (s)':>10} {'Outer':>7} {'Inner':>8} {'Speedup':>10}")
    print("-" * 70)
    for key, v in results.items():
        outer_str = str(v["outer"]) if v.get("outer") is not None else "--"
        inner_str = str(v["inner"]) if v.get("inner") is not None else "--"
        print(f"{key:<30} {v['time_s']:>10.4f} {outer_str:>7} {inner_str:>8} {ref / v['time_s']:>9.1f}x")
