"""Timing and result-printing helpers for solver benchmarks."""

from __future__ import annotations

import time
from pathlib import Path


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


def _fmt_time(t):
    """Format a wall-clock time in seconds for a LaTeX table."""
    if t >= 10:
        return f"{t:.1f}"
    if t >= 0.1:
        return f"{t:.3f}"
    return f"{t:.4f}"


def write_benchmark_rows(path, results, ref_key, footnote_methods=None, method_order=None):
    r"""Write tabular data rows to a .tex file for \\input inclusion.

    Each row: method & time_s & iterations & speedup \\\\
    footnote_methods: set of method names that get a $^\\dagger$ appended.
    method_order: list controlling which methods appear and in what order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = results[ref_key]["time_s"]
    order = method_order if method_order is not None else list(results)
    lines = []
    for method in order:
        if method not in results:
            continue
        v = results[method]
        label = method
        if footnote_methods and method in footnote_methods:
            label = f"{method}$^\\dagger$"
        iters = v.get("inner") if v.get("inner") is not None else v.get("outer")
        iters_str = str(iters) if iters is not None else "--"
        speedup = ref / v["time_s"]
        lines.append(f"{label:<35} & {_fmt_time(v['time_s']):>8} & {iters_str:>6} & {speedup:>6.1f}x \\\\\n")
    path.write_text("".join(lines))


def write_frontier_rows(path, rows, n_pts):
    r"""Write frontier sweep rows to a .tex file for \\input inclusion.

    rows: list of dicts with keys:
      label   str   display name (may contain LaTeX)
      cold    float total cold-start time (s)
      warm    float | None   total warm-start time (s); None if no warm-start API
    n_pts: number of frontier points (used to compute ms/point).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        label = row["label"]
        cold_ms = row["cold"] / n_pts * 1000
        if row.get("warm") is not None:
            warm_ms = row["warm"] / n_pts * 1000
            lines.append(
                f"{label:<32} & {_fmt_time(row['cold']):>6} & {cold_ms:>5.1f}"
                f" & {_fmt_time(row['warm']):>6} & {warm_ms:>5.1f} \\\\\n"
            )
        else:
            lines.append(
                f"{label:<32} & {_fmt_time(row['cold']):>6} & {cold_ms:>5.1f} & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
            )
    path.write_text("".join(lines))
