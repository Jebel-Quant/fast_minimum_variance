"""Shared helpers for marimo notebook scripts."""

import time
from collections.abc import Callable


def set_notebook_plot_style(mpl) -> None:
    """Apply the shared plotting style used by marimo benchmark notebooks."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
        }
    )


def run_timed(fn: Callable, repeats: int = 3):
    """Return ``(result, best_wall_time)`` over repeated calls."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best
