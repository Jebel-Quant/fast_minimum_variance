"""Benchmark: randomized-SVD vs dense-eigh RMT preconditioner for solve_pcg.

Compares the two ways of building the ``pcg_lr`` low-rank RMT preconditioner --
the dense ``eigh`` path (:func:`rmt_target_and_alpha`, which forms the ``n x n``
covariance) versus the matrix-free randomized SVD
(:func:`rmt_preconditioner_rsvd`) -- on both:

* setup cost (wall-clock to produce the factors), and
* total inner CG iterations of ``solve_pcg`` using each preconditioner,

with plain ``solve_cg`` (no preconditioner) as the baseline.  The point: the
randomized-SVD setup is ``O(T n k)`` and never forms ``X^T X``, matching the
solver's matrix-free philosophy, while delivering the same iteration savings.

Run with::

    uv run python benchmarks/bench_rsvd_preconditioner.py
"""

from __future__ import annotations

import time

import numpy as np

from fast_minimum_variance import Problem
from fast_minimum_variance.shrinkage.util import rmt_preconditioner_rsvd, rmt_target_and_alpha


def make_returns(T: int, n: int, k: int, seed: int = 0) -> np.ndarray:  # noqa: N803
    """Return a demeaned (T, n) factor-model matrix with k strong factors."""
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((n, k))
    scales = np.linspace(6.0, 2.0, k)
    factors = rng.standard_normal((T, k)) * scales
    X = factors @ loadings.T + rng.standard_normal((T, n))  # noqa: N806
    return X - X.mean(axis=0)


def _time(fn: object, repeat: int = 3) -> tuple[object, float]:
    """Return (result, best-of-`repeat` wall-clock seconds) for calling ``fn``."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()  # ty:ignore[call-non-callable]
        best = min(best, time.perf_counter() - t0)
    return result, best


def run_case(T: int, n: int, k: int) -> dict[str, object]:  # noqa: N803
    """Benchmark one (T, n, k) case; return a row of timings and iteration counts."""
    X = make_returns(T, n, k)  # noqa: N806
    bar_lam = float(np.sum(X * X)) / (n * T)
    target = bar_lam * np.eye(n)
    alpha = n / (n + T)

    # Setup cost: dense eigh vs randomized SVD.
    (_, dense_lr, _, _), dense_setup = _time(lambda: rmt_target_and_alpha(X))
    rsvd_lr, rsvd_setup = _time(lambda: rmt_preconditioner_rsvd(X, n_components=max(2 * k, 8)))

    # Total inner CG iterations for each solver/preconditioner.
    _, _, cg_iters = Problem(X, alpha=alpha, target=target).solve_cg()
    _, _, pcg_dense = Problem(X, alpha=alpha, target=target, pcg_lr=dense_lr).solve_pcg()
    _, _, pcg_rsvd = Problem(X, alpha=alpha, target=target, pcg_lr=rsvd_lr).solve_pcg()

    return {
        "T": T,
        "n": n,
        "k": k,
        "dense_setup_ms": dense_setup * 1e3,
        "rsvd_setup_ms": rsvd_setup * 1e3,
        "setup_speedup": dense_setup / rsvd_setup,
        "cg_iters": cg_iters,
        "pcg_dense_iters": pcg_dense,
        "pcg_rsvd_iters": pcg_rsvd,
    }


def main() -> None:
    """Run the benchmark grid and print a summary table."""
    cases = [(1000, 200, 5), (2000, 500, 8), (3000, 1000, 10), (4000, 2000, 12)]
    rows = [run_case(*c) for c in cases]

    header = (
        f"{'T':>5} {'n':>5} {'k':>3} | {'dense setup':>12} {'rSVD setup':>11} {'speedup':>8} "
        f"| {'CG it':>6} {'PCG(dense)':>11} {'PCG(rSVD)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['T']:>5} {r['n']:>5} {r['k']:>3} | "
            f"{r['dense_setup_ms']:>10.2f}ms {r['rsvd_setup_ms']:>9.2f}ms {r['setup_speedup']:>7.1f}x | "
            f"{r['cg_iters']:>6} {r['pcg_dense_iters']:>11} {r['pcg_rsvd_iters']:>10}"
        )


if __name__ == "__main__":
    main()
