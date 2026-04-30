"""Generate scaling figures for minvar_paper.tex."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "fast-minimum-variance",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "..", editable = true }
# ///

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fast_minimum_variance.kkt import solve_kkt
from fast_minimum_variance.krylov import solve_cg, solve_minres

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


def lw_params(R):  # noqa: N803
    """Return Ledoit-Wolf (c, gamma) shrinkage parameters for returns matrix R."""
    T, N = R.shape  # noqa: N806
    frob_sq = np.einsum("ti,ti->", R, R)
    return T / (N + T), frob_sq / (N + T)


def run_timed(fn, repeats=3):
    """Return (result, best_time) over repeats calls to fn."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


# ── Panel A: runtime vs n ──────────────────────────────────────────────────────

ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
times = {k: [] for k in ("kkt", "minres", "cg")}
rng = np.random.default_rng(0)
for n in ns:
    T = 2 * n
    R = rng.standard_normal((T, n))
    c, gamma = lw_params(R)
    _, t = run_timed(lambda r=R: solve_kkt(r))
    times["kkt"].append(t)
    _, t = run_timed(lambda r=R, c=c, g=gamma: solve_minres(r, c=c, gamma=g))
    times["minres"].append(t)
    _, t = run_timed(lambda r=R, c=c, g=gamma: solve_cg(r, c=c, gamma=g))
    times["cg"].append(t)

# ── Panel B: iterations vs shrinkage intensity alpha ──────────────────────────
# n > T so the raw sample covariance is rank-deficient (ill-conditioned baseline).
# Increasing alpha lifts small eigenvalues and compresses the spectrum.

n_iter, T_iter = 500, 250  # n/T = 2 → singular sample covariance
R_iter = np.random.default_rng(1).standard_normal((T_iter, n_iter))
frob_sq = np.einsum("ti,ti->", R_iter, R_iter)

alphas = np.concatenate([[0.005, 0.01, 0.02], np.linspace(0.04, 0.60, 28)])
iters_minres, iters_cg = [], []

for alpha in alphas:
    c = 1.0 - alpha
    gamma = alpha * frob_sq / n_iter
    (_, i_m) = solve_minres(R_iter, c=c, gamma=gamma)
    iters_minres.append(i_m)
    (_, i_c) = solve_cg(R_iter, c=c, gamma=gamma)
    iters_cg.append(i_c)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

colors = {"kkt": "#1f77b4", "minres": "#d62728", "cg": "#2ca02c"}
labels = {"kkt": "KKT direct", "minres": "MINRES", "cg": "CG (constr.-elim.)"}
for key in ("kkt", "minres", "cg"):
    ax1.plot(ns, times[key], marker="o", markersize=3, label=labels[key], color=colors[key])
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of assets $n$")
ax1.set_ylabel("Wall-clock time (s)")
ax1.set_title(r"(a) Runtime vs. $n$  ($T=2n$, with LW shrinkage)")
ax1.legend(framealpha=0.9)
ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

ax2.plot(
    alphas,
    iters_minres,
    marker="o",
    markersize=3,
    label="MINRES",
    color=colors["minres"],
)
ax2.plot(
    alphas,
    iters_cg,
    marker="s",
    markersize=3,
    label="CG (constr.-elim.)",
    color=colors["cg"],
)
ax2.set_xlabel(r"Shrinkage intensity $\alpha$  $(\kappa$ decreases $\rightarrow)$")
ax2.set_ylabel("Iterations to convergence")
ax2.set_title(r"(b) Iterations vs. $\alpha$  ($n=500,\,T=250$)")
ax2.legend(framealpha=0.9)
ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

fig.tight_layout(pad=1.0)
fig.savefig("paper/minvar_scaling.pdf", bbox_inches="tight")
fig.savefig("paper/minvar_scaling.png", bbox_inches="tight", dpi=150)
print("Saved paper/minvar_scaling.pdf and paper/minvar_scaling.png")
