"""Generate figures and benchmark numbers for markowitz_paper.tex."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "fast-minimum-variance[convex]",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "..", editable = true }
# ///

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fast_minimum_variance.api import Problem

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


def lw_params(X):  # noqa: N803
    """Return (c, gamma) Ledoit-Wolf shrinkage parameters for returns matrix X."""
    T, N = X.shape  # noqa: N806
    frob_sq = np.einsum("ti,ti->", X, X)
    return T / (N + T), frob_sq / (N + T)


def run_timed(fn, repeats=3):
    """Return (result, best_wall_time) over repeats calls."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def make_constraints(N, n_sectors, cap):  # noqa: N803
    """Build (C, d) for long-only + equal-sized sector caps."""
    sector = np.zeros((N, n_sectors))
    size = N // n_sectors
    for k in range(n_sectors):
        lo = k * size
        hi = lo + size if k < n_sectors - 1 else N
        sector[lo:hi, k] = 1.0
    C = np.hstack([-np.eye(N), sector])  # noqa: N806
    d = np.concatenate([np.zeros(N), np.full(n_sectors, cap)])
    return C, d


# ── Section 1: Synthetic benchmark  N=1000, T=2000 ───────────────────────────

print("=" * 70)
print("Synthetic benchmark  N=1000, T=2000, 5 sector caps at 25%")
print("=" * 70)

rng = np.random.default_rng(0)
N_bench, T_bench = 1000, 2000
X_bench = rng.standard_normal((T_bench, N_bench))
mu_bench = rng.standard_normal(N_bench)
C_bench, d_bench = make_constraints(N_bench, 5, 0.25)
c_lw, gamma_lw = lw_params(X_bench)

configs = [
    ("cvxpy", lambda: Problem(X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench).solve_cvxpy(project=False)),
    ("kkt", lambda: Problem(X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench).solve_kkt(project=False)),
    ("minres", lambda: Problem(X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench).solve_minres(project=False)),
    ("cg", lambda: Problem(X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench).solve_cg(project=False)),
    (
        "minres_lw",
        lambda: Problem(
            np.sqrt(c_lw) * X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench, gamma=gamma_lw
        ).solve_minres(project=False),
    ),
    (
        "cg_lw",
        lambda: Problem(np.sqrt(c_lw) * X_bench, C=C_bench, d=d_bench, rho=0.5, mu=mu_bench, gamma=gamma_lw).solve_cg(
            project=False
        ),
    ),
]

display = {
    "cvxpy": "cvxpy (Clarabel)",
    "kkt": "KKT direct",
    "minres": "MINRES",
    "cg": "CG (constr.-elim.)",
    "minres_lw": "MINRES + LW",
    "cg_lw": "CG + LW",
}

bench_results = {}
print(f"{'method':<30} {'time_s':>10} {'iters':>8}")
print("-" * 52)
for key, fn in configs:
    (w, iters), t = run_timed(fn)
    bench_results[key] = {"time_s": t, "iters": iters, "w": w}
    iters_str = str(iters) if iters is not None else "--"
    print(f"{display[key]:<30} {t:>10.4f} {iters_str:>8}")

ref = bench_results["cvxpy"]["time_s"]
print()
for key in ("kkt", "minres", "cg", "minres_lw", "cg_lw"):
    spd = ref / bench_results[key]["time_s"]
    print(f"  {display[key]}: {spd:.1f}x speedup vs cvxpy")


# ── Section 2: Runtime vs N (constrained Markowitz) ──────────────────────────

ns = [50, 100, 200, 300, 500, 750, 1000]
times_markowitz = {k: [] for k in ("kkt", "minres_lw", "cg_lw")}
rng2 = np.random.default_rng(1)

print()
print("=" * 70)
print("Runtime vs N  (5 sector caps, rho=0.5, LW shrinkage, T=2N)")
print("=" * 70)
print(f"{'N':>6}  {'kkt':>10}  {'minres_lw':>12}  {'cg_lw':>10}")
print("-" * 46)

for n in ns:
    T = 2 * n
    X = rng2.standard_normal((T, n))
    mu = rng2.standard_normal(n)
    C, d = make_constraints(n, 5, 0.25)
    c, gamma = lw_params(X)
    _, t_kkt = run_timed(lambda x=X, cc=C, dd=d, mm=mu: Problem(x, C=cc, d=dd, rho=0.5, mu=mm).solve_kkt(project=False))
    _, t_mr = run_timed(
        lambda x=X, cc=C, dd=d, mm=mu, cv=c, gv=gamma: Problem(
            np.sqrt(cv) * x, C=cc, d=dd, rho=0.5, mu=mm, gamma=gv
        ).solve_minres(project=False)
    )
    _, t_cg = run_timed(
        lambda x=X, cc=C, dd=d, mm=mu, cv=c, gv=gamma: Problem(
            np.sqrt(cv) * x, C=cc, d=dd, rho=0.5, mu=mm, gamma=gv
        ).solve_cg(project=False)
    )
    times_markowitz["kkt"].append(t_kkt)
    times_markowitz["minres_lw"].append(t_mr)
    times_markowitz["cg_lw"].append(t_cg)
    print(f"{n:>6}  {t_kkt:>10.4f}  {t_mr:>12.4f}  {t_cg:>10.4f}")


# ── Section 3: Efficient frontier timing ─────────────────────────────────────

print()
print("=" * 70)
print("Efficient frontier  N=500, T=1000, 5 sector caps (21 rho values)")
print("=" * 70)

N_ef, T_ef = 500, 1000
X_ef = np.random.default_rng(2).standard_normal((T_ef, N_ef))
mu_ef = np.random.default_rng(3).standard_normal(N_ef)
C_ef, d_ef = make_constraints(N_ef, 5, 0.25)
c_ef, gamma_ef = lw_params(X_ef)
rhos = np.linspace(0, 2, 21)


def frontier_kkt():
    """Compute efficient frontier weights for all rho values using KKT direct."""
    return [Problem(X_ef, C=C_ef, d=d_ef, rho=r, mu=mu_ef).solve_kkt(project=False) for r in rhos]


def frontier_cg():
    """Compute efficient frontier weights for all rho values using CG + LW."""
    return [
        Problem(np.sqrt(c_ef) * X_ef, C=C_ef, d=d_ef, rho=r, mu=mu_ef, gamma=gamma_ef).solve_cg(project=False)
        for r in rhos
    ]


def frontier_cvxpy():
    """Compute efficient frontier weights for all rho values using CVXPY."""
    return [Problem(X_ef, C=C_ef, d=d_ef, rho=r, mu=mu_ef).solve_cvxpy(project=False) for r in rhos]


_, t_ef_kkt = run_timed(frontier_kkt)
_, t_ef_cg = run_timed(frontier_cg)
_, t_ef_cvxpy = run_timed(frontier_cvxpy)

print(f"  cvxpy  : {t_ef_cvxpy:.3f} s  ({t_ef_cvxpy / len(rhos) * 1000:.1f} ms/point)")
print(f"  kkt    : {t_ef_kkt:.3f} s  ({t_ef_kkt / len(rhos) * 1000:.1f} ms/point)")
print(f"  cg+lw  : {t_ef_cg:.3f} s  ({t_ef_cg / len(rhos) * 1000:.1f} ms/point)")
print(f"  speedup cg vs cvxpy: {t_ef_cvxpy / t_ef_cg:.1f}x")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

colors = {"kkt": "#1f77b4", "minres_lw": "#d62728", "cg_lw": "#2ca02c"}
labels_plot = {"kkt": "KKT direct", "minres_lw": "MINRES + LW", "cg_lw": "CG + LW"}
for key in ("kkt", "minres_lw", "cg_lw"):
    ax1.plot(ns, times_markowitz[key], marker="o", markersize=3, label=labels_plot[key], color=colors[key])
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of assets $n$")
ax1.set_ylabel("Wall-clock time (s)")
ax1.set_title(r"(a) Runtime vs.\ $n$  (5 sector caps, $\rho=0.5$, LW)")
ax1.legend(framealpha=0.9)
ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

# Panel B: efficient frontier portfolios (return vs variance)
ws_cg = [
    Problem(np.sqrt(c_ef) * X_ef, C=C_ef, d=d_ef, rho=r, mu=mu_ef, gamma=gamma_ef).solve_cg(project=False)[0]
    for r in rhos
]
rets = [mu_ef @ w for w in ws_cg]
vols = [float(np.linalg.norm(X_ef @ w)) for w in ws_cg]
ax2.plot(vols, rets, marker="o", markersize=3, color=colors["cg_lw"])
ax2.set_xlabel(r"Portfolio risk $\|Xw\|$")
ax2.set_ylabel(r"Expected return $\mu^\top w$")
ax2.set_title(r"(b) Efficient frontier  ($n=500$, 5 sector caps)")
ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

fig.tight_layout(pad=1.0)
fig.savefig("paper/markowitz_scaling.pdf", bbox_inches="tight")
fig.savefig("paper/markowitz_scaling.png", bbox_inches="tight", dpi=150)
print()
print("Saved paper/markowitz_scaling.pdf and paper/markowitz_scaling.png")
