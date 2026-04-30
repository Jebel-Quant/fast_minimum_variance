"""Run solver benchmark on real S&P 500 data."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fast-minimum-variance[convex]",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "..", editable = true }
# ///

import time

import numpy as np
import pandas as pd

from fast_minimum_variance.cvx import minvar_cvxpy
from fast_minimum_variance.kkt import minvar_kkt
from fast_minimum_variance.krylov import minvar_cg, minvar_minres

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_parquet("paper/sp500_returns.parquet")
R = df.to_numpy()
T, N = R.shape
print(f"S&P 500 returns: T={T} trading days, N={N} assets")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}\n")

# ── Ledoit-Wolf parameters ─────────────────────────────────────────────────────

frob_sq = np.einsum("ti,ti->", R, R)
c_lw = T / (N + T)
gamma_lw = frob_sq / (N + T)
R_lw = np.vstack([np.sqrt(c_lw) * R, np.sqrt(gamma_lw) * np.eye(N)])

# ── Benchmark ──────────────────────────────────────────────────────────────────


def run_solver(name, fn, repeats=3):
    """Return (norm, time, iters) for a solver function."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    w, iters = result
    return {"name": name, "norm": float(np.linalg.norm(R @ w)), "time_s": best, "iters": iters}


configs_no_lw = [
    ("cvxpy", lambda: (minvar_cvxpy(R), None)),
    ("kkt", lambda: (minvar_kkt(R), None)),
    ("minres", lambda: minvar_minres(R)),
    ("cg", lambda: minvar_cg(R)),
]

configs_lw = [
    ("cvxpy", lambda: (minvar_cvxpy(R_lw), None)),
    ("kkt", lambda: (minvar_kkt(R_lw), None)),
    ("minres", lambda: minvar_minres(R, c=c_lw, gamma=gamma_lw)),
    ("cg", lambda: minvar_cg(R, c=c_lw, gamma=gamma_lw)),
]

display = {
    "cvxpy": "cvxpy (Clarabel)",
    "kkt": "KKT direct",
    "minres": "MINRES",
    "cg": "CG (constraint-eliminated)",
}

panels = [
    ("Without Ledoit-Wolf shrinkage", configs_no_lw),
    ("With Ledoit-Wolf shrinkage", configs_lw),
]

all_results = {}
for label, configs in panels:
    print(f"{label}")
    print(f"{'method':<30} {'norm':>10} {'time_s':>10} {'iters':>8}")
    print("-" * 62)
    panel = {}
    for key, fn in configs:
        r = run_solver(key, fn)
        panel[key] = r
        iters_str = str(r["iters"]) if r["iters"] is not None else "-"
        print(f"{display[key]:<30} {r['norm']:>10.6f} {r['time_s']:>10.4f} {iters_str:>8}")
    all_results[label] = panel
    print()

# ── LaTeX table ────────────────────────────────────────────────────────────────


def tex_row(dn, v, ref):
    """Format one LaTeX table row."""
    iters_str = str(v["iters"]) if v["iters"] is not None else "--"
    cols = f"{v['norm']:.4f} & {v['time_s']:.4f} & {iters_str:>6} & {ref / v['time_s']:.1f}x"
    return f"{dn:<35} & {cols} \\\\"


lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Method & $\|Xw\|$ & Time (s) & Iterations & Speedup \\",
    r"\midrule",
    r"\multicolumn{5}{l}{\textit{Without Ledoit-Wolf shrinkage}} \\[2pt]",
]
panel_no = all_results["Without Ledoit-Wolf shrinkage"]
ref_no = panel_no["cvxpy"]["time_s"]
for key in ("cvxpy", "kkt", "minres", "cg"):
    lines.append(tex_row(display[key], panel_no[key], ref_no))

lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{\textit{With Ledoit-Wolf shrinkage}} \\[2pt]")
panel_lw = all_results["With Ledoit-Wolf shrinkage"]
ref_lw = panel_lw["cvxpy"]["time_s"]
for key in ("cvxpy", "kkt", "minres", "cg"):
    lines.append(tex_row(display[key], panel_lw[key], ref_lw))

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    rf"\caption{{S\&P~500 universe: {N} assets, {T} trading days"
    rf" ({df.index[0].strftime('%b %Y')}--{df.index[-1].strftime('%b %Y')})."
    r" Speedup relative to \texttt{cvxpy} within each panel.}}",
    r"\label{tab:sp500}",
    r"\end{table}",
]

print("\n% LaTeX table")
print("\n".join(lines))
