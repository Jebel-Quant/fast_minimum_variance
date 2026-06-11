"""Experiments for the companion paper.

    "From Marchenko-Pastur to Woodbury: Direct Solvers for
     Long-Only Mean-Variance Portfolios"

Usage:
    uv run experiment_rmt.py      # from the experiment/ directory

Inputs:
    data/sp500_pct_returns.parquet   — S&P 500 daily pct returns
    Fetch with:  uv run fetch_sp500.py

Outputs (stdout):
    A: Preprocessing benchmark — dense eigendecomp vs randomised SVD,
       spectral diagnostics, direct dense-vs-rSVD portfolio comparison
    B: Solver comparison — CVXPY | Clarabel-direct | Cholesky | Woodbury,
       plus optimality cross-checks against interior-point weights
    C: k-sensitivity — portfolio change at k±1
    D: Scaling with preprocessing — runtime vs n
    E: Out-of-sample backtest — RMT-CRE vs LW shrinkage vs equal weight,
       on two universes (S&P 500, FTSE 100)
    F: Incremental eigenpair updates — daily rolling window, rank-two update
    G: Critical line algorithm with the Woodbury kernel — exact frontier

Outputs (files):
    graphs/rmt_frontier.pdf           — efficient frontier coloured by active-asset count
    graphs/rmt_scaling_full.pdf       — scaling: Cholesky vs Woodbury + preprocessing
    tables/rmt_preprocessing.tex
    tables/rmt_solver_comparison.tex
    tables/rmt_k_sensitivity.tex
    tables/rmt_oos.tex

Hardware: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.x.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "fast-minimum-variance",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# fast-minimum-variance = { path = ".." }
# ///

from __future__ import annotations

import time as _time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from util.runner import run_timed

from fast_minimum_variance.data import simulate_equity_returns
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.shrinkage.util import (
    lw_alpha_and_target,
    rmt_target_and_alpha,
)

HERE = Path(__file__).parent
GRAPHS = HERE / "graphs"
TABLES = HERE / "tables"
GRAPHS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

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

SP500_DATA = HERE / "data/sp500_pct_returns.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rmt_target_at_k(X, k):
    """Return (target, lr_factors, k) using exactly k signal factors.

    Trace-preserving CRE: the n-k bulk eigenvalues are replaced by their
    average, so tr(T0) = tr(S).
    """
    T, n = X.shape
    cov = (X.T @ X) / T
    eigs, vecs = np.linalg.eigh(cov)  # ascending
    eigs_k = eigs[-k:]
    vecs_k = vecs[:, -k:]
    bar_lam = float(eigs[:-k].mean())  # trace-preserving bulk mean
    delta_k = eigs_k - bar_lam
    target = bar_lam * np.eye(n) + vecs_k @ np.diag(delta_k) @ vecs_k.T
    lr_factors = (bar_lam, vecs_k, delta_k)
    return target, lr_factors, k


def rsvd_eigenpairs(X, k, p=10):
    """Compute top-k eigenpairs of X^T X / T via randomised SVD.

    The trace-preserving bulk mean is recovered without any dense
    decomposition: tr(S) = ||X||_F^2 / T, so
    bar_lam = (tr(S) - sum of the k signal eigenvalues) / (n - k).
    """
    T, n = X.shape
    _, s, Vt = randomized_svd(X, n_components=k + p, random_state=0)
    eigs_k = (s[:k] ** 2) / T
    vecs_k = Vt[:k].T  # (n, k)
    trace = float(np.linalg.norm(X, "fro") ** 2) / T
    bar_lam = (trace - float(eigs_k.sum())) / (n - k)
    return vecs_k, eigs_k, bar_lam


# ===========================================================================
# Section A: Preprocessing benchmark
# ===========================================================================

print("=" * 70)
print("A: Preprocessing benchmark  (dense eigendecomp vs randomised SVD)")
print("=" * 70)

df_sp = pd.read_parquet(SP500_DATA)
R_sp = df_sp.to_numpy()
R_sp = R_sp - R_sp.mean(axis=0)
T_sp, N_sp = R_sp.shape

_, _, k_sp, _ = rmt_target_and_alpha(R_sp)
print(f"S&P 500: n={N_sp}, T={T_sp}, k={k_sp} signal factors, p=10 oversampling")

# Dense path
cov_sp = (R_sp.T @ R_sp) / T_sp
bar_lam_sp = float(np.trace(cov_sp) / N_sp)

(eigs_dense, vecs_dense), t_dense = run_timed(lambda: np.linalg.eigh((R_sp.T @ R_sp) / T_sp))
U_dense = vecs_dense[:, -k_sp:]

# Randomised SVD path
(U_rsvd, eigs_rsvd, bar_rsvd_sp), t_rsvd = run_timed(lambda: rsvd_eigenpairs(R_sp, k_sp, p=10))

# Subspace distance
proj_dense = U_dense @ U_dense.T
proj_rsvd = U_rsvd @ U_rsvd.T
subspace_err = float(np.linalg.norm(proj_dense - proj_rsvd, "fro"))

# Spectral diagnostics for the paper (condition numbers, MP edge, bulk mean)
lam_max_sp = float(eigs_dense[-1])
lam_min_sp = float(eigs_dense[0])
lam_edge_sp = float(eigs_dense[-(k_sp + 1)])  # (k+1)-th eigenvalue, just below the MP edge
bar_bulk_sp = float(eigs_dense[:-k_sp].mean())  # trace-preserving bulk mean
mp_edge_sp = bar_lam_sp * (1.0 + np.sqrt(N_sp / T_sp)) ** 2
print(f"\n  sigma^2 = tr/n          = {bar_lam_sp:.4e}")
print(f"  bulk mean bar_lambda    = {bar_bulk_sp:.4e}")
print(f"  MP upper edge           = {mp_edge_sp:.4e}")
print(f"  lambda_1 / lambda_k+1   = {lam_max_sp:.4e} / {lam_edge_sp:.4e}")
print(f"  kappa(sample cov)       = {lam_max_sp / lam_min_sp:,.0f}")
print(f"  kappa(T0_RMT)           = {lam_max_sp / bar_bulk_sp:,.0f}")
print(f"  lambda_1 / bar_lambda   = {lam_max_sp / bar_bulk_sp:.1f}")

# Direct portfolio comparison: dense vs rSVD eigenpairs at the same k (alpha=1)
lr_dense_sp = (bar_bulk_sp, U_dense, eigs_dense[-k_sp:] - bar_bulk_sp)
lr_rsvd_sp = (bar_rsvd_sp, U_rsvd, eigs_rsvd - bar_rsvd_sp)
w_prep_dense, _ = MinVarProblem(R_sp, alpha=1.0, target_lr=lr_dense_sp).solve_kkt()
w_prep_rsvd, _ = MinVarProblem(R_sp, alpha=1.0, target_lr=lr_rsvd_sp).solve_kkt()
prep_w_diff_bp = float(np.abs(w_prep_dense - w_prep_rsvd).max()) * 1e4
print(f"  max |w_dense - w_rsvd|  = {prep_w_diff_bp:.3f} bp  (direct portfolio impact of rSVD)")

print(f"\n  {'Method':<35} {'Time (s)':>10} {'Storage':>10} {'Subspace err':>14}")
print(f"  {'-' * 75}")
print(f"  {'Dense eigendecomp (eigh)':<35} {t_dense:>10.4f} {'n² floats':>10} {'---':>14}")
print(f"  {'Randomised SVD (p=10)':<35} {t_rsvd:>10.4f} {'nk floats':>10} {subspace_err:>14.2e}")
print(f"\n  Speedup: {t_dense / t_rsvd:.1f}x")
print(f"  Storage ratio: {N_sp**2 / (N_sp * k_sp):.0f}x  ({N_sp**2} vs {N_sp * k_sp} floats)")

_mant, _expo = f"{subspace_err:.1e}".split("e")
prep_table_lines = (
    f"Dense eigendecomposition & {t_dense:.4f} & $n^2$ & --- \\\\\n"
    f"Randomised SVD ($p=10$)  & {t_rsvd:.4f} & $nk$ & ${_mant}\\times10^{{{int(_expo)}}}$ \\\\\n"
    f"\\midrule\n"
    f"Speedup & ${t_dense / t_rsvd:.1f}\\times$ & ${N_sp**2 // (N_sp * k_sp)}\\times$ & \\\\\n"
)
(TABLES / "rmt_preprocessing.tex").write_text(f"\\def\\dataPreprocessing{{%\n{prep_table_lines}}}\n")
print("  → wrote tables/rmt_preprocessing.tex")


# ===========================================================================
# Section B: Solver comparison
#            CVXPY | KKT-Cholesky | Woodbury on the same RMT estimator
#            n=500, T=1250, synthetic, 50-point efficient frontier
# ===========================================================================

print()
print("=" * 70)
print("B: Solver comparison  (CVXPY | Cholesky | Woodbury, all on RMT, n=500, T=1250)")
print("=" * 70)

n_ef, T_ef = 500, 1250
R_ef = simulate_equity_returns(n_ef, T_ef, rng=42)
rng_ef = np.random.default_rng(42)
betas_ef = rng_ef.uniform(0.4, 0.8, n_ef)
mu_ef = betas_ef * (0.10 / 250)

tgt_rmt_ef, lr_rmt_ef, k_rmt_ef, alpha_rmt_ef = rmt_target_and_alpha(R_ef)
Sigma_rmt_ef = tgt_rmt_ef

rhos_ef = np.linspace(0, 2, 50)
N_PTS = len(rhos_ef)

print(f"  RMT: alpha={alpha_rmt_ef:.4f}, k={k_rmt_ef} signal factors")


def _cold_sweep(solve_fn_name, prob_fn, repeats=3):
    best_total = float("inf")
    best_times = None
    for _ in range(repeats):
        times = []
        for rho in rhos_ef:
            p = prob_fn(rho)
            t0 = _time.perf_counter()
            getattr(p, solve_fn_name)()
            times.append(_time.perf_counter() - t0)
        if sum(times) < best_total:
            best_total = sum(times)
            best_times = times
    return best_times


def _warm_sweep_kkt(prob_fn, repeats=3):
    best_total = float("inf")
    best_times = None
    for _ in range(repeats):
        times = []
        warm = None
        for rho in rhos_ef:
            p = prob_fn(rho)
            t0 = _time.perf_counter()
            _, _, warm = p.solve_kkt_warm(warm_start=warm)
            times.append(_time.perf_counter() - t0)
        if sum(times) < best_total:
            best_total = sum(times)
            best_times = times
    return best_times


print("  Running sweeps...")

# CVXPY reference (cold only — warm-starting not available)
t_cvxpy_cold = []
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
    t0 = _time.perf_counter()
    p.solve_cvxpy()
    t_cvxpy_cold.append(_time.perf_counter() - t0)
cvxpy_c = sum(t_cvxpy_cold)

# Clarabel direct API (cold only) — interior-point without CVXPY's construction overhead
t_clarabel_cold = []
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
    t0 = _time.perf_counter()
    p.solve_clarabel()
    t_clarabel_cold.append(_time.perf_counter() - t0)
clarabel_c = sum(t_clarabel_cold)

# KKT-Cholesky: solve_kkt WITHOUT target_lr (assembles n_a x n_a, then Cholesky)
t_kkt_cold = _cold_sweep(
    "solve_kkt", lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
)
t_kkt_warm = _warm_sweep_kkt(lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef))

# Woodbury: solve_kkt WITH target_lr (never assembles n_a x n_a matrix)
t_wb_cold = _cold_sweep(
    "solve_kkt",
    lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef),
)
t_wb_warm = _warm_sweep_kkt(
    lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
)

print(f"\n  {'Solver':<42} {'Cold (s)':>9} {'ms/pt':>7} {'Warm (s)':>9} {'ms/pt':>7} {'WB speedup':>11}")
print(f"  {'-' * 95}")

kkt_c, kkt_w = sum(t_kkt_cold), sum(t_kkt_warm)
wb_c, wb_w = sum(t_wb_cold), sum(t_wb_warm)
print(f"  {'CVXPY (Clarabel)':<42} {cvxpy_c:>9.3f} {cvxpy_c / N_PTS * 1000:>7.1f} {'---':>9} {'---':>7} {'---':>11}")
print(
    f"  {'Clarabel (direct)':<42} {clarabel_c:>9.3f} {clarabel_c / N_PTS * 1000:>7.1f} "
    f"{'---':>9} {'---':>7} {'---':>11}"
)
print(
    f"  {'KKT-Cholesky':<42} {kkt_c:>9.3f} {kkt_c / N_PTS * 1000:>7.1f} "
    f"{kkt_w:>9.3f} {kkt_w / N_PTS * 1000:>7.1f} {'---':>11}"
)
print(
    f"  {'Woodbury':<42} {wb_c:>9.3f} {wb_c / N_PTS * 1000:>7.1f} "
    f"{wb_w:>9.3f} {wb_w / N_PTS * 1000:>7.1f} {kkt_w / wb_w:>10.1f}x"
)

print(f"\n  Woodbury vs KKT-Cholesky (cold):  {kkt_c / wb_c:.1f}x")
print(f"  Woodbury vs KKT-Cholesky (warm):  {kkt_w / wb_w:.1f}x")
print(f"  Woodbury vs CVXPY (cold):         {cvxpy_c / wb_c:.0f}x")
print(f"  Woodbury vs Clarabel-direct (cold): {clarabel_c / wb_c:.0f}x")

# Optimality cross-check: Woodbury solution vs interior-point solutions
# (verifies the active-set KKT conditions against an independent solver)
for rho_chk, lbl in [(0.0, "rho=0"), (float(rhos_ef[25]), f"rho={rhos_ef[25]:.2f}")]:
    p_wb_chk = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho_chk, mu=mu_ef)
    p_ip_chk = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho_chk, mu=mu_ef)
    w_wb_chk, _ = p_wb_chk.solve_kkt()
    w_cvx_chk, _ = p_ip_chk.solve_cvxpy()
    w_cla_chk, _ = p_ip_chk.solve_clarabel()
    print(
        f"  Weight agreement ({lbl}): |w_WB - w_CVXPY|_inf = {float(np.abs(w_wb_chk - w_cvx_chk).max()):.2e}, "
        f"|w_WB - w_Clarabel|_inf = {float(np.abs(w_wb_chk - w_cla_chk).max()):.2e}"
    )

# Woodbury warm sweep: capture frontier points and active-set sizes
ef_vols_rmt, ef_rets_rmt, active_sizes = [], [], []
warm_rmt = None
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
    w, _, warm_rmt = p.solve_kkt_warm(warm_start=warm_rmt)
    ef_vols_rmt.append(float(np.sqrt(w @ Sigma_rmt_ef @ w)) * np.sqrt(250) * 100)
    ef_rets_rmt.append(float(w @ mu_ef) * 250 * 100)
    active_sizes.append(int((w > 1e-6).sum()))

# KKT-Cholesky warm sweep: frontier overlay (should match Woodbury numerically)
ef_vols_kkt, ef_rets_kkt = [], []
warm_kkt = None
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
    w, _, warm_kkt = p.solve_kkt_warm(warm_start=warm_kkt)
    ef_vols_kkt.append(float(np.sqrt(w @ Sigma_rmt_ef @ w)) * np.sqrt(250) * 100)
    ef_rets_kkt.append(float(w @ mu_ef) * 250 * 100)

print(
    f"\n  Active-set sizes (Woodbury warm): mean={np.mean(active_sizes):.1f}, "
    f"min={min(active_sizes)}, max={max(active_sizes)}"
)
max_diff = float(np.max(np.abs(np.array(ef_vols_rmt) - np.array(ef_vols_kkt))))
print(f"  Max vol difference (WB vs KKT-Chol): {max_diff:.2e}% (numerical precision check)")

# S&P 500 single min-var solve (cold and warm) —
# uses the real return matrix already loaded in Section A
tgt_sp_b, lr_sp_b, k_sp_b, alpha_sp_b = rmt_target_and_alpha(R_sp)


def _sp_cold(solve_fn_name, prob, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = _time.perf_counter()
        getattr(prob, solve_fn_name)()
        best = min(best, _time.perf_counter() - t0)
    return best


p_cvxpy_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b)
p_kkt_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b)
p_wb_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b, target_lr=lr_sp_b)

cvxpy_sp_c = _sp_cold("solve_cvxpy", p_cvxpy_sp)
clarabel_sp_c = _sp_cold("solve_clarabel", p_cvxpy_sp)
kkt_sp_c = _sp_cold("solve_kkt", p_kkt_sp)
wb_sp_c = _sp_cold("solve_kkt", p_wb_sp)

# Warm: one immediate re-solve after a cold run (active-set already converged)
_, _, warm_kkt_sp = p_kkt_sp.solve_kkt_warm(warm_start=None)
t0 = _time.perf_counter()
p_kkt_sp.solve_kkt_warm(warm_start=warm_kkt_sp)
kkt_sp_w = _time.perf_counter() - t0

_, _, warm_wb_sp = p_wb_sp.solve_kkt_warm(warm_start=None)
t0 = _time.perf_counter()
p_wb_sp.solve_kkt_warm(warm_start=warm_wb_sp)
wb_sp_w = _time.perf_counter() - t0

print(f"\n  S&P 500 single min-var solve (n={N_sp}, T={T_sp}, k={k_sp_b}):")
print(f"    CVXPY cold:          {cvxpy_sp_c * 1000:>7.1f} ms")
print(f"    Clarabel direct cold:{clarabel_sp_c * 1000:>7.1f} ms")
print(f"    KKT-Cholesky cold:   {kkt_sp_c * 1000:>7.1f} ms   warm: {kkt_sp_w * 1000:.1f} ms")
print(f"    Woodbury cold:       {wb_sp_c * 1000:>7.1f} ms   warm: {wb_sp_w * 1000:.1f} ms")
print(f"    Woodbury vs KKT-Chol (cold): {kkt_sp_c / wb_sp_c:.1f}x")

# Optimality cross-check on real data
w_wb_sp, _ = p_wb_sp.solve_kkt()
w_cla_sp, _ = p_cvxpy_sp.solve_clarabel()
print(f"    Weight agreement: |w_WB - w_Clarabel|_inf = {float(np.abs(w_wb_sp - w_cla_sp).max()):.2e}")


# Write combined two-panel solver comparison table
def _fmt(t):
    """Format seconds as e.g. 88.1 or 0.042."""
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.3f}"
    if t >= 0.1:
        return f"{t:.4f}"
    return f"{t:.4f}"


def _ms(t):
    return f"{t * 1000:.1f}"


synth_rows = (
    f"{'CVXPY (Clarabel)':<38} & {_fmt(cvxpy_c):>7} & {_ms(cvxpy_c / N_PTS):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'Clarabel (direct)':<38} & {_fmt(clarabel_c):>7} & {_ms(clarabel_c / N_PTS):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'KKT-Cholesky ($k=' + str(k_rmt_ef) + '$)':<38} & {_fmt(kkt_c):>7} & {_ms(kkt_c / N_PTS):>6}"
    f" & {_fmt(kkt_w):>7} & {_ms(kkt_w / N_PTS):>6} \\\\\n"
    f"{'Woodbury ($k=' + str(k_rmt_ef) + '$)':<38} & {_fmt(wb_c):>7} & {_ms(wb_c / N_PTS):>6}"
    f" & {_fmt(wb_w):>7} & {_ms(wb_w / N_PTS):>6} \\\\\n"
)
sp_rows = (
    f"{'CVXPY (Clarabel)':<38} & {_fmt(cvxpy_sp_c):>7} & {_ms(cvxpy_sp_c):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'Clarabel (direct)':<38} & {_fmt(clarabel_sp_c):>7} & {_ms(clarabel_sp_c):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'KKT-Cholesky ($k=' + str(k_sp_b) + '$)':<38} & {_fmt(kkt_sp_c):>7} & {_ms(kkt_sp_c):>6}"
    f" & {_fmt(kkt_sp_w):>7} & {_ms(kkt_sp_w):>6} \\\\\n"
    f"{'Woodbury ($k=' + str(k_sp_b) + '$)':<38} & {_fmt(wb_sp_c):>7} & {_ms(wb_sp_c):>6}"
    f" & {_fmt(wb_sp_w):>7} & {_ms(wb_sp_w):>6} \\\\\n"
)

panel_a = (
    "\\multicolumn{5}{l}{\\textit{Synthetic, $n=500$, $T=1250$, $k=22$,"
    " 50-point sweep (total / ms per point)}} \\\\\n"
    "\\addlinespace[2pt]\n"
)
panel_b = (
    "\\multicolumn{5}{l}{\\textit{S\\&P~500, $n=494$, $T=1213$, $k=23$,"
    " single min-var solve (seconds / ms)}} \\\\\n"
    "\\addlinespace[2pt]\n"
)

(TABLES / "rmt_solver_comparison.tex").write_text(
    "\\def\\dataRmtSolverComp{%\n" + panel_a + synth_rows + "\\midrule\n" + panel_b + sp_rows + "}\n"
)
print("  → wrote tables/rmt_solver_comparison.tex")

# Efficient frontier figure: Woodbury (coloured by active assets) + KKT-Cholesky (dashed overlay)
fig_ef, ax_ef = plt.subplots(figsize=(4.8, 3.4))
sc = ax_ef.scatter(
    ef_vols_rmt, ef_rets_rmt, c=active_sizes, cmap="plasma_r", s=18, zorder=3, label=rf"Woodbury (RMT, $k={k_rmt_ef}$)"
)
ax_ef.plot(ef_vols_rmt, ef_rets_rmt, color="gray", linewidth=0.7, zorder=2)
ax_ef.plot(
    ef_vols_kkt,
    ef_rets_kkt,
    color="steelblue",
    linewidth=1.2,
    linestyle="--",
    zorder=4,
    label=rf"KKT-Cholesky (RMT, $k={k_rmt_ef}$)",
)
ax_ef.scatter(
    [ef_vols_rmt[0]], [ef_rets_rmt[0]], marker="*", s=110, color="#ff7f0e", zorder=5, label="Min-var (Woodbury)"
)
cbar_ef = fig_ef.colorbar(sc, ax=ax_ef, pad=0.02)
cbar_ef.set_label("Active assets")
ax_ef.set_xlabel("Annualised volatility (%)")
ax_ef.set_ylabel("Annualised expected return (%)")
ax_ef.set_title(rf"Efficient frontier ($n={n_ef}$, $T={T_ef}$, RMT $k={k_rmt_ef}$)")
ax_ef.legend(framealpha=0.9, loc="lower right", fontsize=7)
ax_ef.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
fig_ef.tight_layout(pad=1.0)
fig_ef.savefig(GRAPHS / "rmt_frontier.pdf", bbox_inches="tight")
fig_ef.savefig(GRAPHS / "rmt_frontier.png", bbox_inches="tight", dpi=150)
print("  → saved graphs/rmt_frontier.pdf")
plt.close(fig_ef)


# ===========================================================================
# Section C: k-sensitivity
# ===========================================================================

print()
print("=" * 70)
print("C: k-sensitivity on S&P 500  (portfolios at k-1, k, k+1)")
print("=" * 70)

k_ref = k_sp
k_vals = [k_ref - 1, k_ref, k_ref + 1]
w_at_k = {}

for k_try in k_vals:
    tgt_k, lr_k, _ = rmt_target_at_k(R_sp, k_try)
    prob_k = MinVarProblem(R_sp, alpha=1.0, target=tgt_k, target_lr=lr_k)
    w_k, _ = prob_k.solve_kkt()
    w_at_k[k_try] = w_k

w_ref = w_at_k[k_ref]
cov_sp_full = (R_sp.T @ R_sp) / T_sp

print(
    f"\n  k={k_ref} reference: {int((w_ref > 1e-6).sum())} active assets, "
    f"vol={float(np.sqrt(w_ref @ cov_sp_full @ w_ref)) * np.sqrt(250) * 100:.3f}% ann."
)
print(f"\n  {'k':>4} {'Active':>8} {'Ann.Vol(%)':>12} {'||w-w*||_inf':>14} {'||w-w*||_2':>12}")
print(f"  {'-' * 55}")
for k_try in k_vals:
    w = w_at_k[k_try]
    vol = float(np.sqrt(w @ cov_sp_full @ w)) * np.sqrt(250) * 100
    active = int((w > 1e-6).sum())
    diff_inf = float(np.abs(w - w_ref).max())
    diff_2 = float(np.linalg.norm(w - w_ref))
    marker = " ← reference" if k_try == k_ref else ""
    print(f"  {k_try:>4} {active:>8} {vol:>12.3f} {diff_inf:>14.4f} {diff_2:>12.4f}{marker}")

k_sens_lines = ""
for k_try in k_vals:
    w = w_at_k[k_try]
    vol = float(np.sqrt(w @ cov_sp_full @ w)) * np.sqrt(250) * 100
    active = int((w > 1e-6).sum())
    diff_inf = float(np.abs(w - w_ref).max())
    diff_2 = float(np.linalg.norm(w - w_ref))
    k_sens_lines += f"{k_try} & {active} & {vol:.2f} & {diff_inf:.4f} & {diff_2:.4f} \\\\\n"

(TABLES / "rmt_k_sensitivity.tex").write_text(f"\\def\\dataKsensitivity{{%\n{k_sens_lines}}}\n")
print("  → wrote tables/rmt_k_sensitivity.tex")


# ===========================================================================
# Section D: Scaling — Cholesky vs Woodbury + preprocessing
# ===========================================================================

print()
print("=" * 70)
print("D: Scaling  (Cholesky vs Woodbury + preprocessing, T=1250 fixed)")
print("=" * 70)

T_FIXED = 1250
ns = [300, 500, 750, 1000, 1500, 2000, 3000]
t_kkt_scale = []  # KKT-Cholesky (no target_lr)
t_wb_solve = []  # Woodbury solve only
t_wb_dense_prep = []  # dense eigendecomp preprocessing
t_wb_rsvd_prep = []  # randomised SVD preprocessing
t_wb_total_dense = []
t_wb_total_rsvd = []
k_detected = []

print(
    f"\n  {'n':>5} {'k':>4} {'Cholesky(s)':>13} {'WB-solve(s)':>12} "
    f"{'Dense-prep(s)':>14} {'rSVD-prep(s)':>13} {'WB+dense(s)':>12} {'WB+rSVD(s)':>11}"
)
print(f"  {'-' * 95}")

for n in ns:
    R_s = simulate_equity_returns(n, T_FIXED, rng=n)

    tgt_s_rmt, lr_s_rmt, k_s, _ = rmt_target_and_alpha(R_s)

    # KKT-Cholesky (no target_lr): assembles n_a x n_a matrix, then Cholesky
    prob_kkt = MinVarProblem(R_s, alpha=1.0, target=tgt_s_rmt)
    (_, _), t_kkt = run_timed(lambda p=prob_kkt: p.solve_kkt())

    # Dense preprocessing
    _, t_dense_s = run_timed(lambda R=R_s: np.linalg.eigh((R.T @ R) / T_FIXED))

    # Woodbury solve only (using dense eigenpairs, preprocessing already done)
    prob_wb = MinVarProblem(R_s, alpha=1.0, target=tgt_s_rmt, target_lr=lr_s_rmt)
    (_, _), t_wb = run_timed(lambda p=prob_wb: p.solve_kkt())

    # Randomised SVD preprocessing
    _, t_rsvd_s = run_timed(lambda R=R_s, k=k_s: rsvd_eigenpairs(R, k, p=10))

    t_kkt_scale.append(t_kkt)
    t_wb_solve.append(t_wb)
    t_wb_dense_prep.append(t_dense_s)
    t_wb_rsvd_prep.append(t_rsvd_s)
    t_wb_total_dense.append(t_dense_s + t_wb)
    t_wb_total_rsvd.append(t_rsvd_s + t_wb)
    k_detected.append(k_s)

    print(
        f"  {n:>5} {k_s:>4} {t_kkt:>13.4f} {t_wb:>12.4f} "
        f"{t_dense_s:>14.4f} {t_rsvd_s:>13.4f} "
        f"{t_dense_s + t_wb:>12.4f} {t_rsvd_s + t_wb:>11.4f}"
    )

# Scaling figure
fig_sc, ax_sc = plt.subplots(figsize=(4.8, 3.4))
n_arr = np.array(ns, dtype=float)

ax_sc.plot(ns, t_kkt_scale, marker="o", markersize=4, color="#ff7f0e", label=r"KKT-Cholesky (assemble + Chol)")
ax_sc.plot(ns, t_wb_solve, marker="s", markersize=4, color="#2ca02c", label=r"Woodbury solve only")
ax_sc.plot(ns, t_wb_total_rsvd, marker="^", markersize=4, color="#1f77b4", label=r"Woodbury + rSVD prep")
ax_sc.plot(
    ns, t_wb_total_dense, marker="v", markersize=4, color="#d62728", linestyle="--", label=r"Woodbury + dense prep"
)

idx_500 = ns.index(500) if 500 in ns else 0
t_ref = t_kkt_scale[idx_500]
ax_sc.plot(n_arr, t_ref * (n_arr / 500.0), color="gray", linestyle=":", linewidth=0.9, label=r"$O(n)$")
ax_sc.plot(
    n_arr, t_ref * (n_arr / 500.0) ** 2, color="gray", linestyle=(0, (3, 1, 1, 1)), linewidth=0.9, label=r"$O(n^2)$"
)

ax_sc.set_xscale("log")
ax_sc.set_yscale("log")
ax_sc.set_xlabel("Number of assets $n$")
ax_sc.set_ylabel("Wall-clock time (s)")
ax_sc.set_title(r"Scaling with $n$ ($T=1250$): Cholesky vs Woodbury")
ax_sc.legend(framealpha=0.9, fontsize=7, loc="upper left")
ax_sc.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig_sc.tight_layout(pad=1.0)
fig_sc.savefig(GRAPHS / "rmt_scaling_full.pdf", bbox_inches="tight")
fig_sc.savefig(GRAPHS / "rmt_scaling_full.png", bbox_inches="tight", dpi=150)
print("\n  → saved graphs/rmt_scaling_full.pdf")
plt.close(fig_sc)


# ===========================================================================
# Section E: Out-of-sample backtest — RMT vs LW shrinkage vs equal weight
#            on two universes: S&P 500 (n/L ~ 0.98) and FTSE 100 (n/L ~ 0.17)
# ===========================================================================

print()
print("=" * 70)
print("E: Out-of-sample backtest  (S&P 500 + FTSE 100, monthly rebalance)")
print("=" * 70)

HOLD = 21  # holding period (~1 trading month)
EST = 504  # estimation window (~2 years)
ANN = 252

STRATEGIES = ["RMT", "LW-0.5", "LW-oracle", "Equal-weight"]
OOS_LABELS = {
    "RMT": "RMT-CRE ($\\alpha=1$)",
    "LW-0.5": "LW ($\\alpha=0.5$)",
    "LW-oracle": "LW (oracle $\\alpha$)",
    "Equal-weight": "Equal-weight",
}


def _drift(w_prev, r_block):
    """Buy-and-hold drifted weights at the end of a holding block."""
    growth = np.prod(1.0 + r_block, axis=0)
    w_d = w_prev * growth
    s = w_d.sum()
    return w_d / s if s > 0 else w_prev


def run_oos_backtest(R_raw, label):
    """Rolling-window OOS backtest; returns (table_lines, n_windows, k_range).

    Assets with zero full-sample variance or more than 20% exact-zero returns
    (delisted names forward-filled flat, late listings padded with zeros) are
    excluded: a zero-variance asset absorbs the entire minimum-variance
    portfolio and renders the backtest meaningless.
    """
    stds = R_raw.std(axis=0)
    zero_frac = (R_raw == 0).mean(axis=0)
    keep = (stds > 0) & (zero_frac <= 0.20)
    if (~keep).any():
        print(f"  {label}: excluded {int((~keep).sum())} dead/stale asset(s) of {R_raw.shape[1]}")
    R_raw = R_raw[:, keep]
    T_raw, n_u = R_raw.shape
    oos_daily = {s: [] for s in STRATEGIES}
    oos_turnover = {s: [] for s in STRATEGIES}
    prev_w = dict.fromkeys(STRATEGIES)
    prev_block = dict.fromkeys(STRATEGIES)
    n_windows = 0
    k_seen = []

    for t in range(EST, T_raw - 1, HOLD):
        R_in = R_raw[t - EST : t]
        X_in = R_in - R_in.mean(axis=0)
        R_out = R_raw[t : t + HOLD]
        if R_out.shape[0] == 0:
            continue
        n_windows += 1

        bar_in = float(np.linalg.norm(X_in, "fro") ** 2) / (n_u * EST)
        tgt_id = bar_in * np.eye(n_u)
        a_oracle, _ = lw_alpha_and_target(X_in)

        tgt_in, lr_in, k_in, _ = rmt_target_and_alpha(X_in)
        k_seen.append(k_in)
        weights = {
            "RMT": MinVarProblem(X_in, alpha=1.0, target=tgt_in, target_lr=lr_in).solve_kkt()[0],
            "LW-0.5": MinVarProblem(X_in, alpha=0.5, target=tgt_id).solve_kkt()[0],
            "LW-oracle": MinVarProblem(X_in, alpha=a_oracle, target=tgt_id).solve_kkt()[0],
            "Equal-weight": np.full(n_u, 1.0 / n_u),
        }

        for s in STRATEGIES:
            w = weights[s]
            oos_daily[s].extend((R_out @ w).tolist())
            if prev_w[s] is not None:
                w_drift = _drift(prev_w[s], prev_block[s])
                oos_turnover[s].append(0.5 * float(np.abs(w - w_drift).sum()))
            prev_w[s] = w
            prev_block[s] = R_out

    print(
        f"\n  {label}: n={n_u}, {n_windows} rebalances, L={EST} (n/L={n_u / EST:.2f}), "
        f"H={HOLD}, k in [{min(k_seen)}, {max(k_seen)}]"
    )
    print(f"  {'Strategy':<16} {'OOSvol%':>8} {'ret%':>7} {'Sharpe':>7} {'turn%':>7}")
    print(f"  {'-' * 50}")
    lines = ""
    for s in STRATEGIES:
        d = np.asarray(oos_daily[s])
        ann_vol = float(d.std(ddof=1) * np.sqrt(ANN) * 100)
        ann_ret = float(d.mean() * ANN * 100)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        turn = float(np.mean(oos_turnover[s]) * 100) if oos_turnover[s] else 0.0
        print(f"  {s:<16} {ann_vol:>8.2f} {ann_ret:>7.2f} {sharpe:>7.3f} {turn:>7.1f}")
        lines += f"{OOS_LABELS[s]} & {ann_vol:.2f} & {sharpe:.3f} & {turn:.1f} \\\\\n"
    return lines, n_windows, (min(k_seen), max(k_seen)), n_u


sp_oos_lines, sp_oos_windows, sp_oos_k, sp_oos_n = run_oos_backtest(df_sp.to_numpy(), "S&P 500")

df_ftse = pd.read_parquet(HERE / "data/ftse100_pct_returns.parquet")
ftse_oos_lines, ftse_oos_windows, ftse_oos_k, ftse_oos_n = run_oos_backtest(df_ftse.to_numpy(), "FTSE 100")

panel_sp = (
    f"\\multicolumn{{4}}{{l}}{{\\textit{{S\\&P~500: $n={sp_oos_n}$, $n/L\\approx{sp_oos_n / EST:.2f}$,"
    f" {sp_oos_windows} rebalances, $k\\in[{sp_oos_k[0]},{sp_oos_k[1]}]$}}}} \\\\\n"
    "\\addlinespace[2pt]\n"
)
panel_ftse = (
    f"\\multicolumn{{4}}{{l}}{{\\textit{{FTSE~100: $n={ftse_oos_n}$, $n/L\\approx{ftse_oos_n / EST:.2f}$,"
    f" {ftse_oos_windows} rebalances, $k\\in[{ftse_oos_k[0]},{ftse_oos_k[1]}]$}}}} \\\\\n"
    "\\addlinespace[2pt]\n"
)
(TABLES / "rmt_oos.tex").write_text(
    "\\def\\dataRmtOos{%\n" + panel_sp + sp_oos_lines + "\\midrule\n" + panel_ftse + ftse_oos_lines + "}\n"
)
print("\n  → wrote tables/rmt_oos.tex")


# ===========================================================================
# Section F: Incremental eigenpair updates for daily rolling windows
#            (rank-two update of the projected covariance; no re-sketch)
# ===========================================================================

print()
print("=" * 70)
print("F: Incremental eigenpair updates  (daily rolling window, S&P 500)")
print("=" * 70)


def incremental_eigenpair_update(Q, lam, trace, x_new, x_old, T, k):
    """One daily rolling-window update of the tracked eigenpair state.

    State: orthonormal basis Q (n, m), compressed eigenvalues lam (m,),
    and the running trace of Sigma = X^T X / T.  The window update
    Sigma += (x_new x_new^T - x_old x_old^T)/T is rank-two; the basis is
    expanded by the residuals of x_new, x_old, the (m+2)x(m+2) projected
    matrix is re-diagonalised, and the top m directions are retained.
    Outside the tracked basis Sigma is approximated by its bulk floor,
    whose effect on the update is captured through the trace.
    Cost: O(n m + n m^2) for projection/rotation plus O(m^3) for eigh.
    """
    n, m = Q.shape
    trace = trace + (float(x_new @ x_new) - float(x_old @ x_old)) / T

    ext = [Q]
    for x in (x_new, x_old):
        r = x - Q @ (Q.T @ x)
        for prev in ext[1:]:
            r = r - prev * float(prev @ r)
        nrm = float(np.linalg.norm(r))
        if nrm > 1e-10:
            ext.append(r / nrm)
    Q_ext = np.column_stack(ext)
    m_ext = Q_ext.shape[1]

    bulk_floor = max((trace - float(lam[:k].sum())) / (n - k), 1e-16)
    diag_ext = np.concatenate([lam, np.full(m_ext - m, bulk_floor)])
    c_new = Q_ext.T @ x_new
    c_old = Q_ext.T @ x_old
    B = np.diag(diag_ext) + (np.outer(c_new, c_new) - np.outer(c_old, c_old)) / T

    eigs, vecs = np.linalg.eigh(B)  # ascending
    order = np.argsort(eigs)[::-1][:m]
    lam_new = eigs[order]
    Q_new = Q_ext @ vecs[:, order]
    return Q_new, lam_new, trace


# Setup: track eigenpairs over N_DAYS daily updates of a rolling L-window
N_DAYS = 100
EST_F = 504
R_raw_sp = df_sp.to_numpy()
m_track = k_sp + 10  # tracked subspace: k + p

# Initial sketch on window [0, EST_F)  (raw returns; the mean update is a
# further rank-one correction, omitted -- daily means are negligible)
X0 = R_raw_sp[:EST_F]
U0, e0, _ = rsvd_eigenpairs(X0, m_track, p=0)
state_q, state_lam = U0, e0
state_trace = float(np.linalg.norm(X0, "fro") ** 2) / EST_F


def _minvar_from_state(Q, lam, trace, k):
    """Min-var portfolio from tracked state via the MP threshold and Woodbury."""
    n = Q.shape[0]
    sigma2 = trace / n
    mp_edge = sigma2 * (1.0 + np.sqrt(n / EST_F)) ** 2
    k_eff = max(int((lam > mp_edge).sum()), 1)
    bar = (trace - float(lam[:k_eff].sum())) / (n - k_eff)
    lr = (bar, Q[:, :k_eff], lam[:k_eff] - bar)
    dummy_X = np.zeros((2, n))  # X unused at alpha=1 (target_lr path)
    return MinVarProblem(dummy_X, alpha=1.0, target_lr=lr).solve_kkt()[0]


t_inc_total, t_fresh_total = 0.0, 0.0
w_diffs_bp = []
for d in range(N_DAYS):
    x_old = R_raw_sp[d]
    x_new = R_raw_sp[EST_F + d]

    t0 = _time.perf_counter()
    state_q, state_lam, state_trace = incremental_eigenpair_update(
        state_q, state_lam, state_trace, x_new, x_old, EST_F, k_sp
    )
    t_inc_total += _time.perf_counter() - t0

    X_win = R_raw_sp[d + 1 : EST_F + d + 1]
    t0 = _time.perf_counter()
    U_f, e_f, _ = rsvd_eigenpairs(X_win, m_track, p=0)
    t_fresh_total += _time.perf_counter() - t0
    trace_f = float(np.linalg.norm(X_win, "fro") ** 2) / EST_F

    w_inc = _minvar_from_state(state_q, state_lam, state_trace, k_sp)
    w_fresh = _minvar_from_state(U_f, e_f, trace_f, k_sp)
    w_diffs_bp.append(float(np.abs(w_inc - w_fresh).max()) * 1e4)

w_diffs_bp = np.asarray(w_diffs_bp)
print(f"  {N_DAYS} daily updates of an L={EST_F} window (n={N_sp}, tracked m={m_track})")
print(
    f"  per-update time:   incremental {t_inc_total / N_DAYS * 1000:.2f} ms,"
    f"  fresh rSVD {t_fresh_total / N_DAYS * 1000:.2f} ms"
    f"  ({t_fresh_total / t_inc_total:.0f}x speedup)"
)
print(
    f"  weight drift vs fresh rSVD (max |dw|, bp):"
    f"  median {np.median(w_diffs_bp):.2f},  mean {w_diffs_bp.mean():.2f},"
    f"  max {w_diffs_bp.max():.2f}  (after {N_DAYS} days, no re-sketch)"
)

inc_speedup = t_fresh_total / t_inc_total
inc_ms = t_inc_total / N_DAYS * 1000
fresh_ms = t_fresh_total / N_DAYS * 1000


# ===========================================================================
# Section G: Critical line algorithm with the Woodbury kernel
#            (exact parametric frontier; single-asset events)
# ===========================================================================

print()
print("=" * 70)
print("G: CLA with Woodbury kernel  (exact frontier, synthetic n=500)")
print("=" * 70)


def cla_sweep_woodbury(lr, mu, rho_max, w0, tol=1e-9, max_events=10_000):
    """Trace the exact long-only frontier for rho in [0, rho_max].

    On a fixed active set the solution is affine in rho:
    w_A(rho) = a + rho*b with a = v1/s1, b = (v2 - (s2/s1) v1)/2, where
    v1 = T0_A^{-1} 1, v2 = T0_A^{-1} mu_A (both via Woodbury) and
    s1 = 1^T v1, s2 = 1^T v2.  The KKT multiplier nu(rho) = 2*theta(rho) is
    affine as well, so primal exits (w_i hits 0) and dual entries
    (g_i - nu hits 0) are roots of affine functions; the sweep jumps from
    breakpoint to breakpoint, one Woodbury solve per segment.

    Returns (segments, n_events) where each segment is
    (rho_lo, rho_hi, active_idx, a, b).
    """
    bar_lam, U_k, delta_k = lr
    n = U_k.shape[0]
    active = w0 > tol  # converged min-var active set at rho = 0
    rho = 0.0
    segments = []

    for _ in range(max_events):
        idx = np.where(active)[0]
        Ua = U_k[idx, :]
        W = np.diag(1.0 / delta_k) + (Ua.T @ Ua) / bar_lam

        def wb(bvec, Ua=Ua, W=W):
            return bvec / bar_lam - Ua @ (np.linalg.solve(W, Ua.T @ bvec) / bar_lam**2)

        v1 = wb(np.ones(idx.size))
        v2 = wb(mu[idx])
        s1, s2 = float(v1.sum()), float(v2.sum())
        a_seg = v1 / s1
        b_seg = 0.5 * (v2 - (s2 / s1) * v1)

        # gradient g(rho) = ga + rho*gb on the full universe (O(nk) matvecs)
        a_full = np.zeros(n)
        a_full[idx] = a_seg
        b_full = np.zeros(n)
        b_full[idx] = b_seg
        t0a = bar_lam * a_full + U_k @ (delta_k * (U_k.T @ a_full))
        t0b = bar_lam * b_full + U_k @ (delta_k * (U_k.T @ b_full))
        ga, gb = 2.0 * t0a, 2.0 * t0b - mu
        nu0, nu1 = 2.0 / s1, -s2 / s1  # nu(rho) = 2*theta(rho)

        # next event: primal exit or dual entry, whichever root comes first
        rho_next, event = rho_max, None
        neg = b_seg < -tol
        if neg.any():
            roots = -a_seg[neg] / b_seg[neg]
            valid = roots > rho + 1e-12
            if valid.any():
                j = int(np.argmin(np.where(valid, roots, np.inf)))
                if roots[j] < rho_next:
                    rho_next = float(roots[j])
                    event = ("drop", int(idx[np.where(neg)[0][j]]))
        excl = np.where(~active)[0]
        h0 = ga[excl] - nu0
        h1 = gb[excl] - nu1
        falling = h1 < -tol
        if falling.any():
            roots = -h0[falling] / h1[falling]
            valid = roots > rho + 1e-12
            if valid.any():
                j = int(np.argmin(np.where(valid, roots, np.inf)))
                if roots[j] < rho_next:
                    rho_next = float(roots[j])
                    event = ("add", int(excl[np.where(falling)[0][j]]))

        segments.append((rho, rho_next, idx, a_seg, b_seg))
        if event is None or rho_next >= rho_max:
            break
        if event[0] == "drop":
            active[event[1]] = False
        else:
            active[event[1]] = True
        rho = rho_next

    return segments, len(segments) - 1


def cla_eval(segments, rho, n):
    """Evaluate the piecewise-affine CLA solution at a given rho."""
    for rho_lo, rho_hi, idx, a_seg, b_seg in segments:
        if rho_lo - 1e-12 <= rho <= rho_hi + 1e-12:
            w = np.zeros(n)
            w[idx] = a_seg + rho * b_seg
            return np.clip(w, 0.0, None)
    msg = f"rho={rho} outside traced range"
    raise ValueError(msg)


# Trace the exact frontier on the same synthetic problem as Section B
w0_minvar, _ = MinVarProblem(R_ef, alpha=1.0, target=tgt_rmt_ef, target_lr=lr_rmt_ef).solve_kkt()

t0 = _time.perf_counter()
cla_segments, cla_events = cla_sweep_woodbury(lr_rmt_ef, mu_ef, rho_max=2.0, w0=w0_minvar)
t_cla = _time.perf_counter() - t0

# Validate against the 50-point grid sweep (Woodbury warm, Section B problems).
# Where the two differ, compare objective values: the grid solver stops at its
# KKT tolerance, while CLA satisfies the KKT conditions exactly per segment.
max_dw_cla = 0.0
cla_never_worse = True
warm_chk = None
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
    w_grid, _, warm_chk = p.solve_kkt_warm(warm_start=warm_chk)
    w_cla = cla_eval(cla_segments, float(rho), n_ef)
    max_dw_cla = max(max_dw_cla, float(np.abs(w_grid - w_cla).max()))
    obj_cla = float(w_cla @ tgt_rmt_ef @ w_cla - rho * (mu_ef @ w_cla))
    obj_grid = float(w_grid @ tgt_rmt_ef @ w_grid - rho * (mu_ef @ w_grid))
    if obj_cla > obj_grid + 1e-14:
        cla_never_worse = False

print(f"  exact frontier on rho in [0, 2]:  {cla_events} breakpoints, {len(cla_segments)} segments")
print(f"  CLA sweep time: {t_cla * 1000:.1f} ms  (one Woodbury solve per segment)")
print(f"  50-point warm grid sweep:  {wb_w * 1000:.1f} ms  (Table, Section B)")
print(f"  max |w_CLA - w_grid| over the 50 grid points: {max_dw_cla:.2e}")
print(f"  CLA objective <= grid objective at every grid point: {cla_never_worse}")


# ===========================================================================
# Summary
# ===========================================================================

print()
print("=" * 70)
print("Summary of generated outputs")
print("=" * 70)
print("  Tables:")
print("    tables/rmt_preprocessing.tex      (Section A)")
print("    tables/rmt_solver_comparison.tex  (Section B)")
print("    tables/rmt_k_sensitivity.tex      (Section C)")
print("    tables/rmt_oos.tex                (Section E)")
print("  Figures:")
print("    graphs/rmt_frontier.pdf           (Section B)")
print("    graphs/rmt_scaling_full.pdf       (Section D)")
