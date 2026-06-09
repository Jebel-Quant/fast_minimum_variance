"""Reproduce all numerical results and figures for the paper.

    "Shrinkage as Preconditioning: Matrix-Free Methods for
     Long-Only Portfolio Optimization"

Usage:
    uv run experiment.py          # from the paper/ directory

Input:
    ../book/marimo/notebooks/data/sp500_pct_returns.parquet
        Daily percentage returns for S&P 500 constituents.
        Fetch with:  uv run fetch_sp500.py   (or: make load_data)

Outputs (stdout):
    Table 1 — S&P 500 benchmark: four panels (no shrinkage, LW alpha=0.5,
               LW oracle alpha≈0.017, RMT eigenvalue cleaning alpha=1)
               across seven solvers; timings, iterations, and speedup vs CVXPY.
    Scaling table — runtime vs n for KKT / CG / proximal (T=1250 fixed).
    Iterations table — CG iterations vs alpha (n=500, T=250, rank-deficient).
    Frontier table  — warm- vs cold-start sweep timings (n=500, T=1250).

Outputs (files):
    graphs/minvar_scaling.pdf   — Figure 1: runtime vs n (log-log)
    graphs/minvar_iters.pdf     — Figure 2: CG iterations vs alpha
    graphs/minvar_frontier.pdf  — Figure 3: efficient frontier coloured by active assets

Hardware used in the paper: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17, CVXPY 1.8.2, Clarabel 0.11.1.
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

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.runner import print_table, run_timed

from fast_minimum_variance.data import simulate_equity_returns
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.shrinkage.util import (
    lw_alpha_and_target,
    lw_alpha_and_target_hard,
    oas_alpha_and_target,
    rmt_target_and_alpha,
)

HERE = Path(__file__).parent

data = {"sp500": HERE / "data/sp500_pct_returns.parquet", "ftse": HERE / "data/ftse100_pct_returns.parquet"}

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

for name, file in data.items():
    DATA = file
    GRAPHS = HERE / "graphs" / name
    GRAPHS.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------------
    # S&P 500 benchmark  (Table 1)
    # ---------------------------------------------------------------------------

    print("=" * 70)
    print("S&P 500  n=495, T=1192  (long-only minimum variance)")
    print("=" * 70)

    df = pd.read_parquet(DATA)
    R_sp = df.to_numpy()
    R_sp = R_sp - R_sp.mean(axis=0)  # demean column-wise
    _T_sp, N_sp = R_sp.shape

    # Oracle alphas via sklearn (same scaled-identity target bar_lambda*I)
    alpha_lw_sp, target_sp = lw_alpha_and_target(R_sp)
    alpha_oas_sp, _ = oas_alpha_and_target(R_sp)
    alpha_sp = 0.5

    # RMT-clipped target and its oracle alpha
    target_rmt_sp, lr_rmt_sp, k_rmt_sp, alpha_rmt_sp = rmt_target_and_alpha(R_sp)

    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"n={N_sp}, T={_T_sp}, n/T={N_sp / _T_sp:.3f}")
    print(f"LW  oracle alpha = {alpha_lw_sp:.4f}")
    print(f"OAS oracle alpha = {alpha_oas_sp:.4f}")
    print(f"RMT oracle alpha = {alpha_rmt_sp:.4f}  (k={k_rmt_sp} signal factors)")
    print(f"Demonstrational alpha = {alpha_sp}")

    # Each solver entry is (name, callable) where the callable returns (outer, inner, time).
    # We normalise at the call site so print_table always sees {"outer": int|None, "inner": int|None}.

    def _run(prob, method, repeats=3):
        """Return {"time_s", "outer", "inner"} for one solver on one problem."""

        def _call():
            return method(prob)

        raw, t = run_timed(_call, repeats=repeats)
        if len(raw) == 3:  # solve_cg: (w, outer, inner)
            _, outer, inner = raw
        elif (
            method.__name__ == "<lambda>" and "solve_kkt" in method.__code__.co_consts[0]
            if hasattr(method, "__code__")
            else False
        ):
            _, outer = raw
            inner = None
        else:
            _, iters = raw
            outer = None
            inner = iters
        return {"time_s": t, "outer": outer, "inner": inner}

    # Solver specs: (display_name, solve_fn, is_kkt)
    # is_kkt=True  -> (w, outer_steps); display as outer=N, inner="--"
    # is_kkt=False and 3-tuple -> CG; display as outer=N, inner=M
    # is_kkt=False and 2-tuple -> no active-set loop; display as outer="--", inner=N
    def _make_entry(prob, fn, is_kkt=False):
        raw, t = run_timed(lambda: fn(prob))
        if len(raw) == 3:  # solve_cg -> (w, outer, inner)
            _, outer, inner = raw
        elif is_kkt:  # solve_kkt -> (w, outer_steps)
            _, outer = raw
            inner = None
        else:  # proximal / cvxpy / etc -> (w, iters)
            _, inner = raw
            outer = None
        return {"time_s": t, "outer": outer, "inner": inner}

    SOLVERS_ALL = [
        ("cvxpy (Clarabel)", lambda p: p.solve_cvxpy(), False),
        ("cvxpy (OSQP)", lambda p: p.solve_cvxpy(backend="osqp"), False),
        ("Clarabel (direct API)", lambda p: p.solve_clarabel(), False),
        ("OSQP (direct API)", lambda p: p.solve_osqp(), False),
        ("CG (SPD)", lambda p: p.solve_cg(), False),
        ("Proximal gradient", lambda p: p.solve_proximal(), False),
        ("FISTA (Nesterov)", lambda p: p.solve_fista(), False),
    ]
    SOLVERS_KEY = [
        ("cvxpy (Clarabel)", lambda p: p.solve_cvxpy(), False),
        ("CG (SPD)", lambda p: p.solve_cg(), False),
        ("Proximal gradient", lambda p: p.solve_proximal(), False),
        ("FISTA (Nesterov)", lambda p: p.solve_fista(), False),
    ]

    sp_no_lw, sp_lw_oracle, sp_oas_oracle, sp_lw, sp_rmt, sp_pcg = {}, {}, {}, {}, {}, {}
    prob_no_lw = MinVarProblem(R_sp)
    prob_lw_ora = MinVarProblem(R_sp, alpha=alpha_lw_sp, target=target_sp)
    prob_oas_ora = MinVarProblem(R_sp, alpha=alpha_oas_sp, target=target_sp)
    prob_lw = MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp)
    prob_rmt = MinVarProblem(R_sp, alpha=alpha_rmt_sp, target=target_rmt_sp, target_lr=lr_rmt_sp)
    # PCG: oracle-LW system (alpha≈0.017) + RMT target as preconditioner (§5.3)
    prob_pcg = MinVarProblem(R_sp, alpha=alpha_lw_sp, target=target_sp, pcg_lr=lr_rmt_sp)

    for name, fn, is_kkt in SOLVERS_ALL:
        sp_no_lw[name] = _make_entry(prob_no_lw, fn, is_kkt)
        sp_lw[name] = _make_entry(prob_lw, fn, is_kkt)

    for name, fn, is_kkt in SOLVERS_KEY:
        sp_lw_oracle[name] = _make_entry(prob_lw_ora, fn, is_kkt)
        sp_oas_oracle[name] = _make_entry(prob_oas_ora, fn, is_kkt)
        sp_rmt[name] = _make_entry(prob_rmt, fn, is_kkt)

    sp_pcg["cvxpy (Clarabel)"] = _make_entry(prob_pcg, lambda p: p.solve_cvxpy(), False)
    sp_pcg["CG (SPD)"] = _make_entry(prob_pcg, lambda p: p.solve_cg(), False)
    sp_pcg["PCG (RMT precond)"] = _make_entry(prob_pcg, lambda p: p.solve_pcg(), False)

    print_table("Without shrinkage", sp_no_lw, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle LW (alpha={alpha_lw_sp:.4f})", sp_lw_oracle, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle OAS (alpha={alpha_oas_sp:.4f})", sp_oas_oracle, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle RMT (alpha={alpha_rmt_sp:.4f}, k={k_rmt_sp})", sp_rmt, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle LW + RMT precond (alpha={alpha_lw_sp:.4f}, k={k_rmt_sp})", sp_pcg, ref_key="cvxpy (Clarabel)")
    print_table(f"Demonstrational LW (alpha={alpha_sp})", sp_lw, ref_key="cvxpy (Clarabel)")

    # ---------------------------------------------------------------------------
    # Panel A: runtime vs n  (LW shrinkage)
    # ---------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Runtime vs n  (T=1250 fixed, LW shrinkage, long-only minimum variance)")
    print("=" * 70)

    # T is fixed at 1250 (five years of daily returns), n grows.
    # With T fixed, KKT assembly cost O(n^2 T) grows as n^2 for n << T and is
    # dominated by Cholesky O(n^3) for n >> T, while CG cost O(n T sqrt(kappa))
    # grows more slowly.  This regime shows the matrix-free advantage clearly.
    T_FIXED = 1250
    ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    times = {k: [] for k in ("kkt", "cg", "proximal", "fista", "rmt_solve")}

    print(
        f"{'n':>6}  {'k_active':>8}  {'kkt(s)':>10}  {'kkt_out':>8}"
        f"  {'cg(s)':>10}  {'cg_out':>7}  {'cg_in':>7}"
        f"  {'prox(s)':>10}  {'prox_in':>8}"
        f"  {'fista(s)':>10}  {'fista_in':>9}"
        f"  {'rmt(s)':>10}  {'rmt_in':>8}  {'k_rmt':>6}"
    )
    print("-" * 143)

    for n in ns:
        R = simulate_equity_returns(n, T_FIXED, rng=n)
        alpha, tgt = lw_alpha_and_target_hard(R, alpha=0.5)
        prob = MinVarProblem(R, alpha=alpha, target=tgt)

        (w_kkt, kkt_outer), t_kkt = run_timed(lambda p=prob: p.solve_kkt())
        k_active = int((w_kkt > 1e-6).sum())
        (_, cg_outer, cg_inner), t_cg = run_timed(lambda p=prob: p.solve_cg())
        (_, prox_inner), t_prox = run_timed(lambda p=prob: p.solve_proximal())
        (_, fista_inner), t_fista = run_timed(lambda p=prob: p.solve_fista())

        # RMT: preprocessing outside timing; measure solve only
        tgt_rmt_s, lr_rmt_s, k_rmt_s, alpha_rmt_s = rmt_target_and_alpha(R)
        prob_rmt_s = MinVarProblem(R, alpha=alpha_rmt_s, target=tgt_rmt_s, target_lr=lr_rmt_s)
        (_, rmt_outer_s, rmt_inner_s), t_rmt = run_timed(lambda p=prob_rmt_s: p.solve_cg())

        times["kkt"].append(t_kkt)
        times["cg"].append(t_cg)
        times["proximal"].append(t_prox)
        times["fista"].append(t_fista)
        times["rmt_solve"].append(t_rmt)
        print(
            f"{n:>6}  {k_active:>8}  {t_kkt:>10.4f}  {kkt_outer:>8}"
            f"  {t_cg:>10.4f}  {cg_outer:>7}  {cg_inner:>7}"
            f"  {t_prox:>10.4f}  {prox_inner:>8}"
            f"  {t_fista:>10.4f}  {fista_inner:>9}"
            f"  {t_rmt:>10.4f}  {rmt_inner_s:>8}  {k_rmt_s:>6}"
        )

    # ---------------------------------------------------------------------------
    # Panel B: CG iterations vs alpha  (n=500, T=250, rank-deficient)
    # ---------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CG iterations vs alpha  (n=500, T=250, rank-deficient)")
    print("=" * 70)

    n_iter, T_iter = 500, 250
    R_iter = simulate_equity_returns(n_iter, T_iter, rng=1)
    _, tgt_iter = lw_alpha_and_target(R_iter)  # target only; alpha swept below
    alphas = np.linspace(0.01, 0.99, 40)
    cg_iters = []

    print(f"{'alpha':>8}  {'outer':>7}  {'inner':>8}")
    for a in alphas:
        _, outer, inner = MinVarProblem(R_iter, alpha=a, target=tgt_iter).solve_cg()
        cg_iters.append(inner)
        print(f"{a:>8.3f}  {outer:>7}  {inner:>8}")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------

    COLORS = {"kkt": "#1f77b4", "cg": "#ff7f0e", "proximal": "#9467bd", "rmt": "#2ca02c"}
    LABELS = {
        "kkt": "KKT direct",
        "cg": "CG (LW, $\\alpha=0.5$)",
        "proximal": "Proximal gradient",
        "rmt": "CG-RMT (solve)",
    }

    # Figure 1: runtime vs n  (minvar_scaling)
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.2))

    for key in ("cg", "proximal"):
        ax1.plot(ns, times[key], marker="o", markersize=4, label=LABELS[key], color=COLORS[key])
    ax1.plot(ns, times["rmt_solve"], marker="s", markersize=4, label=LABELS["rmt"], color=COLORS["rmt"], linestyle="-")
    n_arr = np.array(ns, dtype=float)
    anchor_idx = ns.index(500)
    t_anchor = times["cg"][anchor_idx]
    ax1.plot(n_arr, t_anchor * (n_arr / 500.0), color="gray", linestyle="--", linewidth=0.9, label=r"$O(n)$")
    ax1.plot(n_arr, t_anchor * (n_arr / 500.0) ** 2, color="gray", linestyle=":", linewidth=0.9, label=r"$O(n^2)$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of assets $n$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title(r"Runtime vs. $n$  ($T=1250$ fixed)")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig1.tight_layout(pad=1.0)
    fig1.savefig(GRAPHS / "minvar_scaling.pdf", bbox_inches="tight")
    fig1.savefig(GRAPHS / "minvar_scaling.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'minvar_scaling.pdf'}")

    # Figure 2: CG iterations vs alpha  (minvar_iters)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
    ax2.plot(alphas, cg_iters, marker="o", markersize=4, color=COLORS["cg"], label=LABELS["cg"])
    ax2.set_xlabel(r"Shrinkage intensity $\alpha$  ($\kappa$ decreases $\rightarrow$)")
    ax2.set_ylabel("CG iterations to convergence")
    ax2.set_title(r"CG iterations vs. $\alpha$  ($n=500,\,T=250$)")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig2.tight_layout(pad=1.0)
    fig2.savefig(GRAPHS / "minvar_iters.pdf", bbox_inches="tight")
    fig2.savefig(GRAPHS / "minvar_iters.png", bbox_inches="tight", dpi=150)
    print(f"Saved {GRAPHS / 'minvar_iters.pdf'}")

    # ---------------------------------------------------------------------------
    # Section 9: Efficient frontier  (n=500, T=1250)
    # ---------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Efficient frontier  (n=500, T=1250, multi-solver)")
    print("=" * 70)

    n_ef, T_ef = 500, 1250
    R_ef = simulate_equity_returns(n_ef, T_ef, rng=42)
    rng_ef = np.random.default_rng(42)
    betas_ef = rng_ef.uniform(0.4, 0.8, n_ef)
    mu_ef = betas_ef * (0.10 / 250)  # 10 % annual market premium -> daily units

    _, tgt_ef = lw_alpha_and_target(R_ef)
    alpha_ef = 0.5
    Sigma_ef = (1 - alpha_ef) * (R_ef.T @ R_ef) / T_ef + alpha_ef * tgt_ef
    print(f"Frontier alpha (LW) = {alpha_ef}")
    target_rmt_ef, lr_rmt_ef, k_rmt_ef, alpha_rmt_ef = rmt_target_and_alpha(R_ef)
    Sigma_rmt_ef = target_rmt_ef  # alpha_rmt_ef often == 1
    print(f"Frontier RMT: alpha={alpha_rmt_ef:.4f}, k={k_rmt_ef} signal factors")

    rhos_ef = np.linspace(0, 2, 50)

    import time as _time

    def _sweep_cold(solve_fn, repeats=3, ef_alpha=None, ef_target=None, ef_target_lr=None):
        """Return best-of-repeats list of per-point times."""
        _alpha = alpha_ef if ef_alpha is None else ef_alpha  # noqa: B023
        _target = tgt_ef if ef_target is None else ef_target  # noqa: B023
        _tgt_lr = None if ef_target_lr is None else ef_target_lr
        runs = []
        for _ in range(repeats):
            times = []
            for rho in rhos_ef:  # noqa: B023
                prob = MinVarProblem(R_ef, alpha=_alpha, target=_target, target_lr=_tgt_lr, rho=rho, mu=mu_ef)  # noqa: B023
                t0 = _time.perf_counter()
                solve_fn(prob)
                times.append(_time.perf_counter() - t0)
            runs.append(times)
        best = runs[int(np.argmin([sum(r) for r in runs]))]
        return best

    # --- Cold sweeps ---
    print("Running cold sweeps...")
    ef_times_cvxpy = _sweep_cold(lambda p: p.solve_cvxpy(), repeats=1)
    ef_times_osqp = _sweep_cold(lambda p: p.solve_osqp(), repeats=3)
    ef_times_proximal = _sweep_cold(lambda p: p.solve_proximal(), repeats=3)
    ef_times_cg_cold = _sweep_cold(lambda p: p.solve_cg(), repeats=3)
    ef_times_rmt_cold = _sweep_cold(
        lambda p: p.solve_cg(), repeats=3, ef_alpha=alpha_rmt_ef, ef_target=target_rmt_ef, ef_target_lr=lr_rmt_ef
    )

    # --- Warm-start sweep (CG and CG-RMT — other solvers have no warm-start API) ---
    print("Running CG warm-start sweep...")
    ef_warm_runs = []
    ef_vols, ef_rets, ef_active = [], [], []
    for rep in range(3):
        times = []
        warm = None
        vols_r, rets_r, act_r = [], [], []
        for rho in rhos_ef:
            prob = MinVarProblem(R_ef, alpha=alpha_ef, target=tgt_ef, rho=rho, mu=mu_ef)
            t0 = _time.perf_counter()
            w, _, _, warm = prob.solve_cg_warm(warm_start=warm)
            times.append(_time.perf_counter() - t0)
            vols_r.append(float(np.sqrt(w @ Sigma_ef @ w)) * np.sqrt(250) * 100)
            rets_r.append(float(w @ mu_ef) * 250 * 100)
            act_r.append(int((w > 1e-6).sum()))
        ef_warm_runs.append(times)
        if rep == 0:
            ef_vols, ef_rets, ef_active = vols_r, rets_r, act_r
    ef_times_cg_warm = ef_warm_runs[int(np.argmin([sum(r) for r in ef_warm_runs]))]

    print("Running RMT warm-start sweep...")
    ef_rmt_warm_runs = []
    ef_vols_rmt, ef_rets_rmt, ef_active_rmt = [], [], []
    for rep in range(3):
        times = []
        warm = None
        vols_r, rets_r, act_r = [], [], []
        for rho in rhos_ef:
            prob = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=target_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
            t0 = _time.perf_counter()
            w, _, _, warm = prob.solve_cg_warm(warm_start=warm)
            times.append(_time.perf_counter() - t0)
            vols_r.append(float(np.sqrt(w @ Sigma_rmt_ef @ w)) * np.sqrt(250) * 100)
            rets_r.append(float(w @ mu_ef) * 250 * 100)
            act_r.append(int((w > 1e-6).sum()))
        ef_rmt_warm_runs.append(times)
        if rep == 0:
            ef_vols_rmt, ef_rets_rmt, ef_active_rmt = vols_r, rets_r, act_r
    ef_times_rmt_warm = ef_rmt_warm_runs[int(np.argmin([sum(r) for r in ef_rmt_warm_runs]))]

    # --- Summary table ---
    N_PTS = len(rhos_ef)
    ref = sum(ef_times_cvxpy)

    def _row(label, times, warm_times=None):
        total = sum(times)
        per_ms = total / N_PTS * 1000  # noqa: B023
        spd = ref / total  # noqa: B023
        if warm_times is not None:
            w_total = sum(warm_times)
            w_per = w_total / N_PTS * 1000  # noqa: B023
            w_spd = ref / w_total  # noqa: B023
            print(
                f"  {label:<28}  cold: {total:6.3f}s ({per_ms:5.1f}ms/pt, {spd:5.0f}x)  "
                f"warm: {w_total:6.3f}s ({w_per:5.1f}ms/pt, {w_spd:5.0f}x)"
            )
        else:
            print(f"  {label:<28}  cold: {total:6.3f}s ({per_ms:5.1f}ms/pt, {spd:5.0f}x)  warm: --")

    print(f"\n{'Solver':<30}  {'Cold total':>12}  {'Warm total':>12}")
    print("-" * 70)
    _row("cvxpy (Clarabel)", ef_times_cvxpy)
    _row("OSQP (direct API)", ef_times_osqp)
    _row("Proximal gradient", ef_times_proximal)
    _row("CG (alpha=0.5, LW)", ef_times_cg_cold, ef_times_cg_warm)
    _row(f"CG (RMT, k={k_rmt_ef})", ef_times_rmt_cold, ef_times_rmt_warm)

    total_ef_cold = sum(ef_times_cg_cold)
    total_ef_warm = sum(ef_times_cg_warm)
    total_rmt_cold = sum(ef_times_rmt_cold)
    total_rmt_warm = sum(ef_times_rmt_warm)
    print(f"\nCG (LW)  warm vs cold speedup: {total_ef_cold / total_ef_warm:.1f}x")
    print(f"CG (LW)  warm vs CVXPY speedup: {ref / total_ef_warm:.0f}x")
    print(f"CG (RMT) warm vs CVXPY speedup: {ref / total_rmt_warm:.0f}x")
    print(f"CG (RMT) cold vs CG (LW) cold:  {total_ef_cold / total_rmt_cold:.2f}x")

    print(f"\n{'rho':>6}  {'vol%ann':>9}  {'ret%ann':>9}  {'active':>7}  {'cg_cold(ms)':>12}  {'cg_warm(ms)':>12}")
    print("-" * 65)
    for rho, vol, ret, act, tc, tw in zip(
        rhos_ef, ef_vols, ef_rets, ef_active, ef_times_cg_cold, ef_times_cg_warm, strict=False
    ):
        print(f"{rho:>6.2f}  {vol:>9.3f}  {ret:>9.3f}  {act:>7}  {tc * 1000:>12.1f}  {tw * 1000:>12.1f}")

    # Figure 3: efficient frontier — LW (coloured by active assets) + RMT overlay
    fig3, ax3 = plt.subplots(figsize=(4.5, 3.2))
    sc = ax3.scatter(ef_vols, ef_rets, c=ef_active, cmap="plasma_r", s=20, zorder=3, label=r"LW ($\alpha=0.5$)")
    ax3.plot(ef_vols, ef_rets, color="gray", linewidth=0.8, zorder=2)
    ax3.plot(
        ef_vols_rmt,
        ef_rets_rmt,
        color="steelblue",
        linewidth=1.2,
        linestyle="--",
        zorder=3,
        label=f"RMT ($k={k_rmt_ef}$ factors)",
    )
    ax3.scatter([ef_vols[0]], [ef_rets[0]], marker="*", s=120, color="#ff7f0e", zorder=4, label="Min-var (LW)")
    ax3.scatter(
        [ef_vols_rmt[0]], [ef_rets_rmt[0]], marker="*", s=120, color="steelblue", zorder=4, label="Min-var (RMT)"
    )
    cbar = fig3.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("Active assets (LW)")
    ax3.set_xlabel("Annualised volatility (\\%)")
    ax3.set_ylabel("Annualised expected return (\\%)")
    ax3.set_title(f"Efficient frontier  ($n={n_ef}$, $T={T_ef}$)")
    ax3.legend(framealpha=0.9, loc="lower right", fontsize=7)
    ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig3.tight_layout(pad=1.0)
    fig3.savefig(GRAPHS / "minvar_frontier.pdf", bbox_inches="tight")
    fig3.savefig(GRAPHS / "minvar_frontier.png", bbox_inches="tight", dpi=150)
    print(f"Saved {GRAPHS / 'minvar_frontier.pdf'}")
