"""Generate scaling figures and table benchmarks for minvar_paper.tex."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "fast-minimum-variance",
#     "marimo"
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from _common import print_table, run_timed, set_notebook_plot_style

    from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


@app.cell
def _():
    set_notebook_plot_style(mpl)

    # ── Table 2: S&P 500 ──────────────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("S&P 500  n=495, T=1192  (long-only minimum variance)")
    print("=" * 70)
    file = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
    df = pd.read_parquet(file)
    R_sp = df.to_numpy()
    T_sp, N_sp = R_sp.shape
    alpha_sp = 0.5
    target_sp = np.var(R_sp) * np.eye(N_sp)  # mean squared entry = bar_lambda
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"alpha = {alpha_sp:.4f}")

    configs_sp_no_lw = [
        ("cvxpy", lambda: MinVarProblem(R_sp).solve_cvxpy()),
        ("clarabel", lambda: MinVarProblem(R_sp).solve_clarabel()),
        ("osqp", lambda: MinVarProblem(R_sp).solve_osqp()),
        ("kkt", lambda: MinVarProblem(R_sp).solve_kkt()),
        ("cg", lambda: MinVarProblem(R_sp).solve_cg()),
        ("nnls", lambda: MinVarProblem(R_sp).solve_nnls()),
    ]
    configs_sp_lw = [
        ("cvxpy", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_cvxpy()),
        ("clarabel", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_clarabel()),
        ("osqp", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_osqp()),
        ("kkt", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_kkt()),
        ("cg", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_cg()),
        ("nnls", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_nnls()),
    ]

    sp_no_lw, sp_lw = {}, {}
    for key, fn in configs_sp_no_lw:
        (w, iters), t = run_timed(fn)
        sp_no_lw[key] = {"time_s": t, "iters": iters}
    for key, fn in configs_sp_lw:
        (_w, iters), t = run_timed(fn)
        sp_lw[key] = {"time_s": t, "iters": iters}

    print_table("Without LW shrinkage (alpha=0)", sp_no_lw)
    print_table(f"With LW shrinkage (alpha={alpha_sp:.4f})", sp_lw)

    # ── Panel A: runtime vs n ──────────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("Runtime vs n  (T=2n, LW shrinkage, long-only minimum variance)")
    print("=" * 70)

    ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    times = {k: [] for k in ("kkt", "cg", "nnls")}
    rng2 = np.random.default_rng(0)
    print(f"{'n':>6}  {'kkt':>10}  {'cg':>10}  {'nnls':>10}")
    print("-" * 44)
    for n in ns:
        T = 2 * n
        R = rng2.standard_normal((T, n))
        alpha = n / (n + T)
        bar_lam = float(np.linalg.norm(R, "fro") ** 2) / (n * T)
        tgt = bar_lam * np.eye(n)
        _, t_kkt = run_timed(lambda r=R, a=alpha, t=tgt: MinVarProblem(r, alpha=a, target=t).solve_kkt())
        _, t_cg = run_timed(lambda r=R, a=alpha, t=tgt: MinVarProblem(r, alpha=a, target=t).solve_cg())
        _, t_nnls = run_timed(lambda r=R, a=alpha, t=tgt: MinVarProblem(r, alpha=a, target=t).solve_nnls())
        times["kkt"].append(t_kkt)
        times["cg"].append(t_cg)
        times["nnls"].append(t_nnls)
        print(f"{n:>6}  {t_kkt:>10.4f}  {t_cg:>10.4f}  {t_nnls:>10.4f}")

    # ── Panel B: iterations vs shrinkage intensity alpha ──────────────────────────

    n_iter, T_iter = 500, 250
    R_iter = np.random.default_rng(1).standard_normal((T_iter, n_iter))
    bar_lambda_iter = float(np.linalg.norm(R_iter, "fro") ** 2) / (n_iter * T_iter)
    alphas = np.linspace(0.01, 0.99, 40)
    cg_iters_by_alpha = []
    print()
    print("=" * 70)
    print(f"Iterations vs alpha  (n={n_iter}, T={T_iter}, rank-deficient)")
    print("=" * 70)
    for a in alphas:
        tgt_iter = bar_lambda_iter * np.eye(n_iter)
        _, iters = MinVarProblem(R_iter, alpha=a, target=tgt_iter).solve_cg()
        cg_iters_by_alpha.append(iters)
        print(f"  alpha={a:.3f}  iters={iters}")

    # ── Plot ───────────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    colors = {"kkt": "#1f77b4", "cg": "#ff7f0e", "nnls": "#2ca02c"}
    labels = {"kkt": "KKT direct", "cg": "CG (matrix-free)", "nnls": "NNLS"}
    for key in ("kkt", "cg", "nnls"):
        ax1.plot(ns, times[key], marker="o", markersize=3, label=labels[key], color=colors[key])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of assets $n$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title(r"(a) Runtime vs. $n$  ($T=2n$, with LW shrinkage)")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    ax2.plot(alphas, cg_iters_by_alpha, marker="o", markersize=3, color=colors["cg"], label=labels["cg"])
    ax2.set_xlabel(r"Shrinkage intensity $\alpha$  $(\kappa$ decreases $\rightarrow)$")
    ax2.set_ylabel("CG iterations to convergence")
    ax2.set_title(r"(b) Iterations vs. $\alpha$  ($n=500,\,T=250$)")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout(pad=1.0)

    folder = Path(__file__).parent / "graphs"
    fig.savefig(folder / "minvar_scaling.pdf", bbox_inches="tight")
    fig.savefig(folder / "minvar_scaling.png", bbox_inches="tight", dpi=150)
    print()
    print("Saved graphs/minvar_scaling.pdf and graphs/minvar_scaling.png")

    # ── Standalone log-log Runtime vs n ───────────────────────────────────────────

    fig2, ax = plt.subplots(figsize=(4.5, 3.2))

    for key in ("kkt", "cg", "nnls"):
        ax.plot(ns, times[key], marker="o", markersize=4, label=labels[key], color=colors[key])

    # Reference slope lines anchored to the KKT curve at n=500
    n_arr = np.array(ns, dtype=float)
    anchor_idx = ns.index(500)
    t_anchor = times["kkt"][anchor_idx]
    n_anchor = 500.0
    for exp, ls, lbl in [(2, "--", r"$O(n^2)$"), (3, ":", r"$O(n^3)$")]:
        ax.plot(
            n_arr,
            t_anchor * (n_arr / n_anchor) ** exp,
            color="gray",
            linestyle=ls,
            linewidth=0.9,
            label=lbl,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of assets $n$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(r"Runtime vs. $n$  ($T = 2n$, LW shrinkage)")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig2.tight_layout(pad=1.0)

    fig2.savefig(folder / "minvar_loglog.pdf", bbox_inches="tight")
    fig2.savefig(folder / "minvar_loglog.png", bbox_inches="tight", dpi=150)
    print("Saved graphs/minvar_loglog.pdf and graphs/minvar_loglog.png")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
