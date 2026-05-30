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

    import numpy as np
    import pandas as pd
    from _common import run_timed

    from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


@app.cell
def _():

    # ── Table 2: S&P 500 ──────────────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("S&P 500  n=495, T=1192  (long-only minimum variance)")
    print("=" * 70)
    file = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
    df = pd.read_parquet(file)
    R_sp = df.to_numpy()
    T_sp, N_sp = R_sp.shape
    target_sp = np.var(R_sp) * np.eye(N_sp)  # mean squared entry = bar_lambda
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")

    alphas = np.linspace(0.05, 0.95, 10)
    results = {}

    for alpha in alphas:
        configs = [("cg", lambda a=alpha: MinVarProblem(R_sp, alpha=a, target=target_sp).solve_precon_cg())]
        row = {}
        for key, fn in configs:
            (_w, iters), t = run_timed(fn)
            row[key] = {"time_s": t, "iters": iters}
        results[alpha] = row

    # ── output ──────────────────────────────────────────────────────────────
    print(f"\n{'Alpha':>8} │ {'CG Iters':>10} │ {'Time (s)':>10}")
    print("─" * 35)
    for alpha, row in results.items():
        iters = row["cg"]["iters"]
        t = row["cg"]["time_s"]
        print(f"{alpha:>8.4f} │ {iters:>10d} │ {t:>10.4f}")

    # configs_sp_lw = [
    #     ("cg", lambda: MinVarProblem(R_sp, alpha=alpha_sp).solve_precon_cg())
    # ]

    # sp_lw = {}
    # for key, fn in configs_sp_lw:
    #     (_w, iters), t = run_timed(fn)
    #     sp_lw[key] = {"time_s": t, "iters": iters}

    # print_table(f"With LW shrinkage (alpha={alpha_sp:.4f})", sp_lw, "cg")
    return


@app.cell
def _():
    import importlib.metadata

    import marimo as mo

    package_name = "fast_minimum_variance"  # <-- Replace with your package name

    try:
        dist = importlib.metadata.distribution(package_name)
        direct_url = dist.read_text("direct_url.json")

        if direct_url and '"editable": true' in direct_url:
            status = mo.md(f"🔄 **{package_name}** is in **EDITABLE** mode. Changes take effect immediately.")
        else:
            status = mo.md(f"📦 **{package_name}** is in **STATIC** mode. You must reinstall to see changes.")
    except importlib.metadata.PackageNotFoundError:
        status = mo.md(f"❌ **{package_name}** is not installed in this environment.")

    status
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
