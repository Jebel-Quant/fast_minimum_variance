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
    from _common import print_table, run_timed

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
    alpha_sp = 0.5
    target_sp = np.var(R_sp) * np.eye(N_sp)  # mean squared entry = bar_lambda
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"alpha = {alpha_sp:.4f}")

    configs_sp_no_lw = [
        ("cg", lambda: MinVarProblem(R_sp).solve_precon_cg()),
    ]
    configs_sp_lw = [
        ("cg", lambda: MinVarProblem(R_sp, alpha=alpha_sp, target=target_sp).solve_precon_cg()),
    ]

    sp_no_lw, sp_lw = {}, {}
    for key, fn in configs_sp_no_lw:
        (w, iters), t = run_timed(fn)
        sp_no_lw[key] = {"time_s": t, "iters": iters}
    for key, fn in configs_sp_lw:
        (_w, iters), t = run_timed(fn)
        sp_lw[key] = {"time_s": t, "iters": iters}

    print_table(f"With LW shrinkage (alpha={alpha_sp:.4f})", sp_lw, "cg")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
