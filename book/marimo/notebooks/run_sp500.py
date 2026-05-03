"""Run solver benchmark on real S&P 500 data."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "marimo",
#     "fast-minimum-variance[convex]",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()

with app.setup:
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from fast_minimum_variance import Problem
    from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem


@app.cell
def _():
    # ── Load data ──────────────────────────────────────────────────────────────────
    file = Path(__file__).parent / "data" / "sp500_returns.parquet"
    df = pd.read_parquet(file)
    R = df.to_numpy()
    T, N = R.shape
    print(f"S&P 500 returns: T={T} trading days, N={N} assets")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}\n")

    # ── Ledoit-Wolf parameters ─────────────────────────────────────────────────────

    alpha_lw = N / (N + T)  # Ledoit-Wolf shrinkage intensity; ridge = alpha * ||R||_F^2/N

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
        ("cvxpy", lambda: Problem(R).solve_cvxpy()),
        ("clarabel", lambda: MinVarProblem(R).solve_clarabel()),
        ("kkt", lambda: MinVarProblem(R).solve_kkt()),
        ("cg", lambda: MinVarProblem(R).solve_cg()),
        ("nnls", lambda: MinVarProblem(R).solve_nnls()),
    ]

    configs_lw = [
        ("cvxpy", lambda: Problem(R, alpha=alpha_lw).solve_cvxpy()),
        ("clarabel", lambda: MinVarProblem(R, alpha=alpha_lw).solve_clarabel()),
        ("kkt", lambda: MinVarProblem(R, alpha=alpha_lw).solve_kkt()),
        ("cg", lambda: MinVarProblem(R, alpha=alpha_lw).solve_cg()),
        ("nnls", lambda: MinVarProblem(R, alpha=alpha_lw).solve_nnls()),
    ]

    display = {
        "cvxpy": "cvxpy (Clarabel)",
        "clarabel": "Clarabel (direct API)",
        "kkt": "KKT direct",
        "cg": "CG (SPD)",
        "nnls": "NNLS (Lawson-Hanson)",
    }

    row_order = ("cvxpy", "clarabel", "kkt", "cg", "nnls")

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
            print(f"  Running {display[key]}...")
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
        cols = f"{v['time_s']:.4f} & {iters_str:>6} & {ref / v['time_s']:.1f}x"
        return f"{dn:<35} & {cols} \\\\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Time (s) & Iterations & Speedup \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Without Ledoit-Wolf shrinkage}} \\[2pt]",
    ]
    panel_no = all_results["Without Ledoit-Wolf shrinkage"]
    ref_no = panel_no["cvxpy"]["time_s"]
    for key in row_order:
        lines.append(tex_row(display[key], panel_no[key], ref_no))

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{With Ledoit-Wolf shrinkage}} \\[2pt]")
    panel_lw = all_results["With Ledoit-Wolf shrinkage"]
    ref_lw = panel_lw["cvxpy"]["time_s"]
    for key in row_order:
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
    return


if __name__ == "__main__":
    app.run()
