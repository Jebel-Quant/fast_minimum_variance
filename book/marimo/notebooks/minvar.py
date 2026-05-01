# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.3",
#     "fast-minimum-variance[convex]",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()

with app.setup:
    from fast_minimum_variance.api import Problem
    from fast_minimum_variance.random import make_returns

    R = make_returns(T=2000, N=1000)
    # R = make_correlated_returns(T=2000, N=1000)


@app.cell
def _():
    import time

    import numpy as np

    T_dim, N_dim = R.shape  # noqa: N806
    frob_sq = np.einsum("ti,ti->", R, R)
    gamma_lw = frob_sq / (N_dim + T_dim)
    R_lw = np.vstack([R, np.sqrt(gamma_lw) * np.eye(N_dim)])  # noqa: N806

    def run_all(shrinkage):
        if shrinkage:
            configs = [
                ("cvxpy", lambda: Problem(X=R_lw).solve_cvxpy()),
                ("kkt", lambda: Problem(X=R_lw).solve_kkt()),
                ("minres", lambda: Problem(X=R, gamma=gamma_lw).solve_minres()),
                ("cg", lambda: Problem(X=R, gamma=gamma_lw).solve_cg()),
            ]
        else:
            configs = [
                ("cvxpy", lambda: Problem(X=R).solve_cvxpy()),
                ("kkt", lambda: Problem(X=R).solve_kkt()),
                ("minres", lambda: Problem(X=R).solve_minres()),
                ("cg", lambda: Problem(X=R).solve_cg()),
            ]
        out = {}
        for name, fn in configs:
            t0 = time.perf_counter()
            w, iters = fn()
            out[name] = {"norm": np.linalg.norm(R @ w), "time_s": time.perf_counter() - t0, "iters": iters}
        return out

    res_no_lw = run_all(shrinkage=False)
    res_lw = run_all(shrinkage=True)

    display_names = {"cvxpy": "cvxpy", "kkt": "KKT direct", "minres": "MINRES", "cg": "CG (constraint-eliminated)"}

    for label, results in [("Without LW shrinkage", res_no_lw), ("With LW shrinkage", res_lw)]:
        ref = results["cvxpy"]["time_s"]
        print(f"\n{label}")
        print(f"{'method':<30} {'norm':>10} {'time_s':>10} {'iters':>8} {'speedup':>10}")
        print("-" * 72)
        for name, v in results.items():
            iters_str = str(v["iters"]) if v["iters"] is not None else "-"
            spd = ref / v["time_s"]
            print(f"{display_names[name]:<30} {v['norm']:>10.6f} {v['time_s']:>10.6f} {iters_str:>8} {spd:>10.1f}x")

    # LaTeX table: two panels separated by \midrule
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $\|Xw\|$ & Time (s) & Iterations & Speedup \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Without Ledoit-Wolf shrinkage}} \\[2pt]",
    ]

    def tex_row(dn, v, ref):
        iters_str = str(v["iters"]) if v["iters"] is not None else "--"
        cols = f"{v['norm']:.4f} & {v['time_s']:.4f} & {iters_str:>6} & {ref / v['time_s']:.1f}x"
        return f"{dn:<30} & {cols} \\\\"

    ref_no = res_no_lw["cvxpy"]["time_s"]
    for name, v in res_no_lw.items():
        lines.append(tex_row(display_names[name], v, ref_no))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{With Ledoit-Wolf shrinkage}} \\[2pt]")
    ref_lw = res_lw["cvxpy"]["time_s"]
    for name, v in res_lw.items():
        lines.append(tex_row(display_names[name], v, ref_lw))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Results on {N_dim} assets, {T_dim} trading days. Speedup relative to cvxpy within each panel.}}",
        r"\end{table}",
    ]
    print("\n% LaTeX table")
    print("\n".join(lines))
    return


if __name__ == "__main__":
    app.run()
