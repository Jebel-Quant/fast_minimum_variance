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
    from fast_minimum_variance.cvx import minvar_cvxpy
    from fast_minimum_variance.kkt import minvar_kkt
    from fast_minimum_variance.krylov import minvar_cg, minvar_minres
    from fast_minimum_variance.random import make_returns

    R = make_returns(T=2000, N=1000)
    # R = make_correlated_returns(T=2000, N=1000)


@app.cell
def _():
    import time

    import numpy as np

    # Build the LW-shrunk effective return matrix:
    #   R_lw = vstack([sqrt(c)*R, sqrt(gamma)*I])
    # so that R_lw.T @ R_lw = c*R.T@R + gamma*I  (the LW covariance, scaled by T).
    # alpha = N/(N+T), gamma = ||R||_F^2/(N+T), c = 1-alpha = T/(N+T).
    T_dim, N_dim = R.shape  # noqa: N806
    frob_sq = np.einsum("ti,ti->", R, R)
    gamma_lw = frob_sq / (N_dim + T_dim)
    c_lw = T_dim / (N_dim + T_dim)
    R_lw = np.vstack([np.sqrt(c_lw) * R, np.sqrt(gamma_lw) * np.eye(N_dim)])  # noqa: N806

    results = {}
    for name, fn in [
        ("cvxpy", lambda: (minvar_cvxpy(R_lw), None)),
        ("kkt", lambda: (minvar_kkt(R_lw), None)),
        ("minres", lambda: minvar_minres(R, c=c_lw, gamma=gamma_lw)),
        ("cg", lambda: minvar_cg(R, c=c_lw, gamma=gamma_lw)),
    ]:
        t0 = time.perf_counter()
        w, iters = fn()
        elapsed = time.perf_counter() - t0
        results[name] = {"norm": np.linalg.norm(R @ w), "sum": np.sum(w), "time_s": elapsed, "iters": iters}

    cvxpy_time = results["cvxpy"]["time_s"]
    header = f"{'method':<15} {'norm':>10} {'sum':>10} {'time_s':>10} {'iters':>8} {'speedup':>10}"
    print(header)
    print("-" * len(header))
    for name, v in results.items():
        iters_str = str(v["iters"]) if v["iters"] is not None else "-"
        speedup = cvxpy_time / v["time_s"]
        print(f"{name:<15} {v['norm']:>10.6f} {v['sum']:>10.6f} {v['time_s']:>10.6f} {iters_str:>8} {speedup:>10.1f}x")
    return


if __name__ == "__main__":
    app.run()
