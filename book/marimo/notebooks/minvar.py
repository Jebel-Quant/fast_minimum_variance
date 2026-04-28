# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.3",
#     "fast-minimum-variance",
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


@app.cell
def _():
    import time

    import numpy as np

    results = {}
    for name, fn in [
        ("cvxpy", lambda: minvar_cvxpy(R)),
        ("kkt", lambda: minvar_kkt(R)),
        ("minres", lambda: minvar_minres(R)),
        ("cg", lambda: minvar_cg(R)),
    ]:
        t0 = time.perf_counter()
        w = fn()
        elapsed = time.perf_counter() - t0
        results[name] = {"norm": np.linalg.norm(R @ w), "sum": np.sum(w), "time_s": elapsed}

    header = f"{'method':<15} {'norm':>10} {'sum':>10} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    for name, v in results.items():
        print(f"{name:<15} {v['norm']:>10.6f} {v['sum']:>10.6f} {v['time_s']:>10.6f}")
    return


if __name__ == "__main__":
    app.run()
