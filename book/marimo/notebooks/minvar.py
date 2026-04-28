# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.3",
#     "numpy>=2.0.0",
#     "scipy>=1.0",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()

with app.setup:
    import numpy as np

    from fast_minimum_variance.cvx import minvar_cvxpy
    from fast_minimum_variance.kkt import build_kkt, minvar_kkt
    from fast_minimum_variance.random import make_returns

    R = make_returns(T=500, N=20)


@app.cell
def _():
    from scipy.sparse.linalg import LinearOperator, cg, minres

    def minvar_minres():
        n = R.shape[1]
        active = np.ones(n, dtype=bool)
        while True:
            A, b = build_kkt(R[:, active])  # noqa: N806
            sol, _ = minres(A, b)
            w_a = sol[: active.sum()]
            if np.all(w_a >= -1e-10):
                break
            active[np.where(active)[0][w_a < 0]] = False
        w = np.zeros(n)
        w[active] = np.maximum(w_a, 0)
        return w

    def minvar_cg():
        n = R.shape[1]
        active = np.ones(n, dtype=bool)
        while True:
            R_a = R[:, active]  # noqa: N806
            n_a = R_a.shape[1]
            P = np.linalg.qr(np.ones((n_a, 1)), mode="complete")[0][:, 1:]  # noqa: N806
            w0 = np.ones(n_a) / n_a
            r0 = R_a @ w0

            def _matvec(v, P=P, R_a=R_a):
                return P.T @ (R_a.T @ (R_a @ (P @ v)))

            op = LinearOperator(shape=(n_a - 1, n_a - 1), matvec=_matvec)
            rhs = -(P.T @ (R_a.T @ r0))
            v, _ = cg(op, rhs)
            w_a = w0 + P @ v
            if np.all(w_a >= -1e-10):
                break
            active[np.where(active)[0][w_a < 0]] = False
        w = np.zeros(n)
        w[active] = np.maximum(w_a, 0)
        w /= w.sum()
        return w

    return (
        minvar_cg,
        minvar_minres,
    )


@app.cell
def _():
    w_cvxpy = minvar_cvxpy(R)
    print(w_cvxpy.round(4))
    return


@app.cell
def _():
    w_kkt = minvar_kkt(R)
    print(w_kkt.round(4))
    return


@app.cell
def _(minvar_cg, minvar_minres):
    import time

    results = {}
    for name, fn in [
        ("cvxpy", lambda: minvar_cvxpy(R)),
        ("kkt", lambda: minvar_kkt(R)),
        ("minres", lambda: minvar_minres()),
        ("cg", lambda: minvar_cg()),
    ]:
        t0 = time.perf_counter()
        w = fn()
        elapsed = time.perf_counter() - t0
        results[name] = {"norm": np.linalg.norm(R @ w), "time_s": elapsed}

    header = f"{'method':<15} {'norm':>10} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    for name, v in results.items():
        print(f"{name:<15} {v['norm']:>10.6f} {v['time_s']:>10.6f}")
    return


if __name__ == "__main__":
    app.run()
