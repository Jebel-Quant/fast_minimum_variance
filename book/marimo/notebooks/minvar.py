# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.3",
#     "numpy>=2.0.0",
#     "cvxpy>=1.0",
#     "scipy>=1.0",
# ]
# [tool.uv.sources]
# fast-minimum-variance = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    def make_returns(T, N, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, N))

    return (make_returns,)


@app.cell
def _(np):
    import cvxpy as cp
    from scipy.sparse.linalg import LinearOperator, cg, minres

    def minvar_cvxpy(R):
        n = R.shape[1]
        w = cp.Variable(n)
        cp.Problem(cp.Minimize(cp.sum_squares(R @ w)), [cp.sum(w) == 1, w >= 0]).solve()
        return w.value

    def _build_kkt(R_a):
        n_a = R_a.shape[1]
        A = np.zeros((n_a + 1, n_a + 1))  # noqa: N806
        A[:n_a, :n_a] = 2 * R_a.T @ R_a
        A[:n_a, n_a] = 1
        A[n_a, :n_a] = 1
        b = np.zeros(n_a + 1)
        b[n_a] = 1
        return A, b

    def minvar_kkt(R):
        n = R.shape[1]
        active = np.ones(n, dtype=bool)
        while True:
            A, b = _build_kkt(R[:, active])  # noqa: N806
            sol = np.linalg.solve(A, b)
            w_a = sol[: active.sum()]
            if np.all(w_a >= -1e-10):
                break
            active[np.where(active)[0][w_a < 0]] = False
        w = np.zeros(n)
        w[active] = np.maximum(w_a, 0)
        return w

    def minvar_minres(R):
        n = R.shape[1]
        active = np.ones(n, dtype=bool)
        while True:
            A, b = _build_kkt(R[:, active])  # noqa: N806
            sol, _ = minres(A, b)
            w_a = sol[: active.sum()]
            if np.all(w_a >= -1e-10):
                break
            active[np.where(active)[0][w_a < 0]] = False
        w = np.zeros(n)
        w[active] = np.maximum(w_a, 0)
        return w

    def minvar_cg(R):
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
        minvar_cvxpy,
        minvar_kkt,
        minvar_minres,
    )


@app.cell
def _(make_returns):
    R = make_returns(T=500, N=20)  # noqa: N806
    return (R,)


@app.cell
def _(R, minvar_cvxpy):
    w_cvxpy = minvar_cvxpy(R)
    print(w_cvxpy.round(4))
    return


@app.cell
def _(R, minvar_kkt):
    w_kkt = minvar_kkt(R)
    print(w_kkt.round(4))
    return


@app.cell
def _(R, minvar_cg, minvar_cvxpy, minvar_kkt, minvar_minres, np):
    import time

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
        results[name] = {"norm": np.linalg.norm(R @ w), "time_s": elapsed}

    header = f"{'method':<15} {'norm':>10} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    for name, v in results.items():
        print(f"{name:<15} {v['norm']:>10.6f} {v['time_s']:>10.6f}")
    return


if __name__ == "__main__":
    app.run()
