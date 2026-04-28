import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd

    pd.options.plotting.backend = "plotly"
    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("data/equity_prices.csv", parse_dates=["date"])
    prices = df.pivot(index="date", columns="ticker", values="close")
    returns = prices.pct_change().dropna()
    returns
    return (returns,)


@app.cell
def _(np):
    import cvxpy as cp
    from scipy.sparse.linalg import minres, symmlq, cg, LinearOperator

    def minvar_3(R):
        cov = np.cov(R.T)
        inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
        ones = np.ones(3)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    def random_forest_minvar(R, n_trees=2500, subset_size=3):
        n = R.shape[1]
        weights = np.zeros(n)
        rng = np.random.default_rng(42)
        for _ in range(n_trees):
            idx = rng.choice(n, size=subset_size, replace=False)
            w = minvar_3(R[:, idx])
            for i, wi in zip(idx, w):
                weights[i] += wi
        avg = weights / n_trees
        avg /= avg.sum()
        return avg

    def minvar_cvxpy(R):
        n = R.shape[1]
        w = cp.Variable(n)
        cp.Problem(cp.Minimize(cp.sum_squares(R @ w)), [cp.sum(w) == 1, w >= 0]).solve()
        return w.value

    def _build_kkt(R_a):
        n_a = R_a.shape[1]
        A = np.zeros((n_a + 1, n_a + 1))
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
            A, b = _build_kkt(R[:, active])
            sol = np.linalg.solve(A, b)
            w_a = sol[:active.sum()]
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
            A, b = _build_kkt(R[:, active])
            sol, _ = minres(A, b)
            w_a = sol[:active.sum()]
            if np.all(w_a >= -1e-10):
                break
            active[np.where(active)[0][w_a < 0]] = False
        w = np.zeros(n)
        w[active] = np.maximum(w_a, 0)
        return w

    def minvar_symmlq(R):
        n = R.shape[1]
        active = np.ones(n, dtype=bool)
        while True:
            A, b = _build_kkt(R[:, active])
            sol, _ = symmlq(A, b)
            w_a = sol[:active.sum()]
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
            R_a = R[:, active]
            n_a = R_a.shape[1]
            P = np.linalg.qr(np.ones((n_a, 1)), mode='complete')[0][:, 1:]
            w0 = np.ones(n_a) / n_a
            r0 = R_a @ w0
            op = LinearOperator(
                shape=(n_a - 1, n_a - 1),
                matvec=lambda v: P.T @ (R_a.T @ (R_a @ (P @ v)))
            )
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
        minvar_symmlq,
        random_forest_minvar,
    )


@app.cell
def _(random_forest_minvar, returns):
    R = returns.values
    w_rf = random_forest_minvar(R)
    print(w_rf.round(4))
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
def _(
    R,
    minvar_cg,
    minvar_cvxpy,
    minvar_kkt,
    minvar_minres,
    minvar_symmlq,
    np,
    pd,
    random_forest_minvar,
    returns,
):
    import time

    tickers = returns.columns
    results = {}
    for name, fn in [
        ("random_forest", lambda: random_forest_minvar(R)),
        ("cvxpy",         lambda: minvar_cvxpy(R)),
        ("kkt",           lambda: minvar_kkt(R)),
        ("minres",        lambda: minvar_minres(R)),
        ("symmlq",        lambda: minvar_symmlq(R)),
        ("cg",            lambda: minvar_cg(R)),
    ]:
        t0 = time.perf_counter()
        w = fn()
        elapsed = time.perf_counter() - t0
        results[name] = {"weights": w, "norm": np.linalg.norm(R @ w), "time_s": elapsed}

    summary = pd.DataFrame(
        {n: {"norm": v["norm"], "time_s": v["time_s"]} for n, v in results.items()}
    ).T
    print(summary.round(6))

    pd.DataFrame(
        {n: pd.Series(v["weights"], index=tickers) for n, v in results.items()}
    ).round(4)
    return


if __name__ == "__main__":
    app.run()
