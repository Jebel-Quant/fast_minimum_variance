# [fast-minimum-variance](https://jebel-quant.github.io/fast_minimum_variance): Solving Minimum Variance Portfolios Fast

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/fast-minimum-variance/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Jebel-Quant/fast_minimum_variance/blob/main/LICENSE)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Ffast_minimum_variance%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)

## Overview

**fast-minimum-variance** is a Python library for computing minimum variance and
mean-variance portfolios without ever forming the sample covariance matrix. By operating
directly on the returns matrix $R \in \mathbb{R}^{T \times N}$, it exposes a clean
hierarchy of solvers — from an exact direct KKT solve to matrix-free Krylov methods —
that scale gracefully as $N$ grows.

The core insight is that minimising portfolio variance is equivalent to minimising
$\|Rw\|^2$, which can be evaluated using two matrix-vector products $w \mapsto R^\top(Rw)$
without constructing $R^\top R$ explicitly. This reframing connects the portfolio
optimisation literature directly to Krylov subspace methods.

Linear equality and inequality constraints ($A^\top w = b$, $C^\top w \leq d$) are
handled via an **active-set method**: violated inequalities are promoted to equalities
one outer iteration at a time, and the process terminates in at most $p$ iterations
where $p$ is the number of inequality constraints.

## Solvers

| Solver | Module | Method | Notes |
|---|---|---|---|
| `solve_kkt` | `kkt` | Direct KKT via `numpy.linalg.solve` | Exact; baseline for accuracy comparisons |
| `solve_minres` | `krylov` | MINRES on the indefinite KKT system | Matrix-free capable; handles indefiniteness correctly |
| `solve_cg` | `krylov` | CG in the constraint-reduced space | Positive-definite reduced system; no indefinite solver needed |
| `solve_cvxpy` | `cvx` | General-purpose convex solver via CVXPY | Reference implementation; slowest but most flexible |

All solvers return a weight vector $w \in \mathbb{R}^N$ satisfying $\sum_i w_i = 1$ and $w_i \geq 0$.

## Quick Start

```python
from fast_minimum_variance.api import API
from fast_minimum_variance.random import make_returns
from fast_minimum_variance.kkt import solve_kkt
from fast_minimum_variance.krylov import solve_cg, solve_minres

# Generate a synthetic return matrix: 500 daily returns, 20 assets
R = make_returns(T=500, N=20, seed=42)
api = API(X=R)

# Solve with any of the available solvers
w_kkt, _ = solve_kkt(api)      # exact KKT solve
w_minres, _ = solve_minres(api) # MINRES on the indefinite KKT system
w_cg, _ = solve_cg(api)        # CG in the constraint-reduced space

# All solutions satisfy the portfolio constraints
assert abs(w_kkt.sum() - 1.0) < 1e-8
assert (w_kkt >= 0).all()
```

## The API Dataclass

All solvers accept an `API` dataclass that bundles the problem data:

| Field | Type | Default | Description |
|---|---|---|---|
| `X` | `ndarray (T, N)` | required | Returns matrix |
| `A` | `ndarray (N, m)` | `ones((N,1))` | Equality constraint matrix: $A^\top w = b$ |
| `b` | `ndarray (m,)` | `[1.0]` | Equality RHS (budget constraint by default) |
| `C` | `ndarray (N, p)` | `-eye(N)` | Inequality constraint matrix: $C^\top w \leq d$ |
| `d` | `ndarray (p,)` | `zeros(N)` | Inequality RHS (long-only by default) |
| `rho` | `float` | `0.0` | Return tilt strength for mean-variance |
| `mu` | `ndarray (N,)` | `None` | Expected returns vector |
| `gamma` | `float` | `0.0` | L2 regularisation (e.g. Ledoit-Wolf shrinkage) |

The defaults recover the long-only minimum variance problem. Pass custom `A`, `b`, `C`, `d` for arbitrary linear equality and inequality constraints.

## The KKT System

The equality-constrained minimum variance problem yields the $(N+m) \times (N+m)$ KKT system:

$$\begin{pmatrix} 2(R^\top R + \gamma I) & A \cr A^\top & 0 \end{pmatrix} \begin{pmatrix} w \cr \lambda \end{pmatrix} = \begin{pmatrix} \rho\,\mu \cr b \end{pmatrix}$$

where $A \in \mathbb{R}^{N \times m}$ collects the active equality and inequality constraints.
With the defaults ($A = \mathbf{1}$, $b = 1$, $\gamma = 0$, $\rho = 0$) this reduces to
the familiar $(N+1) \times (N+1)$ budget-constraint system.

This system is **symmetric but indefinite** — the zero bottom-right block introduces
negative eigenvalues. This rules out standard CG on the full system, but it opens the
door to MINRES. Alternatively, the CG solver eliminates the constraints entirely by
parameterising $w = w_0 + Pv$ where $P$ spans the null space of $A^\top$, yielding a
positive-definite reduced system of size $(N-m) \times (N-m)$.

## Installation

```bash
pip install fast-minimum-variance
```

To use the CVXPY reference solver:

```bash
pip install fast-minimum-variance[convex]
```

For development:

```bash
git clone https://github.com/Jebel-Quant/fast_minimum_variance
cd fast_minimum_variance
make install
```

## Requirements

- Python 3.11+
- numpy
- scipy
- cvxpy *(optional, only required for `solve_cvxpy`)*

## Citing

If you use this library in academic work or research, please cite:

```bibtex
@software{fast_minimum_variance,
  author  = {Schmelzer, Thomas},
  title   = {fast-minimum-variance: Solving Minimum Variance Portfolios Fast},
  url     = {https://github.com/Jebel-Quant/fast_minimum_variance},
  year    = {2026},
  license = {MIT}
}
```

## License

MIT License — see [LICENSE](https://github.com/Jebel-Quant/fast_minimum_variance/blob/main/LICENSE) for details.
