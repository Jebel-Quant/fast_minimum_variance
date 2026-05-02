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

All solvers are methods on both `Problem` and `MinVarProblem`:

| Method | Approach | Notes |
|---|---|---|
| `solve_kkt()` | Direct KKT via `numpy.linalg.solve` | Exact; baseline for accuracy comparisons |
| `solve_minres()` | MINRES on the indefinite KKT system | Matrix-free; handles indefiniteness correctly |
| `solve_cg()` | CG in the constraint-reduced space | Positive-definite reduced system; fastest for large $N$ |
| `solve_cvxpy()` | General-purpose convex solver via CVXPY | Reference implementation; requires `[convex]` extra |

All solvers return `(w, n_iters)` where $w \in \mathbb{R}^N$ satisfies $\sum_i w_i = 1$ and $w_i \geq 0$.

## Quick Start

```python
import numpy as np
from fast_minimum_variance import Problem

# Returns matrix: 500 daily returns, 20 assets
R = np.random.default_rng(42).standard_normal((500, 20))

# No custom constraints → fast shrinking active-set solver
p = Problem(R)
w_kkt,    _ = p.solve_kkt()    # exact KKT solve
w_minres, _ = p.solve_minres() # MINRES on the indefinite KKT system
w_cg,     _ = p.solve_cg()    # CG in the constraint-reduced space

assert abs(w_kkt.sum() - 1.0) < 1e-8
assert (w_kkt >= 0).all()

# Ledoit-Wolf shrinkage
T, N = R.shape
w, iters = Problem(R, alpha=N / (N + T)).solve_minres()

# Custom constraints → general growing active-set solver
import numpy as np
A = np.ones((N, 1))          # budget constraint only
b = np.ones(1)
C = -np.eye(N)               # long-only
d = np.zeros(N)
w, _ = Problem(R, A=A, b=b, C=C, d=d).solve_kkt()
```

## The `Problem` Factory

`Problem(X, ...)` is the single entry point. It dispatches automatically:

- **No `A`, `b`, `C`, `d`** → shrinking active-set (faster; KKT system shrinks from
  $(N+1)\times(N+1)$ to $(N^*+1)\times(N^*+1)$ where $N^*$ is the final portfolio size)
- **Any of `A`, `b`, `C`, `d` provided** → growing active-set (handles arbitrary linear
  equality and inequality constraints)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `ndarray (T, N)` | required | Returns matrix |
| `A` | `ndarray (N, m)` | — | Equality constraint matrix: $A^\top w = b$ |
| `b` | `ndarray (m,)` | — | Equality RHS |
| `C` | `ndarray (N, p)` | — | Inequality constraint matrix: $C^\top w \leq d$ |
| `d` | `ndarray (p,)` | — | Inequality RHS |
| `alpha` | `float` | `0.0` | Ledoit-Wolf shrinkage intensity; ridge = $\alpha \|X\|_F^2 / N$ |
| `rho` | `float` | `0.0` | Return tilt strength for mean-variance |
| `mu` | `ndarray (N,)` | `None` | Expected returns vector |

When `A`, `b`, `C`, `d` are omitted, the defaults are $A = \mathbf{1}$, $b = 1$,
$C = -I$, $d = 0$ (budget + long-only). We suggest to use `alpha = N / (N + T)` for shrinkage.

## `_MinVarProblem` vs `_Problem`

Under the hood, `Problem(...)` returns one of two solver classes. You never need to
instantiate them directly — the factory does the right thing — but understanding the
difference explains the performance characteristics.

### `_MinVarProblem` — shrinking active-set

Used when **no custom constraints are passed**. Designed exclusively for the long-only
minimum-variance problem ($\sum w_i = 1$, $w_i \geq 0$).

The active-set strategy works by **removing** assets: whenever an asset's optimal weight
is negative, it is dropped from the subproblem entirely. The KKT system shrinks from
$(N+1)\times(N+1)$ down to $(N^*+1)\times(N^*+1)$, where $N^* \ll N$ is the final
number of assets held. On real equity data — where a minimum-variance portfolio
concentrates in a small fraction of the universe — this can reduce MINRES iterations
by an order of magnitude (e.g. 214 vs 1 065 on S&P 500 without shrinkage).

### `_Problem` — growing active-set

Used when **any of `A`, `b`, `C`, `d` are provided**. Handles arbitrary linear equality
and inequality constraints.

The active-set strategy works by **adding** violated inequalities as equalities. The
solver always operates on the full $N$-dimensional system, and the KKT matrix grows
from $(N+m)\times(N+m)$ (equality constraints only) up to
$(N+m+p^*)\times(N+m+p^*)$ as $p^*$ inequality constraints are activated. This is the
right choice whenever you need custom turnover limits, sector caps, or any constraint
structure that goes beyond the default long-only budget problem.

### The good news: you don't have to choose

```python
# Calls _MinVarProblem internally — fast shrinking active-set
w, _ = Problem(R).solve_minres()

# Calls _Problem internally — general growing active-set
w, _ = Problem(R, A=A, b=b, C=C, d=d).solve_kkt()
```

The `Problem(...)` factory inspects whether any constraints were supplied and routes
to the appropriate class automatically. Both expose the same four solver methods
(`solve_kkt`, `solve_minres`, `solve_cg`, `solve_cvxpy`) and return `(w, n_iters)`.

## The KKT System

The equality-constrained minimum variance problem yields the $(N+m) \times (N+m)$ KKT system:

$$\begin{pmatrix} 2\!\left(R^\top R + \tfrac{\alpha\|R\|_F^2}{N} I\right) & A \cr A^\top & 0 \end{pmatrix} \begin{pmatrix} w \cr \lambda \end{pmatrix} = \begin{pmatrix} \rho\mu \cr b \end{pmatrix}$$

where $A \in \mathbb{R}^{N \times m}$ collects the active equality and inequality constraints.
With the defaults ($A = \mathbf{1}$, $b = 1$, $\alpha = 0$, $\rho = 0$) this reduces to
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
