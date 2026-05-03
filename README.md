# [fast-minimum-variance](https://jebel-quant.github.io/fast_minimum_variance): Solving Minimum Variance Portfolios Fast

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/fast-minimum-variance/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Jebel-Quant/fast_minimum_variance/blob/main/LICENSE)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Ffast_minimum_variance%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)

## Overview

**fast-minimum-variance** solves the long-only minimum variance portfolio without ever
forming the sample covariance matrix. The key observation is that the KKT stationarity
condition $2\Sigma w = \lambda\mathbf{1}$ immediately gives $w \propto \Sigma^{-1}\mathbf{1}$:
the entire problem reduces to one symmetric positive definite linear system $\Sigma v =
\mathbf{1}$, solved matrix-free by conjugate gradients. The budget constraint is recovered
by a single rescaling $w = v / (\mathbf{1}^\top v)$.

Working directly with the returns matrix $X \in \mathbb{R}^{T \times N}$ — rather than
the assembled covariance $X^\top X$ — has two consequences. First, each conjugate gradient
iteration costs $O(TN)$ rather than $O(N^2)$, and $X^\top X$ is never stored. Second,
Ledoit-Wolf shrinkage enters as a simple row-augmentation of $X$: stacking
$[\sqrt{1-\alpha}\,X;\,\sqrt{\gamma}\,I]$ yields a matrix whose Gram matrix equals
$\Sigma_{\text{LW}}$. The same CG code handles both the plain and shrunk problem without
modification.

## Quick Start

```python
import numpy as np
from fast_minimum_variance import Problem

# 500 daily returns, 20 assets
X = np.random.default_rng(42).standard_normal((500, 20))

w, iters = Problem(X).solve_cg()   # matrix-free CG — recommended
w, iters = Problem(X).solve_kkt()  # direct dense solve — exact baseline

assert abs(w.sum() - 1.0) < 1e-8
assert (w >= 0).all()
```

## Ledoit-Wolf Shrinkage

Ledoit-Wolf shrinkage plays a dual role: statistically it reduces estimation error; numerically
it compresses the eigenvalue spectrum and directly cuts CG iteration counts. Use
`alpha = N / (N + T)` as a simple analytical estimate of the optimal shrinkage intensity:

```python
T, N = X.shape
w, iters = Problem(X, alpha=N / (N + T)).solve_cg()
```

On S&P 500 equity data (495 assets, 1192 days), shrinkage cuts CG iterations from 685 to
205 and makes the matrix-free solver the fastest option by a wide margin.

## Solvers

All solvers are methods on `Problem` and return `(w, iters)` where
$w \in \mathbb{R}^N$, $\sum_i w_i = 1$, $w_i \geq 0$.

| Method | Approach | When to use |
|---|---|---|
| `solve_cg()` | Matrix-free conjugate gradients on the SPD reduced system | Default — fastest for large $N$, especially with shrinkage |
| `solve_kkt()` | Direct dense factorisation via `numpy.linalg.solve` | Small problems or when an exact solve is needed |
| `solve_nnls()` | Non-negative least squares via Lawson-Hanson | Single-shot; useful when no outer loop is desired |
| `solve_clarabel()` | Clarabel interior-point solver (direct API) | Comparison baseline without CVXPY overhead |
| `solve_cvxpy()` | CVXPY + Clarabel | Ground-truth reference; requires `[convex]` extra |

### `solve_cg` — matrix-free conjugate gradients

The inner step builds a `LinearOperator` that applies

$$v \;\mapsto\; (1-\alpha)\,X_a^\top(X_a v) + \gamma v, \qquad \gamma = \frac{\alpha\|X\|_F^2}{N}$$

to a vector using two matrix-vector products with the active-asset submatrix $X_a$, without
ever forming $\Sigma_a = X_a^\top X_a$. Standard CG then solves $\Sigma_a v = \mathbf{1}$.
Ledoit-Wolf shrinkage ($\alpha > 0$) compresses the eigenvalue spectrum and reduces
iteration counts dramatically — from nearly 2000 iterations at $\alpha \approx 0$ to
single digits at $\alpha \approx 1$ in rank-deficient settings.

### `solve_kkt` — direct dense solve

Assembles $\Sigma_a = (1-\alpha)X_a^\top X_a + \gamma I$ explicitly and calls
`numpy.linalg.solve`. Exact to machine precision. Scales as $O(N^3)$ in the active
portfolio size, so it becomes expensive for $N \gtrsim 500$ without shrinkage (which
reduces the number of active assets). With shrinkage, the active-set outer loop converges
in 2–4 steps and the inner systems are small, making the direct solve competitive.

### `solve_nnls` — non-negative least squares

Reformulates the problem as a non-negative least squares problem on an augmented matrix:

$$\min_{w \geq 0}\;\left\|\begin{pmatrix}\sqrt{1-\alpha}\,X \\ \sqrt{\gamma}\,I \\ M\mathbf{1}^\top\end{pmatrix}w - \begin{pmatrix}\mathbf{0} \\ \mathbf{0} \\ M\end{pmatrix}\right\|^2$$

where $M = \|X\|_F \cdot T$ enforces the budget constraint as a large penalty. The
Lawson-Hanson algorithm handles $w \geq 0$ natively, so no outer primal-dual loop is
needed. Single-shot but does not benefit from the matrix-free structure: Lawson-Hanson
implicitly forms normal equations of the augmented matrix. With shrinkage the augmented
matrix grows from $T \times N$ to $(T+N) \times N$, making `solve_nnls` slower with
shrinkage than without.

### `solve_clarabel` — Clarabel direct API

Calls the Clarabel interior-point solver directly, bypassing CVXPY's problem-construction
overhead. Assembles $P = 2\Sigma_{\text{LW}}$ as a sparse CSC matrix and solves the
standard QP. Useful for benchmarking: on a 1000-asset synthetic problem, Clarabel direct
takes 0.28 s while the CVXPY wrapper takes 8.2 s — over 97% of `solve_cvxpy`'s time is
Python interface overhead, not solving. CG is still 15× faster than Clarabel direct.

## The Primal-Dual Active-Set Loop

Long-only weights are enforced by an outer loop that wraps any inner solver:

1. **Primal step.** Solve the budget-only equality system over the current active asset
   set. Drop any asset with weight below $-\varepsilon$ (multiple assets at once if
   violations are large).
2. **Dual step.** Once all active weights are non-negative, compute the gradient
   $\nabla_i f(w) = 2[(1-\alpha)(X^\top X w)_i + \gamma w_i] - \rho\mu_i$ for every
   excluded asset. If any excluded asset has $\nabla_i f(w) < \lambda$ (the budget
   multiplier), it would decrease variance if added — re-insert the most-violated asset
   and repeat.
3. **Termination.** The loop exits when primal and dual feasibility hold simultaneously.
   Combined with stationarity from the inner solve, this is sufficient for global optimality.

With Ledoit-Wolf shrinkage at the analytically optimal $\alpha$, the loop typically
converges in 2–4 outer iterations on real equity data.

## Problem Variants

The same solver handles a range of portfolio construction problems by choosing $\alpha$, $\rho$, $\mu$:

| Problem | `alpha` | `rho` | `mu` |
|---|---|---|---|
| Minimum variance | $0$ | $0$ | — |
| Mean-variance (Markowitz) | any | $> 0$ | expected returns |
| Minimum tracking error to benchmark $b$ | any | $2$ | `X.T @ (X @ b)` |
| LW-regularised minimum variance | $N/(N+T)$ | $0$ | — |

```python
# Mean-variance
mu = np.array([...])  # expected returns, shape (N,)
w, _ = Problem(X, rho=1.0, mu=mu).solve_cg()

# Minimum tracking error to benchmark b
b = np.ones(N) / N  # equal-weight benchmark
mu_te = X.T @ (X @ b)
w, _ = Problem(X, rho=2.0, mu=mu_te).solve_cg()
```

When `rho != 0`, two SPD solves are performed per outer step: $\Sigma_a v_1 = \mathbf{1}$
and $\Sigma_a v_2 = \mu_a$. The budget multiplier $\lambda$ is recovered analytically
from the budget constraint, avoiding the full saddle-point system.

## Custom Constraints

For problems beyond budget + long-only (sector limits, turnover bounds, factor-exposure
constraints), pass explicit constraint matrices:

```python
A = np.ones((N, 1))   # budget: 1'w = 1
b = np.ones(1)
C = -np.eye(N)        # long-only: w >= 0
d = np.zeros(N)
w, _ = Problem(X, A=A, b=b, C=C, d=d).solve_kkt()
```

This routes to a general active-set solver that handles arbitrary linear equality and
inequality constraints. Use this path sparingly — the default path (no `A`, `b`, `C`, `d`)
is significantly faster for the standard long-only problem.

## Benchmarks

All timings on Apple M4 Pro, Python 3.12, NumPy 2.4, SciPy 1.17.

### Synthetic: $N=1000$, $T=2000$, i.i.d. Gaussian returns

| Method | Time (s) | Speedup vs CVXPY |
|---|---|---|
| `solve_cvxpy` | 8.16 | 1× |
| `solve_clarabel` | 0.28 | 29× |
| `solve_kkt` | 0.063 | 129× |
| **`solve_cg`** | **0.019** | **430×** |
| `solve_nnls` | 1.69 | 5× |

*With Ledoit-Wolf shrinkage ($\alpha = 0.333$), 56 CG iterations.*

### S&P 500: $N=495$, $T=1192$ (Jul 2021–Apr 2026)

| Method | Time (s) | Speedup vs CVXPY |
|---|---|---|
| `solve_cvxpy` | 1.48 | 1× |
| `solve_clarabel` | 0.067 | 22× |
| `solve_kkt` | 0.018 | 84× |
| **`solve_cg`** | **0.0091** | **162×** |
| `solve_nnls` | 0.088 | 17× |

*With Ledoit-Wolf shrinkage ($\alpha = 0.293$), 205 CG iterations.*

## Installation

```bash
pip install fast-minimum-variance
```

To use the CVXPY and Clarabel reference solvers:

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
- cvxpy *(optional, only required for `solve_cvxpy` and `solve_clarabel`)*

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
