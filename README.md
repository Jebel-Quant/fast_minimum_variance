# [fast-minimum-variance](https://jebel-quant.github.io/fast_minimum_variance): Solving Minimum Variance Portfolios Fast

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/fast-minimum-variance/)
[![Downloads](https://static.pepy.tech/personalized-badge/fast-minimum-variance?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/fast-minimum-variance)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Jebel-Quant/fast_minimum_variance/blob/main/LICENSE)
[![Coverage](https://jebel-quant.github.io/fast_minimum_variance/coverage-badge.svg)](https://jebel-quant.github.io/fast_minimum_variance/reports/html-coverage/)
[![CodeFactor](https://www.codefactor.io/repository/github/jebel-quant/fast_minimum_variance/badge)](https://www.codefactor.io/repository/github/jebel-quant/fast_minimum_variance)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Ffast_minimum_variance%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)

## Overview

**fast-minimum-variance** solves the global (equality-constrained) minimum variance
portfolio. The key observation is that the KKT stationarity condition
$2\Sigma w = \lambda\mathbf{1}$ immediately gives $w \propto \Sigma^{-1}\mathbf{1}$: the
entire problem reduces to one symmetric positive definite linear system
$\Sigma v = \mathbf{1}$, solved by conjugate gradients. The budget constraint is recovered
by a single rescaling $w = v / (\mathbf{1}^\top v)$. Weights are sign-unconstrained, so
short positions are allowed.

The sample covariance $\Sigma = X^\top X / T$ is formed as a plain dense NumPy array (with
Ledoit-Wolf shrinkage $\Sigma \mapsto (1-\alpha)\Sigma + \alpha T_0$ folded in when a
target is supplied), and CG runs directly on it. The implementation depends only on NumPy
and SciPy.

## Quick Start

```python
import numpy as np
from fast_minimum_variance import Problem

# 500 daily returns, 20 assets
X = np.random.default_rng(42).standard_normal((500, 20))

w, outer, inner = Problem(X).solve_cg()   # conjugate gradients on a dense Sigma

assert abs(w.sum() - 1.0) < 1e-8          # budget holds exactly; weights may be negative
```

## Ledoit-Wolf Shrinkage

Ledoit-Wolf shrinkage plays a dual role: statistically it reduces estimation error; numerically
it compresses the eigenvalue spectrum and directly cuts CG iteration counts. Use
`alpha = N / (N + T)` as a simple analytical estimate of the optimal shrinkage intensity:

```python
T, N = X.shape
w, outer, inner = Problem(X, alpha=N / (N + T)).solve_cg()
```

On S&P 500 equity data (495 assets, 1192 days), shrinkage cuts CG iterations from 685 to
205 — the entire solve runs in under 10 ms (see [Benchmarks](#benchmarks)).

## The Solver

`Problem.solve_cg()` runs conjugate gradients on the dense SPD system and
returns `(w, outer_steps, inner_iters)` where $w \in \mathbb{R}^N$ and $\sum_i w_i = 1$.
`outer_steps` is always `1` (there is no outer loop; the field is retained for API
compatibility).

The solve forms the dense system matrix

$$\Sigma = \frac{1-\alpha}{T}\,X^\top X + \alpha\,T_0$$

as a NumPy array and hands it to `scipy.sparse.linalg.cg`, which solves
$\Sigma v = \mathbf{1}$ (and $\Sigma v_\mu = \mu$ when a return tilt is set). Ledoit-Wolf
shrinkage ($\alpha > 0$) compresses the eigenvalue spectrum and reduces CG iteration
counts.

Because weights are sign-unconstrained, the KKT system is linear and a **single** CG
solve suffices — no active-set iteration. If you need a long-only ($w \ge 0$) portfolio,
project or re-optimise downstream; this library targets the fast global-minimum-variance
solve.

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
mu = np.random.default_rng(0).standard_normal(N)  # expected returns, shape (N,)
w, *_ = Problem(X, rho=1.0, mu=mu).solve_cg()

# Minimum tracking error to benchmark b
b = np.ones(N) / N  # equal-weight benchmark
mu_te = X.T @ (X @ b)
w, *_ = Problem(X, rho=2.0, mu=mu_te).solve_cg()
```

When `rho != 0`, two SPD solves are performed: $\Sigma v_1 = \mathbf{1}$ and
$\Sigma v_2 = \mu$. The budget multiplier $\lambda$ is recovered analytically from the
budget constraint, avoiding the full saddle-point system.

## Balance Systems

To replace the default budget constraint $\mathbf{1}^\top w = 1$ with a general set of
linear equality constraints $B w = c$ (e.g. sleeve budgets, factor-exposure targets),
pass a balance system `(B, c)`:

```python
B = np.zeros((2, N)); B[0, :N // 2] = 1.0; B[1, N // 2:] = 1.0  # each half holds...
c = np.array([0.5, 0.5])                                        # ...half of the budget
w, *_ = Problem(X, B=B, c=c).solve_cg()
```

`B` must have full row rank. Weights remain sign-unconstrained; the multiplier for the
`p` constraints is recovered from a small $p \times p$ Schur system.

## Benchmarks

All timings on Apple M4 Pro, Python 3.12, NumPy 2.4, SciPy 1.17.

| Universe | $N$ | $T$ | `solve_cg` time (s) |
|---|---|---|---|
| Synthetic i.i.d. Gaussian | 1000 | 2000 | 0.019 |
| S&P 500 (Jul 2021–Apr 2026) | 495 | 1192 | 0.0091 |

*Both with Ledoit-Wolf shrinkage ($\alpha = 0.333$ synthetic / $0.293$ S&P), 56 and 205
CG iterations respectively.*

## Installation

```bash
pip install fast-minimum-variance
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
