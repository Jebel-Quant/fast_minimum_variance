# [fast-minimum-variance](https://jebel-quant.github.io/fast_minimum_variance): Solving Minimum Variance Portfolios Fast

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/fast-minimum-variance/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Jebel-Quant/fast_minimum_variance/blob/main/LICENSE)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Ffast_minimum_variance%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)

## Overview

**fast-minimum-variance** is a Python library for computing long-only minimum variance
portfolios without ever forming the sample covariance matrix. By operating directly on
the returns matrix $R \in \mathbb{R}^{T \times N}$, it exposes a clean hierarchy of
solvers — from an exact direct KKT solve to matrix-free Krylov methods — that scale
gracefully as $N$ grows.

The core insight is that minimising portfolio variance is equivalent to minimising
$\|Rw\|^2$, which can be evaluated using two matrix-vector products $w \mapsto R^\top(Rw)$
without constructing $R^\top R$ explicitly. This reframing connects the portfolio
optimisation literature directly to Krylov subspace methods.

The long-only constraint $w \geq 0$ is handled throughout via an **active-set method**:
solve the unconstrained problem on the current active set, drop assets with negative
weights, and repeat. The process terminates in at most $N$ iterations.

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
from fast_minimum_variance.random import make_returns
from fast_minimum_variance.kkt import solve_kkt
from fast_minimum_variance.krylov import solve_cg, solve_minres
from fast_minimum_variance.cvx import solve_cvxpy

# Generate a synthetic return matrix: 500 daily returns, 20 assets
R = make_returns(T=500, N=20, seed=42)

# Solve with any of the available solvers
w_kkt = solve_kkt(R)  # exact KKT solve
w_minres = solve_minres(R)  # MINRES on the indefinite KKT system
w_cg = solve_cg(R)  # CG in the constraint-reduced space
w_cvxpy = solve_cvxpy(R)  # CVXPY reference

# All solutions satisfy the portfolio constraints
assert abs(w_kkt.sum() - 1.0) < 1e-8
assert (w_kkt >= 0).all()
```

## The KKT System

The equality-constrained minimum variance problem yields the $(N+1) \times (N+1)$ KKT system:

$$\begin{pmatrix} 2R^\top R & \mathbf{1} \cr \mathbf{1}^\top & 0 \end{pmatrix} \begin{pmatrix} w \cr \lambda \end{pmatrix} = \begin{pmatrix} \mathbf{0} \cr 1 \end{pmatrix}$$

This system is **symmetric but indefinite** — the zero in the bottom-right corner of the
KKT matrix introduces a negative eigenvalue. This rules out standard CG on the full system,
but it opens the door to MINRES. Alternatively, the CG solver eliminates the constraint
entirely by parameterising $w = w_0 + Pv$ where $P$ spans the null space of
$\mathbf{1}^\top$, yielding a positive-definite reduced system of size $(N-1) \times (N-1)$.

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
- cvxpy

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
