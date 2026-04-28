# Solving the Minimum Variance Portfolio Fast: A Krylov Perspective

**Thomas Schmelzer**

---

## Abstract

We revisit the long-only minimum variance portfolio problem and reframe it as a linear algebra problem amenable to Krylov subspace methods. Rather than forming the sample covariance matrix explicitly, we operate directly on the returns matrix using matrix-vector products. This yields a family of solvers — direct KKT, MINRES, SYMMLQ, and constraint-eliminated CG — that expose a clean hierarchy of accuracy, speed, and scalability. We draw an explicit connection between Ledoit-Wolf shrinkage and preconditioning, unifying two literatures that rarely speak to each other.

---

## 1. Introduction

The minimum variance portfolio is the canonical starting point for systematic equity allocation. Given a matrix of asset returns, the goal is to find the portfolio with the smallest possible variance — a well-posed convex optimisation problem with a classical solution via Markowitz (1952).

In practice, the problem is almost always solved by forming the sample covariance matrix and passing it to a general-purpose QP solver. This works, but obscures the structure of the problem. The covariance matrix is never observed directly — it is estimated from a returns matrix — and the solver has no knowledge of this origin.

We argue that working directly with the returns matrix is both numerically sounder and algorithmically richer. The resulting formulations connect naturally to Krylov subspace methods and to shrinkage estimation.

---

## 2. Problem Formulation

Let $X \in \mathbb{R}^{T \times n}$ be the matrix of daily returns for $n$ assets over $T$ trading days. The long-only minimum variance problem is:

$$\min_{w} \; w^\top \Sigma w \quad \text{subject to} \quad \mathbf{1}^\top w = 1, \quad w \geq 0$$

where $\Sigma = \frac{1}{T} X^\top X$ is the sample covariance matrix. Since $\Sigma \propto X^\top X$, this is equivalent to:

$$\min_{w} \; \|Xw\|^2 \quad \text{subject to} \quad \mathbf{1}^\top w = 1, \quad w \geq 0$$

This reframing is key: the objective is now a squared norm of $Xw$, and we can evaluate $w \mapsto X^\top(Xw)$ using two matrix-vector products, without ever forming $X^\top X$.

---

## 3. The KKT System

Ignoring the inequality constraints for now, the Lagrangian for the equality-constrained problem is:

$$\mathcal{L}(w, \lambda) = \|Xw\|^2 - \lambda(\mathbf{1}^\top w - 1)$$

Setting derivatives to zero gives the stationarity condition $2X^\top X w = \lambda \mathbf{1}$. Combined with the constraint $\mathbf{1}^\top w = 1$, this yields the augmented KKT system:

$$\begin{pmatrix} 2X^\top X & \mathbf{1} \\ \mathbf{1}^\top & 0 \end{pmatrix} \begin{pmatrix} w \\ \lambda \end{pmatrix} = \begin{pmatrix} \mathbf{0} \\ 1 \end{pmatrix}$$

This $(n+1) \times (n+1)$ system is **symmetric but indefinite** — the zero in the bottom-right corner ensures the matrix has both positive and negative eigenvalues. This rules out standard CG, but it opens the door to MINRES and SYMMLQ.

The long-only constraint $w \geq 0$ is handled via an **active set method**: solve the KKT system on the current active set, drop any asset with a negative weight, and repeat. The process terminates in at most $n$ iterations, each requiring one solve.

---

## 4. Krylov Solvers for the Indefinite System

### 4.1 Direct Solve (Baseline)

For small $n$, `numpy.linalg.solve` applied to the explicit KKT matrix is fast and exact. It requires forming $X^\top X$ — an $O(n^2 T)$ operation — but for $n \ll T$ this is not the bottleneck.

### 4.2 MINRES

MINRES (Paige & Saunders, 1975) is a Krylov method for symmetric indefinite systems. It minimises the residual $\|Ax - b\|$ over the Krylov subspace $\mathcal{K}_k(A, b)$. Applied to our KKT system:

- No need to form $X^\top X$ explicitly if we define the matrix-vector product $v \mapsto Av$ via the returns matrix
- Convergence depends on the eigenvalue distribution of the KKT matrix
- In practice, for well-conditioned problems MINRES converges in very few iterations

### 4.3 SYMMLQ

SYMMLQ (Paige & Saunders, 1975) is the companion to MINRES. While MINRES minimises $\|r_k\|$, SYMMLQ minimises $\|e_k\|_A$ — the error in the $A$-norm. For indefinite systems, SYMMLQ tends to produce a smoother convergence path and often yields a more accurate final solution at the same iteration count.

Both methods apply cleanly to our setting. The KKT system is symmetric by construction, and the block structure means the indefiniteness is mild.

### 4.4 Constraint-Eliminated CG

A cleaner approach avoids the indefinite system entirely. We parameterise the solution as:

$$w = w_0 + Pv$$

where $w_0 = \frac{1}{n}\mathbf{1}$ is any particular solution satisfying $\mathbf{1}^\top w_0 = 1$, and $P \in \mathbb{R}^{n \times (n-1)}$ is an orthonormal basis for the null space of $\mathbf{1}^\top$, obtained via QR factorisation of $\mathbf{1}$.

Any choice of $v$ gives a feasible $w$. The reduced unconstrained problem in $v$ is:

$$\min_v \; \|Xw_0 + XPv\|^2$$

with normal equations:

$$(XP)^\top (XP) \, v = -(XP)^\top (Xw_0)$$

The matrix $(XP)^\top(XP)$ is **symmetric positive definite**, so standard CG applies. The key point: the matrix-vector product $v \mapsto P^\top X^\top (X(Pv))$ requires four cheap operations and never forms $X^\top X$ or $(XP)^\top(XP)$ explicitly. For large sparse problems, this is the dominant advantage.

---

## 5. Numerical Results

| Method | $\|Xw\|$ | Time (s) |
|---|---|---|
| cvxpy | 0.1793 | 0.005 |
| KKT direct | 0.1793 | 0.000052 |
| MINRES | 0.1793 | 0.000155 |
| SYMMLQ | 0.1793 | ~0.000155 |
| CG (constraint-eliminated) | 0.1793 | 0.002 |

*Results on 8 assets, 862 trading days (Jan 2023 – Apr 2026).*

All four methods reach the same optimum. The KKT direct solve is fastest at this scale. For large $n$, the constraint-eliminated CG — which never forms $X^\top X$ — becomes the method of choice.

---

## 6. Preconditioning and Ledoit-Wolf Shrinkage

MINRES and SYMMLQ converge in a number of iterations proportional to the spread of the eigenvalues of the system matrix. When assets are highly correlated, $X^\top X$ is ill-conditioned — a handful of large eigenvalues and many small ones — and convergence degrades.

The natural remedy is preconditioning: replace $Ax = b$ with $M^{-1}Ax = M^{-1}b$ where $M \approx A$ and $M^{-1}$ is cheap to apply. A diagonal preconditioner:

$$M = \operatorname{diag}(2 \cdot \operatorname{diag}(X^\top X), \; 1)$$

corrects for scale differences between assets. A better choice is Ledoit-Wolf shrinkage:

$$\Sigma_{\text{LW}} = (1-\alpha) \cdot \frac{X^\top X}{T} + \alpha I$$

Adding $\alpha I$ bounds the smallest eigenvalue away from zero: $\lambda_{\min}(\Sigma_{\text{LW}}) \geq \alpha$. The condition number satisfies:

$$\kappa(\Sigma_{\text{LW}}) \leq \frac{\lambda_{\max} + \alpha}{\alpha}$$

which is directly controlled by the shrinkage intensity $\alpha$. Ledoit-Wolf provides an analytically optimal $\alpha$ that minimises the expected Frobenius distance to the true covariance — bypassing the need to cross-validate.

The punchline: **shrinkage and preconditioning are the same operation seen from different disciplines**. The statistician shrinks to reduce estimation error; the numerical analyst shrinks to compress the eigenvalue spectrum and accelerate convergence. Both arrive at the same modified matrix.

---

## 7. Conclusion

The minimum variance portfolio, when formulated directly in terms of the returns matrix, becomes a structured linear algebra problem with a rich hierarchy of solvers:

1. **KKT direct** — exact, fast for small $n$, forms $X^\top X$ explicitly
2. **MINRES / SYMMLQ** — Krylov methods for the indefinite system, no explicit $X^\top X$ at scale
3. **Constraint-eliminated CG** — positive definite reduced system, matrix-free, optimal for large sparse problems

Ledoit-Wolf shrinkage enters naturally as a preconditioner, unifying the statistical motivation (reduce estimation error) with the numerical motivation (improve conditioning). The optimal shrinkage intensity of Ledoit-Wolf is not just a regularisation choice — it is an analytically calibrated preconditioner.

The deeper message is one of perspective: many ideas in quantitative finance are linear algebra in disguise, and the tools developed by Paige, Saunders, Trefethen, and Ledoit-Wolf are more connected than the literature typically acknowledges.

---

## References

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Paige, C. C., & Saunders, M. A. (1975). Solution of sparse indefinite systems of linear equations. *SIAM Journal on Numerical Analysis*, 12(4), 617–629.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
