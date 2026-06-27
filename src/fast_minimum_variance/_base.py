"""Common base for portfolio-optimisation problem classes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import clarabel
import cvxpy as cp
import numpy as np
import osqp
from cvx.linalg import cholesky
from scipy.sparse import csc_matrix, triu


@dataclass(frozen=True)
class _BaseProblem(ABC):
    """Shared fields, utilities, and solver templates for portfolio problems.

    Subclasses must implement the seven abstract hooks:

    * ``_constraint_active_set(solve_fn)`` — outer constraint-handling loop
    * ``_kkt_step(mask) -> (w, iters)`` — one direct-KKT inner step
    * ``_cg_step(mask) -> (w, iters)`` — one CG inner step
    * ``_cvxpy_constraints(w, cp) -> list`` — CVXPY constraint list
    * ``_clarabel_constraints() -> (A, b, cones)`` — Clarabel constraint data
    * ``_osqp_constraints() -> (A, l, u)`` — OSQP constraint data
    * ``_nnls_solve() -> (w, 1)`` — NNLS direct solve

    All ``solve_*`` methods are implemented here as template methods that
    call ``_constraint_active_set`` with the appropriate ``_XXX_step``
    method, then optionally clip-and-renormalize.
    """

    X: np.ndarray
    target: np.ndarray | None = None
    alpha: float = 0.0
    rho: float = 0.0
    mu: np.ndarray | None = None
    target_lr: tuple[float, np.ndarray, np.ndarray] | None = None  # (bar_lam, U_k, delta_k) — low-rank + identity
    pcg_lr: tuple[float, np.ndarray, np.ndarray] | None = None  # (bar_lam, U_k, delta_k) — RMT preconditioner (§5.3)

    def __post_init__(self) -> None:
        """Validate target/target_lr shapes when supplied."""
        n = self.n
        if self.target is not None and self.target.shape != (n, n):
            raise ValueError(f"target must be a square {n} x {n} matrix, got {self.target.shape}")  # noqa: TRY003
        if self.target_lr is not None:
            _bar_lam, U_k, delta_k = self.target_lr  # noqa: N806
            if U_k.shape[0] != n or U_k.shape[1] != delta_k.shape[0]:
                raise ValueError(  # noqa: TRY003
                    f"target_lr: U_k must be ({n}, k) and delta_k (k,), got {U_k.shape}, {delta_k.shape}"
                )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def t(self) -> int:
        """Return the number of rows in X."""
        return int(self.X.shape[0])

    @property
    def n(self) -> int:
        """Number of assets (columns of X)."""
        return int(self.X.shape[1])

    @staticmethod
    def _clip_and_renormalize(w: np.ndarray) -> np.ndarray:
        """Clip weights to ``[0, ∞)`` and renormalize to sum to 1."""
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    # ------------------------------------------------------------------
    # Abstract hooks (raise NotImplementedError — subclasses must override)
    # ------------------------------------------------------------------
    @abstractmethod
    def _constraint_active_set(
        self,
        solve_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> tuple[np.ndarray, int, int]:  # pragma: no cover
        """Run the outer constraint-handling loop, calling ``solve_fn`` each iteration."""
        raise NotImplementedError

    @abstractmethod
    def _kkt_step(self, active: np.ndarray) -> tuple[np.ndarray, int]:  # pragma: no cover
        """Solve one inner direct-KKT step; return ``(w, iters)``."""
        raise NotImplementedError

    @abstractmethod
    def _cvxpy_constraints(self, w: cp.Variable, cp: object) -> list[Any]:  # pragma: no cover
        """Return the list of CVXPY constraints for ``solve_cvxpy``."""
        raise NotImplementedError

    @abstractmethod
    def _cg_step(self, active: np.ndarray) -> tuple[np.ndarray, int]:
        """Solve one inner CG step; return ``(w, iters)``."""
        raise NotImplementedError  # pragma: no cover

    def _pcg_step(self, active: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, int]:  # pragma: no cover
        """Solve one inner PCG step with RMT preconditioner; return ``(w, iters)``.

        Subclasses that support PCG (e.g. ``_MinVarProblem``) override this.
        The base implementation raises so callers get a clear error if PCG is
        invoked on a problem type that has not implemented it.
        """
        raise NotImplementedError

    @abstractmethod
    def _nnls_solve(self) -> tuple[np.ndarray, int]:  # pragma: no cover
        """Solve via NNLS directly (no outer loop); return ``(w, 1)``."""
        raise NotImplementedError

    @abstractmethod
    def _clarabel_constraints(self) -> tuple[csc_matrix, np.ndarray, list[Any]]:  # pragma: no cover
        """Return ``(A_mat, b_vec, cones)`` for the Clarabel QP solver."""
        raise NotImplementedError

    @abstractmethod
    def _osqp_constraints(self) -> tuple[csc_matrix, np.ndarray, np.ndarray]:  # pragma: no cover
        """Return ``(A_mat, l_vec, u_vec)`` for the OSQP solver."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template solvers
    # ------------------------------------------------------------------

    def solve_kkt(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via the direct KKT system.

        Args:
            project: Clip weights to ``[0, ∞)`` and renormalize to sum to 1
                     after solving.  Set to ``False`` for custom constraints.

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and number of
            outer iterations taken.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_kkt()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, outer, _inner = self._constraint_active_set(self._kkt_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, outer

    def solve_cvxpy(self, *, project: bool = True, backend: str = "clarabel") -> tuple[np.ndarray, int]:
        """Solve via CVXPY with a configurable backend solver.

        Requires the ``convex`` extra::

            pip install fast-minimum-variance[convex]

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).
            backend: CVXPY solver name (default ``"clarabel"``; ``"osqp"`` is
                also supported).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and solver
            iteration count.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_cvxpy()
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= -1e-6).all())
            True
        """
        w = cp.Variable(self.n)
        if self.target is not None:
            # target is the penalty matrix M; decompose as M = chol chol^T so ||chol^T w||^2 = w^T M w
            chol = cholesky(self.target)
            objective = (1.0 - self.alpha) * cp.sum_squares(self.X @ w) / self.t + self.alpha * cp.sum_squares(
                chol.T @ w
            )
        else:
            objective = cp.sum_squares(self.X @ w) / self.t
        if self.rho != 0.0 and self.mu is not None:
            objective = objective - self.rho * (self.mu @ w)

        cvxpy_solver = cp.CLARABEL if backend.lower() == "clarabel" else cp.OSQP
        problem = cp.Problem(cp.Minimize(objective), self._cvxpy_constraints(w, cp))
        problem.solve(solver=cvxpy_solver)

        result = w.value
        if result is None:
            raise RuntimeError("CVXPY solver failed to find a solution")  # noqa: TRY003
        if project:
            result = self._clip_and_renormalize(result)
        return result, int(problem.solver_stats.num_iters or 0)

    def solve_cg(self, *, project: bool = True) -> tuple[np.ndarray, int, int]:
        """Solve via matrix-free conjugate gradients.

        Args:
            project: Clip weights to ``[0, ∞)`` and renormalize to sum to 1
                     after solving.  Set to ``False`` for custom constraints.

        Returns:
            ``(w, outer_steps, inner_iters)`` — weight vector, number of outer
            active-set steps, and total CG iterations summed across all steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, outer, inner = Problem(X).solve_cg()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, outer, inner = self._constraint_active_set(self._cg_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, outer, inner

    def solve_pcg(self, *, project: bool = True) -> tuple[np.ndarray, int, int]:
        """Solve via matrix-free PCG with RMT preconditioner (Section 5.3).

        Solves ``Sigma_LW_oracle x = 1`` using ``T0^RMT`` as preconditioner.
        Requires ``pcg_lr = (bar_lam, U_k, delta_k)`` from RMT preprocessing.
        The preconditioner is applied via the Woodbury identity at O(nk) per step;
        the system matvec costs O(nT).  Returns the oracle-LW minimum-variance
        portfolio — not the RMT portfolio — in O(sqrt(1/alpha_oracle)) iterations.

        Returns:
            ``(w, outer_steps, inner_iters)``

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> rng = np.random.default_rng(0)
            >>> X = rng.standard_normal((100, 5))
            >>> bar_lam = float(np.trace(X.T @ X / 100) / 5)
            >>> U_k = np.eye(5, 2)
            >>> delta_k = np.array([0.1, 0.05])
            >>> w, outer, inner = Problem(X, alpha=0.1, pcg_lr=(bar_lam, U_k, delta_k)).solve_pcg()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        if self.pcg_lr is None:
            raise ValueError("pcg_lr must be set; pass pcg_lr=(bar_lam, U_k, delta_k)")  # noqa: TRY003
        w, outer, inner = self._constraint_active_set(self._pcg_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, outer, inner

    def solve_nnls(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via non-negative least squares (scipy.optimize.nnls).

        The budget constraint is enforced by augmenting the return matrix
        with a heavily weighted all-ones row; non-negativity is handled
        natively by the Lawson-Hanson algorithm.  The covariance matrix
        ``X'X`` is formed internally by scipy.  Return tilt (``rho != 0``)
        is not supported.

        Args:
            project: Renormalize weights to sum to 1 after solving.
                     Clipping is a no-op (NNLS already gives ``w >= 0``).

        Returns:
            ``(w, 1)`` — weight vector of shape ``(N,)`` and iteration
            count (always 1; NNLS is a single direct solve).

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_nnls()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, iters = self._nnls_solve()
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_osqp(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via OSQP (operator-splitting QP solver).

        Assembles ``P = 2·Σ_LW`` as a sparse upper-triangular CSC matrix and
        calls OSQP directly. Returns ``(w, iters)`` where ``iters`` is the
        OSQP iteration count.

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and number of
            OSQP iterations.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_osqp()
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= -1e-6).all())
            True
        """
        n = self.n

        if self.target is None:
            p_dense = 2.0 * ((self.X.T @ self.X) / self.t)
        else:
            p_dense = 2.0 * ((1 - self.alpha) * (self.X.T @ self.X) / self.t + self.alpha * self.target)

        p_upper = triu(csc_matrix(p_dense), format="csc")

        q = np.zeros(n)
        if self.rho != 0.0 and self.mu is not None:
            q = -self.rho * self.mu

        a_mat, l_vec, u_vec = self._osqp_constraints()

        solver = osqp.OSQP()
        solver.setup(
            p_upper,
            q,
            a_mat,
            l_vec,
            u_vec,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )
        result = solver.solve()

        w = np.array(result.x)
        if project:
            w = self._clip_and_renormalize(w)
        return w, result.info.iter

    def solve_clarabel(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via Clarabel interior-point solver (direct API, no CVXPY overhead).

        Assembles ``P = 2·Σ_LW`` as a sparse CSC matrix and calls Clarabel
        directly, bypassing CVXPY's problem-construction overhead.  The
        problem-specific constraint data is supplied by ``_clarabel_constraints``.
        Returns ``(w, iters)`` where ``iters`` is the Clarabel iteration count.

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and number of
            interior-point iterations.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_clarabel()
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= -1e-6).all())
            True
        """
        n = self.n

        if self.target is None:
            p_dense = 2.0 * ((self.X.T @ self.X) / self.t)
        else:
            p_dense = 2.0 * ((1 - self.alpha) * (self.X.T @ self.X) / self.t + self.alpha * self.target)

        p_csc = csc_matrix(p_dense)

        q = np.zeros(n)
        if self.rho != 0.0 and self.mu is not None:
            q = -self.rho * self.mu

        a_mat, b_vec, cones = self._clarabel_constraints()

        settings = clarabel.DefaultSettings()  # ty:ignore[unresolved-attribute]
        settings.verbose = False
        sol = clarabel.DefaultSolver(p_csc, q, a_mat, b_vec, cones, settings).solve()  # ty:ignore[unresolved-attribute]

        w = np.array(sol.x)
        if project:
            w = self._clip_and_renormalize(w)
        return w, sol.iterations

    def solve_proximal(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via proximal gradient descent projected onto the probability simplex.

        Minimises ``0.5 * w^T Σ_LW w`` subject to ``w >= 0, sum(w) = 1``.
        The gradient is computed in two separate terms so that the per-step
        cost remains ``O(nT)`` regardless of whether shrinkage is applied:

            grad = (1-alpha)/T * X^T(Xw) + alpha * target @ w

        This avoids stacking a (T+n)xn matrix, which would inflate per-step
        cost by O(n) under shrinkage. Return tilt (``rho != 0``) is not supported.

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and the number
            of gradient steps taken.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_proximal()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        from .proximal import prox_gradient

        extra_grad: Callable[[np.ndarray], np.ndarray] | None
        if self.target is not None and self.alpha > 0.0:
            c = 1.0 - self.alpha
            mat = np.sqrt(c) / np.sqrt(self.t) * self.X
            alpha, target = self.alpha, self.target

            def extra_grad(v: np.ndarray, a: float = alpha, tgt: np.ndarray = target) -> np.ndarray:
                """Return the shrinkage gradient contribution a * target @ v."""
                result: np.ndarray = a * (tgt @ v)
                return result
        else:
            mat = self.X / np.sqrt(self.t)
            extra_grad = None

        vec = np.zeros(self.t)
        w, n_iters = prox_gradient(mat, vec, extra_grad=extra_grad)
        if project:
            w = self._clip_and_renormalize(w)
        return w, n_iters

    def solve_fista(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        r"""Solve via Nesterov-accelerated proximal gradient (FISTA).

        Uses the Beck-Teboulle momentum sequence to achieve $O(1/k^2)$
        convergence for convex objectives; for strongly convex $f$ with
        condition number $\\kappa$ the linear rate is $(1-1/\\sqrt{\\kappa})^k$,
        matching CG's asymptotic iteration count.  Same per-step cost
        $O(nT)$ as ``solve_proximal``, typically 2--10$\\times$ fewer
        iterations.

        Args:
            project: Clip and renormalize after solving (see ``solve_proximal``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and the number
            of gradient steps taken.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_fista()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        from .proximal import fista_gradient

        extra_grad: Callable[[np.ndarray], np.ndarray] | None
        if self.target is not None and self.alpha > 0.0:
            c = 1.0 - self.alpha
            mat = np.sqrt(c) / np.sqrt(self.t) * self.X
            alpha, target = self.alpha, self.target

            def extra_grad(v: np.ndarray, a: float = alpha, tgt: np.ndarray = target) -> np.ndarray:
                """Return the shrinkage gradient contribution a * target @ v."""
                result: np.ndarray = a * (tgt @ v)
                return result
        else:
            mat = self.X / np.sqrt(self.t)
            extra_grad = None

        vec = np.zeros(self.t)
        w, n_iters = fista_gradient(mat, vec, extra_grad=extra_grad)
        if project:
            w = self._clip_and_renormalize(w)
        return w, n_iters
