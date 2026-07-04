"""Common base for portfolio-optimisation problem classes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from cvx.linalg import cholesky


@dataclass(frozen=True)
class _BaseProblem(ABC):
    """Shared fields, utilities, and solver templates for portfolio problems.

    Subclasses must implement the four abstract hooks:

    * ``_constraint_active_set(solve_fn)`` — outer constraint-handling loop
    * ``_kkt_step(mask) -> (w, iters)`` — one direct-KKT inner step
    * ``_cg_step(mask) -> (w, iters)`` — one CG inner step
    * ``_cvxpy_constraints(w, cp) -> list`` — CVXPY constraint list

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

    def solve_cvxpy(self, *, project: bool = True) -> tuple[np.ndarray, int]:
        """Solve via CVXPY with the Clarabel backend (reference solver).

        Requires the ``convex`` extra::

            pip install fast-minimum-variance[convex]

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

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

        problem = cp.Problem(cp.Minimize(objective), self._cvxpy_constraints(w, cp))
        problem.solve(solver=cp.CLARABEL)

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
