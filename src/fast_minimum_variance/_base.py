"""Common base for portfolio-optimisation problem classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class _BaseProblem(ABC):
    """Shared fields, utilities, and solver templates for portfolio problems.

    Subclasses must implement the five abstract hooks:

    * ``_constraint_active_set(solve_fn)`` — outer active-set loop
    * ``_kkt_step(mask) -> (w, iters)`` — one direct-KKT inner step
    * ``_minres_step(mask) -> (w, iters)`` — one MINRES inner step
    * ``_cg_step(mask) -> (w, iters)`` — one CG inner step
    * ``_cvxpy_constraints(w, cp) -> list`` — CVXPY constraint list

    All ``solve_*`` methods are implemented here as template methods that
    call ``_constraint_active_set`` with the appropriate ``_XXX_step``
    method, then optionally clip-and-renormalize.
    """

    X: np.ndarray
    alpha: float = 0.0
    rho: float = 0.0
    mu: np.ndarray = field(default=None)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of assets (columns of X)."""
        return self.X.shape[1]

    @staticmethod
    def _clip_and_renormalize(w: np.ndarray) -> np.ndarray:
        """Clip weights to ``[0, ∞)`` and renormalize to sum to 1."""
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    def _ridge(self) -> float:
        """Ridge coefficient: ``alpha * ||X||_F^2 / N``."""
        return self.alpha * np.einsum("ti,ti->", self.X, self.X) / self.n

    # ------------------------------------------------------------------
    # Abstract hooks (raise NotImplementedError — subclasses must override)
    # ------------------------------------------------------------------
    @abstractmethod
    def _constraint_active_set(self, solve_fn):  # pragma: no cover
        """Run the outer active-set loop, calling ``solve_fn`` each iteration."""
        raise NotImplementedError

    @abstractmethod
    def _kkt_step(self, mask):  # pragma: no cover
        """Solve one inner direct-KKT step; return ``(w, iters)``."""
        raise NotImplementedError

    @abstractmethod
    def _minres_step(self, mask):  # pragma: no cover
        """Solve one inner MINRES step; return ``(w, iters)``."""
        raise NotImplementedError

    @abstractmethod
    def _cg_step(self, mask):  # pragma: no cover
        """Solve one inner CG null-space step; return ``(w, iters)``."""
        raise NotImplementedError

    @abstractmethod
    def _cvxpy_constraints(self, w, cp):  # pragma: no cover
        """Return the list of CVXPY constraints for ``solve_cvxpy``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template solvers
    # ------------------------------------------------------------------

    def solve_kkt(self, *, project: bool = True):
        """Solve via the direct KKT system with active-set method.

        Args:
            project: Clip weights to ``[0, ∞)`` and renormalize to sum to 1
                     after solving.  Set to ``False`` for custom constraints.

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and number of
            active-set steps taken.

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
        w, iters = self._constraint_active_set(self._kkt_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_minres(self, *, project: bool = True):
        """Solve via MINRES with active-set method.

        Each active-set step solves a KKT saddle-point system matrix-free
        using MINRES.  No explicit covariance matrix is formed.

        To apply Ledoit-Wolf shrinkage::

            T, N = X.shape
            w, iters = Problem(X, alpha=N/(N+T)).solve_minres()

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and total
            MINRES iterations across all active-set steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_minres()
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= 0).all())
            True
            >>> iters > 0
            True
        """
        w, iters = self._constraint_active_set(self._minres_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_cg(self, *, project: bool = True):
        """Solve via CG in the constraint-eliminated null space with active-set.

        Each active-set step eliminates the equality constraints via QR,
        then applies CG to the reduced positive-definite system.

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and total
            CG iterations across all active-set steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_cg()
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= 0).all())
            True
            >>> iters > 0
            True
        """
        w, iters = self._constraint_active_set(self._cg_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_cvxpy(self, *, project: bool = True):
        """Solve via CVXPY / Clarabel (reference interior-point solver).

        Requires the ``convex`` extra::

            pip install fast-minimum-variance[convex]

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and Clarabel
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
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(  # noqa: TRY003
                "cvxpy is required; install with: pip install fast-minimum-variance[convex]"
            ) from e

        w = cp.Variable(self.n)
        ridge = self._ridge()
        objective = (1.0 - self.alpha) * cp.sum_squares(self.X @ w)
        if self.alpha != 0.0:
            objective = objective + ridge * cp.sum_squares(w)
        if self.rho != 0.0 and self.mu is not None:
            objective = objective - self.rho * (self.mu @ w)

        problem = cp.Problem(cp.Minimize(objective), self._cvxpy_constraints(w, cp))
        problem.solve(solver=cp.CLARABEL)

        result = w.value
        if result is None:
            raise RuntimeError("CVXPY solver failed to find a solution")  # noqa: TRY003
        if project:
            result = self._clip_and_renormalize(result)
        return result, problem.solver_stats.num_iters
