"""Common base for portfolio-optimisation problem classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import clarabel
import cvxpy as cp
import numpy as np
import osqp
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

    def __post_init__(self):
        """Validate target shape when supplied; shrinkage is only active when target is not None."""
        n = self.n
        if self.target is not None and self.target.shape != (n, n):
            raise ValueError(f"target must be a square {n} x {n} matrix, got {self.target.shape}")  # noqa: TRY003

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def t(self) -> int:
        """Return the number of rows in X."""
        return self.X.shape[0]

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

    # ------------------------------------------------------------------
    # Abstract hooks (raise NotImplementedError — subclasses must override)
    # ------------------------------------------------------------------
    @abstractmethod
    def _constraint_active_set(self, solve_fn, tol=1e-6, max_iter=10_000):  # pragma: no cover
        """Run the outer constraint-handling loop, calling ``solve_fn`` each iteration."""
        raise NotImplementedError

    @abstractmethod
    def _kkt_step(self, active):  # pragma: no cover
        """Solve one inner direct-KKT step; return ``(w, iters)``."""
        raise NotImplementedError

    @abstractmethod
    def _cvxpy_constraints(self, w, cp):  # pragma: no cover
        """Return the list of CVXPY constraints for ``solve_cvxpy``."""
        raise NotImplementedError

    @abstractmethod
    def _cg_step(self, active):
        """Solve one inner CG step; return ``(w, iters)``."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _precon_cg_step(self, active):
        """Solve one inner CG step; return ``(w, iters)``."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _nnls_solve(self):  # pragma: no cover
        """Solve via NNLS directly (no outer loop); return ``(w, 1)``."""
        raise NotImplementedError

    @abstractmethod
    def _clarabel_constraints(self):  # pragma: no cover
        """Return ``(A_mat, b_vec, cones)`` for the Clarabel QP solver."""
        raise NotImplementedError

    @abstractmethod
    def _osqp_constraints(self):  # pragma: no cover
        """Return ``(A_mat, l_vec, u_vec)`` for the OSQP solver."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template solvers
    # ------------------------------------------------------------------

    def solve_kkt(self, *, project: bool = True):
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
        w, iters = self._constraint_active_set(self._kkt_step)
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
        w = cp.Variable(self.n)
        if self.target is not None:
            # target is the penalty matrix M; decompose as M = chol chol^T so ||chol^T w||^2 = w^T M w
            chol = np.linalg.cholesky(self.target)
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
        return result, problem.solver_stats.num_iters

    def solve_cg(self, *, project: bool = True):
        """Solve via matrix-free conjugate gradients.

        Args:
            project: Clip weights to ``[0, ∞)`` and renormalize to sum to 1
                     after solving.  Set to ``False`` for custom constraints.

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and total CG
            iteration count across all outer active-set steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_cg()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, iters = self._constraint_active_set(self._cg_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_precon_cg(self, *, project: bool = True):
        """Solve via matrix-free conjugate gradients.

        Args:
            project: Clip weights to ``[0, ∞)`` and renormalize to sum to 1
                     after solving.  Set to ``False`` for custom constraints.

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and total CG
            iteration count across all outer active-set steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_precon_cg()
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """
        w, iters = self._constraint_active_set(self._precon_cg_step)
        if project:
            w = self._clip_and_renormalize(w)
        return w, iters

    def solve_nnls(self, *, project: bool = True):
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

    def solve_clarabel(self, *, project: bool = True):
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

        settings = clarabel.DefaultSettings()  # type: ignore[attr-defined]
        settings.verbose = False
        sol = clarabel.DefaultSolver(p_csc, q, a_mat, b_vec, cones, settings).solve()  # type: ignore[attr-defined]

        w = np.array(sol.x)
        if project:
            w = self._clip_and_renormalize(w)
        return w, sol.iterations

    def solve_osqp(self, *, project: bool = True):
        """Solve via OSQP (operator-splitting QP solver, direct API, no CVXPY overhead).

        Assembles ``P = 2·Σ_LW`` as a sparse upper-triangular CSC matrix and
        calls OSQP directly.  The problem-specific constraint data is supplied
        by ``_osqp_constraints``.  Returns ``(w, iters)`` where ``iters`` is
        the number of ADMM iterations.

        Args:
            project: Clip and renormalize after solving (see ``solve_kkt``).

        Returns:
            ``(w, n_iters)`` — weight vector of shape ``(N,)`` and number of
            ADMM iterations.

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

        # p_dense = 2.0 * ((1-self.alpha) * (self.X.T @ self.X)/self.t + self.alpha * self.target)
        p_upper = triu(p_dense, format="csc")

        q = np.zeros(n)
        if self.rho != 0.0 and self.mu is not None:
            q = -self.rho * self.mu

        a_mat, l_vec, u_vec = self._osqp_constraints()

        prob = osqp.OSQP()
        prob.setup(p_upper, q, a_mat, l_vec, u_vec, verbose=False)
        res = prob.solve()

        w = np.array(res.x)
        if project:
            w = self._clip_and_renormalize(w)
        return w, res.info.iter
