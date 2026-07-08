"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .active_set import run_active_set
from .operators import build_system_operator, cg_solve_reduced


@dataclass(frozen=True)
class _MinVarProblem:
    """Minimum-variance portfolio solver via primal-dual active-set iteration.

    Solves::

        min  (1-alpha)||X w||^2 + alpha*(||X||_F^2/N)*||w||^2 - rho*mu^T w
        s.t. B w = c,  w >= 0

    where ``B`` is a ``(p, N)`` balance system with full row rank on every
    active set the loop visits.  The default (``B = None``) is the budget
    constraint ``1^T w = 1`` (``p = 1``).

    Each inner step solves the equality-constrained subproblem over the current
    active asset set.  Stationarity gives ``2*Sigma_a*w_a = B_a^T lambda + rho*mu_a``
    where ``Sigma_a = (1-alpha)*X_a^T X_a + ridge*I``.  Solving the ``n_a x n_a``
    SPD system ``Sigma_a V = B_a^T`` (``p`` right-hand sides, plus
    ``Sigma_a v_mu = mu_a`` when ``rho != 0``) and recovering ``lambda`` from the
    ``p x p`` Schur system ``(B_a V) lambda = c`` avoids the indefinite
    ``(n_a+p) x (n_a+p)`` saddle-point system entirely.  The outer primal-dual
    loop enforces ``w >= 0`` and terminates when both primal and dual feasibility
    hold simultaneously.

    Use ``alpha = N/(N+T)`` for Ledoit-Wolf shrinkage intensity::

        T, N = X.shape
        w, outer, inner = Problem(X, alpha=N/(N+T)).solve_cg()

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance import Problem
        >>> X = np.random.default_rng(0).standard_normal((100, 5))
        >>> w, *_ = Problem(X).solve_cg()
        >>> float(round(w.sum(), 6))
        1.0
        >>> bool((w >= 0).all())
        True
    """

    X: np.ndarray
    target: np.ndarray | None = None
    alpha: float = 0.0
    rho: float = 0.0
    mu: np.ndarray | None = None
    target_lr: tuple[float, np.ndarray, np.ndarray] | None = None  # (bar_lam, U_k, delta_k) — low-rank + identity
    B: np.ndarray | None = None  # (p, N) balance system: B w = c; None = budget 1^T w = 1
    c: np.ndarray | None = None  # (p,) balance targets

    def __post_init__(self) -> None:
        """Validate target/target_lr and balance-system shapes."""
        self._validate_target()
        self._validate_target_lr()
        self._validate_balance()

    def _validate_target(self) -> None:
        """Reject a dense target whose shape is not ``(n, n)``."""
        n = self.n
        if self.target is not None and self.target.shape != (n, n):
            raise ValueError(f"target must be a square {n} x {n} matrix, got {self.target.shape}")  # noqa: TRY003

    def _validate_target_lr(self) -> None:
        """Reject a low-rank target whose ``U_k`` / ``delta_k`` shapes are inconsistent."""
        if self.target_lr is None:
            return
        n = self.n
        _bar_lam, U_k, delta_k = self.target_lr  # noqa: N806
        if U_k.shape[0] != n or U_k.shape[1] != delta_k.shape[0]:
            raise ValueError(  # noqa: TRY003
                f"target_lr: U_k must be ({n}, k) and delta_k (k,), got {U_k.shape}, {delta_k.shape}"
            )

    def _validate_balance(self) -> None:
        """Reject a mis-paired or mis-shaped balance system ``(B, c)``."""
        if (self.B is None) != (self.c is None):
            raise ValueError("B and c must be supplied together")  # noqa: TRY003
        if self.B is None:
            return
        c = self.c
        assert c is not None  # noqa: S101
        n = self.n
        if self.B.ndim != 2 or self.B.shape[1] != n:
            raise ValueError(f"B must have shape (p, {n}), got {self.B.shape}")  # noqa: TRY003
        if c.shape != (self.B.shape[0],):
            raise ValueError(f"c must have shape ({self.B.shape[0]},), got {c.shape}")  # noqa: TRY003

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

    # ------------------------------------------------------------------
    # Balance-system helpers (budget is the p = 1 special case)
    # ------------------------------------------------------------------

    @property
    def _p(self) -> int:
        """Number of balance constraints (1 for the default budget)."""
        return 1 if self.B is None else int(self.B.shape[0])

    def _c_vec(self) -> np.ndarray:
        """Balance right-hand side ``c`` (the budget gives ``[1.0]``)."""
        return np.ones(1) if self.c is None else self.c

    def _balance_rows(self, active: np.ndarray) -> np.ndarray:
        """Return ``B_a``, the balance system restricted to active assets, shape ``(p, n_a)``."""
        if self.B is None:
            return np.ones((1, int(active.sum())))
        return self.B[:, active]

    def _recover_balance(self, v_eq: np.ndarray, v_mu: np.ndarray | None, b_a: np.ndarray) -> np.ndarray:
        """Recover ``w_a`` from the Schur reduction of the balance system.

        Given ``V = Sigma_a^{-1} B_a^T`` (columns of ``v_eq``) and optionally
        ``v_mu = Sigma_a^{-1} mu_a``, stationarity ``2*Sigma_a*w = B_a^T lambda
        + rho*mu_a`` and feasibility ``B_a w = c`` pin the multiplier through
        the ``p x p`` SPD Schur system ``(B_a V) eta = c - rho/2 * B_a v_mu``
        with ``eta = lambda/2``, so ``w = V eta + rho/2 * v_mu``.
        """
        schur = b_a @ v_eq  # (p, p)
        rhs = self._c_vec().astype(np.float64)
        if v_mu is not None:
            rhs = rhs - 0.5 * self.rho * (b_a @ v_mu)
        eta = np.linalg.solve(schur, rhs) if self._p > 1 else rhs / schur[0, 0]
        w = v_eq @ eta
        if v_mu is not None:
            w = w + 0.5 * self.rho * v_mu
        return w

    # ------------------------------------------------------------------
    # Outer loop: primal elimination + dual feasibility check
    # ------------------------------------------------------------------

    def _constraint_active_set(
        self,
        solve_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> tuple[np.ndarray, int, int]:
        """Run the primal-dual active-set loop enforcing ``w >= 0`` (see :func:`run_active_set`)."""
        return run_active_set(self, solve_fn, tol, max_iter)

    # ------------------------------------------------------------------
    # Inner step
    # ------------------------------------------------------------------

    def _cg_step(self, active: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Solve the reduced SPD system via matrix-free CG; return ``(w_a, iters)``.

        Builds the active-set system operator (:func:`_build_system_operator`),
        runs CG over it restricted to the active set (:func:`_cg_solve_reduced`)
        without ever forming ``Sigma_a`` explicitly, and recovers ``w_a`` from the
        balance Schur system. Low-rank and dense targets share one path.

        Args:
            active: Boolean mask selecting the active asset subset.
            x0: Optional initial guess for the first CG solve (warm start).
                When provided it must have shape ``(active.sum(),)``.
        """
        sigma = build_system_operator(self.X, self.alpha, self.target, self.target_lr, self.t)
        b_a = self._balance_rows(active)
        mu_active = self.mu[active] if self.rho != 0.0 and self.mu is not None else None
        v_eq, v_mu, iters = cg_solve_reduced(sigma, active, b_a, mu_active, self._p, x0)
        return self._recover_balance(v_eq, v_mu, b_a), iters

    # ------------------------------------------------------------------
    # Weight projection
    # ------------------------------------------------------------------

    def _clip_and_renormalize(self, w: np.ndarray) -> np.ndarray:
        """Project onto the budget simplex; identity when a balance system is set.

        Renormalising by the weight sum would break a general ``B w = c``, and
        the active-set loop already exits primal-feasible, so balance-system
        solves return the iterate unchanged.
        """
        if self.B is not None:
            return w
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
