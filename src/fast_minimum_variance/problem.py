"""Global minimum-variance solver: matrix-free CG on the equality-constrained KKT system."""

from dataclasses import dataclass

import numpy as np

from .operators import build_system_operator, cg_solve_reduced


@dataclass(frozen=True)
class Problem:
    """Global minimum-variance portfolio (equality-constrained, sign-unconstrained).

    Solves::

        min  (1-alpha)||X w||^2/T + alpha*w^T T0 w - rho*mu^T w
        s.t. B w = c

    where ``B`` is a ``(p, N)`` balance system of full row rank.  The default
    (``B = None``) is the budget constraint ``1^T w = 1`` (``p = 1``).  Weights
    are sign-unconstrained, so this is the classic global minimum-variance
    portfolio and short positions are allowed.

    With no inequality constraint the KKT system is linear: stationarity
    ``2*Sigma*w = B^T lambda + rho*mu`` with ``Sigma = (1-alpha)/T X^T X +
    alpha*T0``, together with ``B w = c``, is solved in a single pass — no outer
    loop.  Matrix-free CG solves the SPD system ``Sigma V = B^T`` (``p``
    right-hand sides, plus ``Sigma v_mu = mu`` when ``rho != 0``) and a ``p x p``
    Schur solve recovers the multiplier, so the indefinite ``(n+p) x (n+p)``
    saddle-point system is never formed.  Call :meth:`solve_cg` to solve it.

    Args:
        X:      Returns matrix of shape ``(T, N)``.
        target: Optional ``(N, N)`` regularisation matrix; when supplied the
                shrinkage term ``alpha * w^T target w`` is added to the objective.
                ``None`` disables shrinkage entirely.
        alpha:     Shrinkage intensity; only active when ``target``/``target_lr``
                   is provided.  Use ``alpha = N/(N+T)`` for Ledoit-Wolf.
        rho:       Return tilt strength (Markowitz mean-variance).
        mu:        Expected returns vector ``(N,)``; required when ``rho != 0``.
        target_lr: Low-rank factored target ``(bar_lam, U_k, delta_k)`` for RMT
                   eigenvalue-cleaning; replaces ``target`` in the CG matvec.
        B:         Balance system ``(p, N)``: ``B w = c`` replaces the budget.
                   Must have full row rank; required together with ``c``.
        c:         Balance RHS ``(p,)``; required together with ``B``.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance import Problem
        >>> X = np.random.default_rng(0).standard_normal((100, 5))
        >>> w, *_ = Problem(X).solve_cg()
        >>> float(round(w.sum(), 6))
        1.0

        A two-sleeve balance system — each half of the universe holds half of
        the budget:

        >>> B = np.zeros((2, 5)); B[0, :3] = 1.0; B[1, 3:] = 1.0
        >>> w, *_ = Problem(X, B=B, c=np.array([0.5, 0.5])).solve_cg()
        >>> [float(round(s, 8)) for s in B @ w]
        [0.5, 0.5]
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

    def solve_cg(self) -> tuple[np.ndarray, int, int]:
        """Solve the equality-constrained problem via matrix-free conjugate gradients.

        Returns:
            ``(w, outer_steps, inner_iters)`` — the weight vector, ``outer_steps``
            (always ``1``; retained for API compatibility), and the total CG
            iteration count.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, outer, inner = Problem(X).solve_cg()
            >>> float(round(w.sum(), 10))
            1.0
        """
        w, iters = self._cg_step(np.ones(self.n, dtype=bool))
        return w, 1, iters

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
    # Matrix-free CG solve of the equality-constrained KKT system
    # ------------------------------------------------------------------

    def _cg_step(self, active: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Solve the SPD KKT system via matrix-free CG; return ``(w_a, iters)``.

        Builds the system operator (:func:`~fast_minimum_variance.operators.build_system_operator`),
        runs CG over it (:func:`~fast_minimum_variance.operators.cg_solve_reduced`)
        without ever forming ``Sigma`` explicitly, and recovers ``w`` from the
        balance Schur system. Low-rank and dense targets share one path.

        ``active`` is a boolean mask selecting the asset subset; ``solve_cg``
        always passes the full universe. ``x0`` is an optional warm-start guess
        of shape ``(active.sum(),)``.
        """
        sigma = build_system_operator(self.X, self.alpha, self.target, self.target_lr, self.t)
        b_a = self._balance_rows(active)
        mu_active = self.mu[active] if self.rho != 0.0 and self.mu is not None else None
        v_eq, v_mu, iters = cg_solve_reduced(sigma, active, b_a, mu_active, self._p, x0)
        return self._recover_balance(v_eq, v_mu, b_a), iters
