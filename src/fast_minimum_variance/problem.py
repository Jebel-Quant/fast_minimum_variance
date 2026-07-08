"""Global minimum-variance solver: a dense NumPy solve of the equality-constrained KKT system."""

from dataclasses import dataclass

import numpy as np


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
    loop.  ``Sigma`` is formed as a dense ``(N, N)`` matrix and ``np.linalg.solve``
    solves the SPD system ``Sigma V = B^T`` (``p`` right-hand sides, plus
    ``Sigma v_mu = mu`` when ``rho != 0``); a ``p x p`` Schur solve recovers the
    multiplier, so the indefinite ``(n+p) x (n+p)`` saddle-point system is never
    formed.  Call :meth:`solve` to obtain the weights.  Depends only on NumPy.

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
                   eigenvalue-cleaning; expanded to ``bar_lam*I + U_k diag(delta_k) U_k^T``
                   when building ``Sigma``.
        B:         Balance system ``(p, N)``: ``B w = c`` replaces the budget.
                   Must have full row rank; required together with ``c``.
        c:         Balance RHS ``(p,)``; required together with ``B``.

    Examples:
        >>> import numpy as np
        >>> from fast_minimum_variance import Problem
        >>> X = np.random.default_rng(0).standard_normal((100, 5))
        >>> w = Problem(X).solve()
        >>> float(round(w.sum(), 6))
        1.0

        A two-sleeve balance system — each half of the universe holds half of
        the budget:

        >>> B = np.zeros((2, 5)); B[0, :3] = 1.0; B[1, 3:] = 1.0
        >>> w = Problem(X, B=B, c=np.array([0.5, 0.5])).solve()
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

    def solve(self) -> np.ndarray:
        """Solve the equality-constrained problem and return the weight vector ``w``.

        Builds the dense system matrix ``Sigma`` (:meth:`_sigma`), solves
        ``Sigma V = B^T`` (and ``Sigma v_mu = mu`` when ``rho != 0``) with
        ``np.linalg.solve``, and recovers ``w`` from the ``p x p`` balance Schur
        system.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w = Problem(X).solve()
            >>> float(round(w.sum(), 10))
            1.0
        """
        sigma = self._sigma()
        b = self._b_matrix()
        v_eq = np.linalg.solve(sigma, b.T)  # (N, p): column j is Sigma^{-1} B[j]
        v_mu = np.linalg.solve(sigma, self.mu) if self.rho != 0.0 and self.mu is not None else None
        return self._recover_balance(v_eq, v_mu, b)

    def _sigma(self) -> np.ndarray:
        """Build the dense ``(N, N)`` system matrix ``Sigma = (1-alpha)/T X^T X + alpha T0``.

        ``T0`` is the shrinkage target: the low-rank ``bar_lam*I + U_k diag(delta_k)
        U_k^T`` when ``target_lr`` is set, else the dense ``target``. With no target
        the data term carries the full weight.
        """
        gram = (self.X.T @ self.X) / self.t
        if self.target_lr is not None:
            bar_lam, u_k, delta_k = self.target_lr
            t0 = bar_lam * np.eye(self.n) + (u_k * delta_k) @ u_k.T
            sigma = (1.0 - self.alpha) * gram + self.alpha * t0
        elif self.target is not None:
            sigma = (1.0 - self.alpha) * gram + self.alpha * self.target
        else:
            sigma = gram
        result: np.ndarray = sigma
        return result

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

    def _b_matrix(self) -> np.ndarray:
        """Balance matrix ``B`` of shape ``(p, N)`` (the budget gives a single ones row)."""
        return np.ones((1, self.n)) if self.B is None else self.B

    def _recover_balance(self, v_eq: np.ndarray, v_mu: np.ndarray | None, b: np.ndarray) -> np.ndarray:
        """Recover ``w`` from the Schur reduction of the balance system.

        Given ``V = Sigma^{-1} B^T`` (columns of ``v_eq``) and optionally
        ``v_mu = Sigma^{-1} mu``, stationarity ``2*Sigma*w = B^T lambda + rho*mu``
        and feasibility ``B w = c`` pin the multiplier through the ``p x p`` SPD
        Schur system ``(B V) eta = c - rho/2 * B v_mu`` with ``eta = lambda/2``,
        so ``w = V eta + rho/2 * v_mu``.
        """
        schur = b @ v_eq  # (p, p)
        rhs = self._c_vec().astype(np.float64)
        if v_mu is not None:
            rhs = rhs - 0.5 * self.rho * (b @ v_mu)
        eta = np.linalg.solve(schur, rhs) if self._p > 1 else rhs / schur[0, 0]
        w = v_eq @ eta
        if v_mu is not None:
            w = w + 0.5 * self.rho * v_mu
        return w
