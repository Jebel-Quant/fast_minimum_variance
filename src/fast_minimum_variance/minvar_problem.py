"""Minimum-variance solver: primal asset elimination with dual-feasibility check."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from cvx.linalg import DenseOperator, FactorOperator, GramOperator, SumOperator
from scipy.sparse.linalg import LinearOperator, cg

from ._base import _BaseProblem


@dataclass(frozen=True)
class _MinVarProblem(_BaseProblem):
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

    B: np.ndarray | None = None  # (p, N) balance system: B w = c; None = budget 1^T w = 1
    c: np.ndarray | None = None  # (p,) balance targets

    def __post_init__(self) -> None:
        """Validate balance-system shapes on top of the base checks."""
        super().__post_init__()
        if (self.B is None) != (self.c is None):
            raise ValueError("B and c must be supplied together")  # noqa: TRY003
        if self.B is not None:
            c = self.c
            assert c is not None  # noqa: S101
            if self.B.ndim != 2 or self.B.shape[1] != self.n:
                raise ValueError(f"B must have shape (p, {self.n}), got {self.B.shape}")  # noqa: TRY003
            if c.shape != (self.B.shape[0],):
                raise ValueError(f"c must have shape ({self.B.shape[0]},), got {c.shape}")  # noqa: TRY003

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
    # Shared helpers used by both active-set loop variants
    # ------------------------------------------------------------------

    def _compute_gradient(self, w: np.ndarray) -> np.ndarray:
        """Return the full objective gradient at w, including rho*mu adjustment."""
        data_grad = (self.X.T @ (self.X @ w)) / self.t
        if self.target_lr is not None:
            bar_lam, U_k, delta_k = self.target_lr  # noqa: N806
            tgt_w = bar_lam * w + U_k @ (delta_k * (U_k.T @ w))
            grad = 2.0 * ((1 - self.alpha) * data_grad + self.alpha * tgt_w)
        elif self.target is not None:
            grad = 2.0 * ((1 - self.alpha) * data_grad + self.alpha * self.target @ w)
        else:
            grad = 2.0 * data_grad
        if self.rho != 0.0 and self.mu is not None:
            grad = grad - self.rho * self.mu
        result: np.ndarray = grad
        return result

    @staticmethod
    def _primal_drop(w_a: np.ndarray, asset_active: np.ndarray, tol: float) -> bool:
        """Drop negative-weight assets from active set in-place; return True if any dropped."""
        if not np.any(w_a < -tol):
            return False
        idx = np.where(asset_active)[0]
        strong = w_a < -10 * tol
        if np.any(strong):
            asset_active[idx[strong]] = False
        else:
            asset_active[idx[np.argmin(w_a)]] = False
        return True

    def _dual_add(self, grad: np.ndarray, asset_active: np.ndarray, tol: float) -> int:
        """Return index of excluded asset that violates KKT dual condition, or -1 if none.

        The multiplier is estimated from the active gradient: for the budget the
        stationary ``lambda`` is a location estimate of ``g_a`` (median for
        robustness on larger sets); for a general balance system it is the
        least-squares solution of ``B_a^T lambda = g_a``.  The bound multiplier
        estimate is then ``nu = grad - B^T lambda``, which must be non-negative
        on excluded assets at the optimum.
        """
        excluded = ~asset_active
        if not excluded.any():
            return -1
        g_a = grad[asset_active]
        if self.B is None:
            lambda_ = np.median(g_a) if g_a.size > 5 else g_a.mean()
            nu = grad - lambda_
        else:
            b_a = self.B[:, asset_active]
            lam, *_ = np.linalg.lstsq(b_a.T, g_a, rcond=None)
            nu = grad - self.B.T @ lam
        idx_ex = np.where(excluded)[0]
        j = idx_ex[np.argmin(nu[excluded])]
        return int(j) if nu[j] < -tol else -1

    # ------------------------------------------------------------------
    # Outer loop: primal elimination + dual feasibility check
    # ------------------------------------------------------------------
    def _constraint_active_set(
        self,
        solve_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> tuple[np.ndarray, int, int]:
        """Run the primal-dual active-set loop enforcing ``w >= 0``.

        Calls ``solve_fn(active_mask)`` repeatedly.  The *primal step* drops assets
        with negative weights; the *dual step* re-adds any excluded asset whose KKT
        gradient condition is violated.  Terminates when both conditions hold
        simultaneously, which together with stationarity is sufficient for global
        optimality.
        """
        n = self.n
        asset_active = np.ones(n, dtype=bool)
        total_inner_iters = 0
        outer_steps = 0
        prev_active = None
        w = np.zeros(n)

        for _ in range(max_iter):
            if prev_active is not None and np.array_equal(prev_active, asset_active):
                break  # pragma: no cover - structurally unreachable safety guard
            prev_active = asset_active.copy()

            w_a, step_iters = solve_fn(asset_active)
            outer_steps += 1
            total_inner_iters += step_iters

            if self._primal_drop(w_a, asset_active, tol):
                continue

            w = np.zeros(n)
            w[asset_active] = w_a

            j = self._dual_add(self._compute_gradient(w), asset_active, tol)
            if j < 0:
                break
            asset_active[j] = True

        return w, outer_steps, total_inner_iters

    # ------------------------------------------------------------------
    # Inner steps
    # ------------------------------------------------------------------

    def _system_operator(self) -> SumOperator:
        """Build ``Sigma = (1-alpha)/T * X^T X + alpha * T0`` as a cvx-linalg operator.

        A :class:`~cvx.linalg.SumOperator` of the data Gram term and, when present,
        the target term (a :class:`~cvx.linalg.FactorOperator` for a low-rank RMT
        target, else a :class:`~cvx.linalg.DenseOperator`). The full-universe
        operators are sliced to the active set via ``apply_free``; nothing is
        formed at ``n x n``. Without a target the data term carries the full weight.
        """
        has_target = self.target_lr is not None or self.target is not None
        c_data = (1.0 - self.alpha) if has_target else 1.0
        terms: list[tuple[float, Any]] = [(c_data / self.t, GramOperator(self.X))]
        if self.target_lr is not None:
            bar_lam, u_k, delta_k = self.target_lr
            terms.append((self.alpha, FactorOperator(np.full(u_k.shape[0], bar_lam), u_k, np.diag(delta_k))))
        elif self.target is not None:
            terms.append((self.alpha, DenseOperator(self.target)))
        return SumOperator(terms)

    @staticmethod
    def _free_matvec(sigma: SumOperator, active_idx: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Return the free-block action ``v -> Sigma[A, A] v`` with the slice hoisted out.

        When cvx-linalg exposes ``restricted`` (>= 0.9.6), the pre-sliced free-block
        operator is built once here and its plain ``matvec`` is returned. Calling
        ``apply_free(idx, v)`` per CG iteration instead re-gathers the operator's
        storage (the Gram factor columns) on every call, which costs an order of
        magnitude more wall clock at identical iteration counts. The fallback keeps
        older cvx-linalg releases working.
        """
        restricted = getattr(sigma, "restricted", None)
        if restricted is not None:
            try:
                restricted_op = restricted(active_idx)
            except NotImplementedError:
                restricted_op = None
            if restricted_op is not None:
                matvec: Callable[[np.ndarray], np.ndarray] = restricted_op.matvec
                return matvec
        return lambda v: sigma.apply_free(active_idx, v)

    def _cg_step(self, active: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Solve the reduced SPD system via matrix-free CG; return ``(w_a, iters)``.

        Runs conjugate gradients over the active-set system operator
        (:meth:`_system_operator`), restricted to the active set once per step so
        the reduced matvec is ``O(n_a T)`` rather than ``O(n T)``, without ever
        forming ``Sigma_a`` explicitly. Low-rank and dense targets share one path.

        Args:
            active: Boolean mask selecting the active asset subset.
            x0: Optional initial guess for the first CG solve (warm start).
                When provided it must have shape ``(active.sum(),)``.
        """
        n_a = int(active.sum())
        sigma = self._system_operator()
        active_idx = np.flatnonzero(active)
        free_matvec = self._free_matvec(sigma, active_idx)
        count = [0]

        def matvec(v: np.ndarray) -> np.ndarray:
            """Apply Sigma_a to v via the pre-sliced free-block operator."""
            count[0] += 1
            return free_matvec(v)

        op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]

        b_a = self._balance_rows(active)
        # x0 approximates the final w, which is proportional to the single
        # solve column only in the budget case; skip the guess for p > 1.
        guess = x0 if self._p == 1 else None
        v_eq = np.column_stack([cg(op, b_a[j], x0=guess)[0] for j in range(self._p)])
        v_mu = cg(op, self.mu[active], x0=guess)[0] if self.rho != 0.0 and self.mu is not None else None
        return self._recover_balance(v_eq, v_mu, b_a), count[0]

    # ------------------------------------------------------------------
    # Budget-specific overrides
    # ------------------------------------------------------------------

    def _clip_and_renormalize(self, w: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Project onto the budget simplex; identity when a balance system is set.

        Renormalising by the weight sum would break a general ``B w = c``, and
        the active-set loop already exits primal-feasible, so balance-system
        solves return the iterate unchanged.
        """
        if self.B is not None:
            return w
        return _BaseProblem._clip_and_renormalize(w)
