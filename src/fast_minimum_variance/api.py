"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator


def clip_and_renormalize(w: np.ndarray) -> np.ndarray:
    """Clip weights to non-negative and renormalize to sum to one.

    This is a numerical cleanup step for iterative solvers that may produce
    small negative weights due to floating-point error.  It is only valid when
    the problem has a single budget constraint (sum(w) = 1) and long-only
    inequalities (w >= 0); applying it to general constraints corrupts the
    solution.
    """
    w = np.maximum(w, 0)
    w /= w.sum()
    return w


@dataclass(frozen=True)
class API:
    """Dataclass encoding the mean-variance portfolio problem.

    Solves::

        min  ||X w||^2 + gamma * ||w||^2 - rho * mu^T w
        s.t. A^T w  = b      (equality constraints)
             C^T w <= d      (inequality constraints)

    The first term ``||X w||^2`` is the sample portfolio variance (X is the
    demeaned return matrix of shape T x N).  The ``gamma`` term adds a
    Tikhonov / Ledoit-Wolf ridge to improve conditioning.  The ``rho * mu``
    term tilts the portfolio toward higher-expected-return assets (Markowitz).

    Defaults reproduce the long-only minimum-variance problem:

    * ``A = ones(N, 1)``, ``b = [1]``  — budget constraint: sum(w) = 1
    * ``C = -I``,         ``d = 0``    — long-only: -w <= 0, i.e. w >= 0

    All solvers (KKT, MINRES, CG, CVXPY) accept an ``API`` instance and
    delegate constraint handling to :meth:`constraint_active_set`.
    """

    X: np.ndarray
    rho: float = 0.0
    mu: np.ndarray | None = None
    # Ridge regularization strength; bakes in Ledoit-Wolf shrinkage when set.
    gamma: float = 0.0
    A: np.ndarray = field(default=None)  # type: ignore[assignment]
    b: np.ndarray = field(default=None)  # type: ignore[assignment]
    C: np.ndarray = field(default=None)  # type: ignore[assignment]
    d: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        """Fill in default constraint matrices when not supplied."""
        n = self.n
        if self.A is None:
            object.__setattr__(self, "A", np.ones((n, 1)))
        if self.b is None:
            object.__setattr__(self, "b", np.ones(1))
        if self.C is None:
            # Encode long-only as C^T w <= d  ⟺  -I w <= 0  ⟺  w >= 0.
            # The negative sign is intentional: the convention is C^T w <= d,
            # so long-only requires C = -I (not +I).
            object.__setattr__(self, "C", -np.eye(n))
        if self.d is None:
            object.__setattr__(self, "d", np.zeros(n))
        # object.__setattr__ is required because the dataclass is frozen;
        # ordinary attribute assignment raises FrozenInstanceError.

    @property
    def n(self) -> int:
        """Number of assets in the return matrix."""
        return self.X.shape[1]

    @property
    def m(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[1]

    def kkt(self, active=None):
        """Build the (N+m) x (N+m) KKT saddle-point system, optionally pinning active inequalities.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (K, rhs) where K is the KKT matrix and rhs is the right-hand side.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        # Augment the equality system with any currently active inequality
        # constraints: pinning C[:, active] as equalities turns the
        # inequality-constrained QP into a pure equality-constrained QP that
        # has a closed-form KKT solution.
        A = np.hstack([self.A, self.C[:, active]])  # noqa: N806
        b = np.concatenate([self.b, self.d[active]])

        m = A.shape[1]

        # The KKT first-order conditions for
        #   min  w^T H w - rho*mu^T w    s.t. A^T w = b
        # give the saddle-point (block) system
        #   [ H   A ] [ w ]   [ rho*mu ]
        #   [ A^T 0 ] [ λ ] = [ b      ]
        # where H = 2(X^T X + gamma I) is the Hessian of the objective.
        # The (0,0) block is the factor of 2 from differentiating w^T H w.
        # The bottom-right zero block has no quadratic term in the dual λ.
        K = np.zeros((self.n + m, self.n + m))  # noqa: N806
        K[: self.n, : self.n] = 2 * (self.X.T @ self.X + self.gamma * np.eye(self.n))
        K[: self.n, self.n :] = A
        K[self.n :, : self.n] = A.T

        rhs = np.zeros(self.n + m)
        if self.rho != 0.0 and self.mu is not None:
            rhs[: self.n] = self.rho * self.mu
        rhs[self.n :] = b

        return K, rhs

    def kkt_operator(self, active=None):
        """Build the matrix-free KKT saddle-point operator and RHS for MINRES.

        Returns a ``LinearOperator`` that applies the (N+m) x (N+m) KKT system
        matrix-free (never forming X^T X explicitly), and the matching RHS vector.

        The matrix-free approach costs O(T N) per matvec instead of O(N^2) for
        an explicit Gram matrix, which is the dominant saving when T << N^2.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (operator, rhs) where operator is a ``LinearOperator`` of shape
            (N+m, N+m) and rhs is the right-hand side vector of length N+m.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        na, ma = self.n, aa.shape[1]

        # The closure captures xx, a, n_, m_, gam as default arguments rather
        # than closing over `self`.  This avoids an attribute-lookup indirection
        # on every matvec call and makes the captured state explicit.
        def _matvec(x, xx=self.X, a=aa, n_=na, m_=ma, gam=self.gamma):
            """Apply the KKT saddle-point operator to vector x."""
            out = np.empty(n_ + m_)
            # Top block:    2(X^T(Xw) + gamma*w) + A*lambda
            # Computed as two sequential matrix-vector products to avoid forming X^T X.
            out[:n_] = 2.0 * (xx.T @ (xx @ x[:n_]) + gam * x[:n_]) + a @ x[n_:]
            # Bottom block: A^T w
            out[n_:] = a.T @ x[:n_]
            return out

        rhs = np.zeros(na + ma)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:na] = self.rho * self.mu
        rhs[na:] = np.concatenate([self.b, self.d[active]])

        return LinearOperator(shape=(na + ma, na + ma), matvec=_matvec), rhs  # type: ignore[call-arg]

    def null_space_operator(self, active=None):
        """Build the reduced null-space operator and RHS for CG.

        Computes a ``LinearOperator`` for ``P^T (X^T X + gamma I) P``, the
        projected RHS, the particular solution ``w0``, and the null-space basis
        ``P``.  When the constraints fully determine ``w`` (no free directions),
        returns ``(None, None, w0, None)``.

        Args:
            active: Boolean mask over inequality columns to pin as equalities.
                    Defaults to no active inequalities.

        Returns:
            Tuple (op, rhs, w0, P) where op is a ``LinearOperator`` of shape
            (n_free, n_free), rhs has length n_free, w0 has length N, and P
            has shape (N, n_free).  Returns (None, None, w0, None) when the
            system is fully determined.
        """
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        m_ext = aa.shape[1]
        n_free = self.n - m_ext
        b_ext = np.concatenate([self.b, self.d[active]])

        # lstsq gives the minimum-||w||^2 solution to A^T w = b_ext, placing
        # w0 in range(A) (the row space of A^T).  Any feasible w decomposes
        # as w = w0 + P v for some free vector v, where P is a basis for
        # null(A^T) — the orthogonal complement of range(A).
        w0 = np.linalg.lstsq(aa.T, b_ext, rcond=None)[0]

        if n_free <= 0:
            # The active constraints uniquely determine w = w0; no optimisation
            # step is needed.
            return None, None, w0, None

        # QR with mode="complete" gives a full square orthogonal Q.
        # Columns 0..m_ext-1 of Q span range(aa); the remaining n_free columns
        # span null(aa^T) and form an orthonormal basis P.
        Q, _ = np.linalg.qr(aa, mode="complete")  # noqa: N806
        P = Q[:, m_ext:]  # noqa: N806

        # Gradient of the objective at w = w0 (v = 0), without the factor of 2
        # (divided out consistently from both sides of the reduced system below).
        # Because w0 ∈ range(aa) and P ∈ null(aa^T) are orthogonal subspaces,
        # P^T w0 = 0, so the particular solution does not contribute to the
        # projected gradient — only the curvature at w0 matters.
        g0 = self.X.T @ (self.X @ w0) + self.gamma * w0
        if self.rho != 0.0 and self.mu is not None:
            # The full gradient is 2(X^T X + gamma I)w - rho*mu.  Dividing by 2
            # gives g0 = (X^T X + gamma I)w0 - (rho/2)*mu, consistent with the
            # un-scaled reduced operator below.
            g0 = g0 - (self.rho / 2.0) * self.mu

        # The projected RHS solves  (P^T(X^T X + gamma I)P) v = -P^T g0.
        # This system is SPD: X^T X is PSD, gamma >= 0, and P is full-rank,
        # so CG is applicable without a preconditioner.
        rhs = -(P.T @ g0)

        def _matvec(y, pp=P, xx=self.X, gam=self.gamma):
            # Applies P^T(X^T X + gamma I)P y in two steps to avoid forming X^T X.
            # Note: P^T P = I (P is orthonormal), so P^T(gamma I)P y = gamma y.
            pv = pp @ y
            return pp.T @ (xx.T @ (xx @ pv)) + gam * y

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        return op, rhs, w0, P

    def constraint_active_set(self, solve_fn):
        """Run the constraint active-set loop, promoting violated inequalities to equalities.

        Starts with no inequality constraints active and iteratively adds violated
        ones until all inactive constraints are satisfied.

        The loop is guaranteed to terminate: each iteration identifies at least one
        newly violated constraint and promotes it to the active set.  With finitely
        many inequality constraints there are finitely many active sets, so cycling
        cannot occur under the non-degeneracy assumption of the underlying QP.

        Args:
            solve_fn: Callable ``(active) -> (w, n_iters)`` that solves the
                      equality-constrained subproblem for the given active-constraint
                      mask and returns the full weight vector of shape (N,) and the
                      number of solver iterations.

        Returns:
            Tuple (w, total_iters).
        """
        p = self.d.size
        active = np.zeros(p, dtype=bool)
        total_iters = 0

        while True:
            w, step_iters = solve_fn(active)
            # Check only the currently inactive constraints; the active ones are
            # treated as equalities by the solver and are satisfied by construction.
            violations = self.C[:, ~active].T @ w - self.d[~active]
            total_iters += step_iters
            if np.all(violations <= 1e-10):
                # 1e-10 tolerance absorbs floating-point rounding from the solver
                # without accepting economically meaningful constraint violations.
                break
            # Promote all newly violated constraints simultaneously to avoid
            # multiple outer iterations for constraints that are jointly active.
            active[~active] |= violations > 1e-10

        return w, total_iters
