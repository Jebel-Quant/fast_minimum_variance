"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


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
class Problem:
    """Mean-variance portfolio problem specification and solver interface.

    Encodes the optimization problem::

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

    Solvers are available as methods::

        w, iters = Problem(X).solve_kkt()
        w, iters = Problem(X).solve_minres()
        w, iters = Problem(X).solve_cg()
        w, iters = Problem(X).solve_cvxpy()   # requires fast-minimum-variance[convex]

    Each returns ``(w, n_iters)`` where ``w`` is the weight vector and
    ``n_iters`` is the iteration count (1 for the direct KKT solver).
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
    def _m(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[1]

    def _kkt(self, active=None):
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

    def _kkt_operator(self, active=None):
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

    def _null_space_operator(self, active=None):
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

    def _constraint_active_set(self, solve_fn):
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

    def solve_kkt(self, *, project: bool = True):
        """Solve via the direct KKT system with active-set method.

        Iteratively promotes violated inequality constraints to equalities until
        all inactive constraints are satisfied, solving the KKT system exactly at
        each iteration via ``numpy.linalg.solve``.

        Args:
            project: If True (default), clip weights to non-negative and renormalize
                     to sum to one after solving.  Only correct for the default
                     long-only minimum-variance problem; set to False when using
                     custom constraints.

        Returns:
            Tuple (w, n_iters) where w is the weight vector of shape (N,) and
            n_iters is the number of active-set steps taken.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance.api import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_kkt()
            >>> w.shape
            (5,)
            >>> float(round(w.sum(), 10))
            1.0
            >>> bool((w >= 0).all())
            True
        """

        def fn(active):
            """Solve the KKT system for the current active set."""
            # Pin active inequalities as equalities by appending their columns to A.
            # When active is empty, hstack returns A unchanged (C[:,active] is (n, 0)).
            K, rhs = self._kkt(active=active)  # noqa: N806
            # np.linalg.solve returns the full KKT solution [w; lambda], where the
            # first N entries are the primal weights and the remainder are the dual
            # Lagrange multipliers.  Only w is needed here.
            w = np.linalg.solve(K, rhs)[: self.n]
            # Return 1 as the iteration count: the direct KKT solve is a single
            # linear-algebra step, not an iterative method.
            return w, 1

        w, iters = self._constraint_active_set(fn)
        if project:
            w = clip_and_renormalize(w)
        return w, iters

    def solve_minres(self, *, project: bool = True):
        """Solve via MINRES on the KKT saddle-point system with active-set method.

        Iteratively promotes violated inequality constraints to equalities.  At each
        outer iteration the KKT saddle-point system for all assets with the currently
        active constraints pinned as equalities

            [ 2(X^T X + gamma I)   A_ext ] [ w   ]   [ rho * mu ]
            [ A_ext^T               0    ] [ λ   ] = [ b_ext    ]

        is solved matrix-free via MINRES, where ``A_ext = [A, C[:, active]]`` and
        ``b_ext = [b, d[active]]``.  No explicit matrix is ever formed.

        With the defaults (``A = ones``, ``b = [1]``, ``C = -I``, ``d = 0``) this
        recovers the long-only minimum variance solver of the companion paper.

        To apply Ledoit-Wolf shrinkage, pre-scale the return matrix and set gamma::

            T, N     = X.shape
            frob_sq  = (X * X).sum()
            alpha    = N / (N + T)
            X_scaled = np.sqrt(1.0 - alpha) * X
            gamma    = frob_sq / (N + T)
            w, iters = Problem(X_scaled, gamma=gamma).solve_minres()

        Args:
            project: If True (default), clip weights to non-negative and renormalize
                     to sum to one after solving.  Only correct for the default
                     long-only minimum-variance problem; set to False when using
                     custom constraints.

        Returns:
            Tuple (w, n_iters) where w is the weight vector of shape (N,) and
            n_iters is the total number of MINRES iterations across all active-set
            steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance.api import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_minres()
            >>> w.shape
            (5,)
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= 0).all())
            True
            >>> iters > 0
            True
        """

        def _solve(active):
            """Solve the MINRES saddle-point system for the current active set."""
            # MINRES (not CG) is required here because the KKT saddle-point matrix
            # is symmetric but indefinite: the zero bottom-right block causes negative
            # eigenvalues, ruling out CG which requires positive definiteness.
            kkt, rhs = self._kkt_operator(active)
            # Use a mutable list to count iterations from inside the callback.
            # A plain int cannot be rebound in the enclosing scope via the callback;
            # mutating a list element sidesteps that restriction without `nonlocal`.
            iters = [0]
            sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
            # The solution vector is [w; lambda].  Discard the dual part (lambda)
            # and return only the primal weights.
            return sol[: self.n], iters[0]

        w, iters = self._constraint_active_set(_solve)
        if project:
            w = clip_and_renormalize(w)
        return w, iters

    def solve_cg(self, *, project: bool = True):
        """Solve via CG in the constraint-reduced null space with active-set method.

        At each active-set iteration the equality-constrained subproblem (with
        currently active inequalities pinned as equalities) is solved by projecting
        onto the null space of ``A_ext^T`` via QR factorisation, then applying CG
        to the reduced positive-definite system

            ((X P)^T (X P) + gamma I) v = -P^T (X^T X w0 + gamma w0 - (rho/2) mu)

        where ``A_ext = [A, C[:, active]]``, ``P`` is an orthonormal null-space
        basis for ``A_ext^T``, and ``w0`` is the minimum-norm particular solution
        of ``A_ext^T w = b_ext``.  The full weight vector is recovered as
        ``w = w0 + P v``.

        The reduced operator ``P^T(X^T X + gamma I)P`` is SPD, so CG converges
        in at most ``n_free`` iterations and benefits from any spectral clustering
        induced by shrinkage (larger ``gamma`` compresses the eigenvalue spread).

        See ``solve_minres`` for the Ledoit-Wolf shrinkage recipe.

        Args:
            project: If True (default), clip weights to non-negative and renormalize
                     to sum to one after solving.  Only correct for the default
                     long-only minimum-variance problem; set to False when using
                     custom constraints.

        Returns:
            Tuple (w, n_iters) where w is the weight vector of shape (N,) and
            n_iters is the total number of CG iterations across all active-set
            steps.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance.api import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_cg()
            >>> w.shape
            (5,)
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= 0).all())
            True
            >>> iters > 0
            True
        """

        def _solve(active):
            """Solve the CG null-space subproblem for the current active set."""
            op, rhs, w0, P = self._null_space_operator(active)  # noqa: N806
            if op is None:
                # All free directions are pinned by the active constraints; the
                # constraints uniquely determine w = w0 with no further optimisation.
                return w0, 0
            # Same mutable-list trick as in solve_minres: the callback cannot rebind
            # a plain int from the enclosing scope.
            iters = [0]
            sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
            # Reconstruct the full weight vector: w0 is the particular solution that
            # satisfies the constraints exactly, and P @ sol is the correction in the
            # constraint null space that minimises the objective.
            return w0 + P @ sol, iters[0]

        w, iters = self._constraint_active_set(_solve)
        if project:
            w = clip_and_renormalize(w)
        return w, iters

    def solve_cvxpy(self, *, project: bool = True):
        """Solve via CVXPY (reference interior-point solver).

        Solves the problem using CLARABEL via CVXPY as a reference implementation.
        Useful for validating the KKT and Krylov solvers, but significantly slower
        for large N.

        Requires the ``convex`` extra::

            pip install fast-minimum-variance[convex]

        Args:
            project: If True (default), clip weights to non-negative and renormalize
                     to sum to one after solving.  Only correct for the default
                     long-only minimum-variance problem; set to False when using
                     custom constraints.

        Returns:
            Tuple (w, n_iters) where w is the weight vector of shape (N,) and
            n_iters is the number of interior-point iterations reported by CLARABEL.

        Examples:
            >>> import numpy as np
            >>> from fast_minimum_variance.api import Problem
            >>> X = np.random.default_rng(0).standard_normal((100, 5))
            >>> w, iters = Problem(X).solve_cvxpy()
            >>> w.shape
            (5,)
            >>> float(round(w.sum(), 6))
            1.0
            >>> bool((w >= -1e-6).all())
            True
        """
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(  # noqa: TRY003
                "cvxpy is required for solve_cvxpy; install with: pip install fast-minimum-variance[convex]"
            ) from e

        w = cp.Variable(self.n)

        # sum_squares(X @ w) = ||X w||^2 = w^T (X^T X) w.
        # CVXPY recognises this as a quadratic form and passes it to CLARABEL as a
        # second-order cone constraint without ever forming X^T X explicitly.
        objective = cp.sum_squares(self.X @ w)

        # Ridge / Ledoit-Wolf regularization: add gamma * ||w||^2.
        if self.gamma != 0.0:
            objective = objective + self.gamma * cp.sum_squares(w)

        # Subtract the return term when rho > 0 (Markowitz mean-variance tilt).
        if self.rho != 0.0:
            objective = objective - self.rho * (self.mu @ w)

        constraints = [self.A.T @ w == self.b, self.C.T @ w <= self.d]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        # CLARABEL is an interior-point solver for second-order cone programs.
        problem.solve(solver=cp.CLARABEL)

        result = w.value
        if result is None:
            raise RuntimeError("CVXPY solver failed to find a solution")  # noqa: TRY003
        if project:
            result = clip_and_renormalize(result)
        # solver_stats.num_iters is the interior-point iteration count reported by
        # CLARABEL; it is analogous to the Krylov iteration counts returned by the
        # other solvers, enabling a fair comparison.
        return result, problem.solver_stats.num_iters
