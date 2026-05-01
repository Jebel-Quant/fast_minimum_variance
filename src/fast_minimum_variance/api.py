"""Shared utilities for the minimum variance and Markowitz portfolio solvers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, minres


def clip_and_renormalize(w: np.ndarray) -> np.ndarray:
    """Clip weights to non-negative and renormalize to sum to one."""
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
    gamma: float = 0.0
    A: np.ndarray = field(default=None)  # type: ignore[assignment]
    b: np.ndarray = field(default=None)  # type: ignore[assignment]
    C: np.ndarray = field(default=None)  # type: ignore[assignment]
    d: np.ndarray = field(default=None)  # type: ignore[assignment]
    backend: str = "numpy"

    def __post_init__(self):
        """Fill in default constraint matrices when not supplied."""
        n = self.n
        if self.A is None:
            object.__setattr__(self, "A", np.ones((n, 1)))
        if self.b is None:
            object.__setattr__(self, "b", np.ones(1))
        if self.C is None:
            # Negative sign: convention is C^T w <= d, so long-only requires C = -I (not +I).
            object.__setattr__(self, "C", -np.eye(n))
        if self.d is None:
            object.__setattr__(self, "d", np.zeros(n))
        # object.__setattr__ is required because the dataclass is frozen.
        if self.backend not in ("numpy", "jax"):
            raise ValueError(  # noqa: TRY003
                f"Unknown backend {self.backend!r}. Accepted values are 'numpy' and 'jax'."
            )

    @property
    def n(self) -> int:
        """Number of assets in the return matrix."""
        return self.X.shape[1]

    @property
    def _m(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[1]

    def _kkt(self, active=None):
        """Build the (N+m) x (N+m) KKT saddle-point system."""
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        A = np.hstack([self.A, self.C[:, active]])  # noqa: N806
        b = np.concatenate([self.b, self.d[active]])
        m = A.shape[1]

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
        """Build the matrix-free KKT saddle-point operator and RHS for MINRES."""
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        na, ma = self.n, aa.shape[1]

        # Capture as default args rather than closing over `self` to avoid
        # attribute-lookup indirection on every matvec call.
        def _matvec(x, xx=self.X, a=aa, n_=na, m_=ma, gam=self.gamma):
            out = np.empty(n_ + m_)
            out[:n_] = 2.0 * (xx.T @ (xx @ x[:n_]) + gam * x[:n_]) + a @ x[n_:]
            out[n_:] = a.T @ x[:n_]
            return out

        rhs = np.zeros(na + ma)
        if self.rho != 0.0 and self.mu is not None:
            rhs[:na] = self.rho * self.mu
        rhs[na:] = np.concatenate([self.b, self.d[active]])

        return LinearOperator(shape=(na + ma, na + ma), matvec=_matvec), rhs  # type: ignore[call-arg]

    def _null_space_operator(self, active=None):
        """Build the reduced null-space operator and RHS for CG."""
        if active is None:
            active = np.zeros(self.C.shape[1], dtype=bool)
        aa = np.hstack([self.A, self.C[:, active]])
        m_ext = aa.shape[1]
        n_free = self.n - m_ext
        b_ext = np.concatenate([self.b, self.d[active]])

        w0 = np.linalg.lstsq(aa.T, b_ext, rcond=None)[0]

        if n_free <= 0:
            return None, None, w0, None

        Q, _ = np.linalg.qr(aa, mode="complete")  # noqa: N806
        P = Q[:, m_ext:]  # noqa: N806

        g0 = self.X.T @ (self.X @ w0) + self.gamma * w0
        if self.rho != 0.0 and self.mu is not None:
            g0 = g0 - (self.rho / 2.0) * self.mu

        rhs = -(P.T @ g0)

        def _matvec(y, pp=P, xx=self.X, gam=self.gamma):
            pv = pp @ y
            return pp.T @ (xx.T @ (xx @ pv)) + gam * y

        op = LinearOperator(shape=(n_free, n_free), matvec=_matvec)  # type: ignore[call-arg]
        return op, rhs, w0, P

    def _constraint_active_set(self, solve_fn):
        """Run the active-set loop, promoting violated inequalities to equalities."""
        p = self.d.size
        active = np.zeros(p, dtype=bool)
        total_iters = 0

        while True:
            w, step_iters = solve_fn(active)
            violations = self.C[:, ~active].T @ w - self.d[~active]
            total_iters += step_iters
            if np.all(violations <= 1e-10):
                break
            # Promote all violated constraints at once to avoid redundant outer iterations.
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
            K, rhs = self._kkt(active=active)  # noqa: N806
            w = np.linalg.solve(K, rhs)[: self.n]
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
            kkt, rhs = self._kkt_operator(active)
            # Mutable list to count iterations from inside the callback; a plain int
            # cannot be rebound in the enclosing scope without `nonlocal`.
            iters = [0]
            sol, _ = minres(kkt, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
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

        When ``backend='jax'`` this method dispatches to ``_solve_cg_jax``, which
        runs the matvec kernel on JAX (e.g. Apple Silicon via ``jax-metal``).

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
        if self.backend == "jax":
            return self._solve_cg_jax(project=project)

        def _solve(active):
            """Solve the CG null-space subproblem for the current active set."""
            op, rhs, w0, P = self._null_space_operator(active)  # noqa: N806
            if op is None:
                return w0, 0
            # Mutable list to count iterations from inside the callback; a plain int
            # cannot be rebound in the enclosing scope without `nonlocal`.
            iters = [0]
            sol, _ = cg(op, rhs, callback=lambda _x: iters.__setitem__(0, iters[0] + 1))
            return w0 + P @ sol, iters[0]

        w, iters = self._constraint_active_set(_solve)
        if project:
            w = clip_and_renormalize(w)
        return w, iters

    def _solve_cg_jax(self, *, project: bool = True):
        """Solve via JAX-accelerated CG in the null space (JAX backend).

        Implements the same null-space reduction as the NumPy CG path, but uses
        ``jax.scipy.sparse.linalg.cg`` with JAX arrays so that the two matvecs
        per iteration run on the available JAX accelerator (e.g. Apple Silicon
        via ``jax-metal``).

        All arrays are converted to ``float32`` before the JAX solve because
        ``jax-metal`` has limited ``float64`` support.  Results are returned as
        NumPy arrays so the rest of the active-set loop is unaffected.

        Requires JAX to be installed::

            pip install fast-minimum-variance[jax]
            pip install jax-metal          # Apple Silicon only

        Args:
            project: If True (default), clip weights to non-negative and
                     renormalize to sum to one after solving.

        Returns:
            Tuple (w, n_iters) where w is the weight vector of shape (N,) and
            n_iters is the total number of CG iterations across all active-set
            steps.
        """
        try:
            import jax.numpy as jnp
            from jax.scipy.sparse.linalg import cg as jax_cg
        except ImportError as e:
            raise ImportError(  # noqa: TRY003
                "JAX is required for backend='jax'; install with: "
                "pip install fast-minimum-variance[jax]"
            ) from e

        # Convert to float32 — jax-metal has limited float64 support.
        xx = jnp.array(self.X, dtype=jnp.float32)  # noqa: N806
        aa_eq = jnp.array(self.A, dtype=jnp.float32)
        bb_eq = jnp.array(self.b, dtype=jnp.float32)
        cc = jnp.array(self.C, dtype=jnp.float32)
        dd = jnp.array(self.d, dtype=jnp.float32)
        mu_jax = jnp.array(self.mu, dtype=jnp.float32) if self.mu is not None else None
        gam = float(self.gamma)
        rho = float(self.rho)

        def _solve(active):
            """Solve the JAX CG null-space subproblem for the current active set."""
            # Build the extended constraint matrix on CPU (NumPy) for QR / lstsq.
            active_np = np.asarray(active)
            aa_np = np.hstack([np.asarray(aa_eq), np.asarray(cc)[:, active_np]])
            m_ext = aa_np.shape[1]
            n_free = self.n - m_ext
            b_ext_np = np.concatenate([np.asarray(bb_eq), np.asarray(dd)[active_np]])

            w0_np = np.linalg.lstsq(aa_np.T, b_ext_np, rcond=None)[0]

            if n_free <= 0:
                return w0_np, 0

            Q_np, _ = np.linalg.qr(aa_np, mode="complete")  # noqa: N806
            P_np = Q_np[:, m_ext:]  # noqa: N806

            # Move P and w0 to JAX.
            P_jax = jnp.array(P_np, dtype=jnp.float32)  # noqa: N806
            w0_jax = jnp.array(w0_np, dtype=jnp.float32)

            g0 = xx.T @ (xx @ w0_jax) + gam * w0_jax
            if rho != 0.0 and mu_jax is not None:
                g0 = g0 - (rho / 2.0) * mu_jax
            rhs_jax = -(P_jax.T @ g0)

            def _matvec(y, pp=P_jax, xx=xx, gam=gam):
                pv = pp @ y
                return pp.T @ (xx.T @ (xx @ pv)) + gam * y

            sol_jax, _ = jax_cg(_matvec, rhs_jax)
            w_jax = w0_jax + P_jax @ sol_jax
            # Return as NumPy so the active-set loop stays backend-agnostic.
            return np.asarray(w_jax, dtype=np.float64), 1

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

        objective = cp.sum_squares(self.X @ w)
        if self.gamma != 0.0:
            objective = objective + self.gamma * cp.sum_squares(w)
        if self.rho != 0.0:
            objective = objective - self.rho * (self.mu @ w)

        constraints = [self.A.T @ w == self.b, self.C.T @ w <= self.d]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL)

        result = w.value
        if result is None:
            raise RuntimeError("CVXPY solver failed to find a solution")  # noqa: TRY003
        if project:
            result = clip_and_renormalize(result)
        return result, problem.solver_stats.num_iters
