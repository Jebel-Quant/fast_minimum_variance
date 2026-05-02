"""Tests for fast_minimum_variance.api."""

import numpy as np

from fast_minimum_variance.problem import Problem


class TestProblemDefaults:
    """Tests for default field initialisation in __post_init__."""

    def test_n(self, problem):
        """N equals the number of columns in X."""
        assert problem.n == problem.X.shape[1]

    def test_m_default(self, problem):
        """M equals 1 for the default budget constraint."""
        assert problem._m == 1

    def test_A_default_shape(self, problem):  # noqa: N802
        """Default A is ones vector of shape (N, 1)."""
        assert problem.A.shape == (problem.n, 1)
        np.testing.assert_array_equal(problem.A, 1.0)

    def test_b_default(self, problem):
        """Default b is [1.0]."""
        np.testing.assert_array_equal(problem.b, [1.0])

    def test_C_default_shape(self, problem):  # noqa: N802
        """Default C is -eye(N), encoding long-only constraints."""
        np.testing.assert_array_equal(problem.C, -np.eye(problem.n))

    def test_d_default(self, problem):
        """Default d is zeros(N)."""
        np.testing.assert_array_equal(problem.d, np.zeros(problem.n))

    def test_rho_default(self, problem):
        """Default rho is 0.0."""
        assert problem.rho == 0.0

    def test_mu_default(self, problem):
        """Default mu is None."""
        assert problem.mu is None

    def test_gamma_default(self, problem):
        """Default gamma is 0.0."""
        assert problem.gamma == 0.0


class TestProblemCustomConstraints:
    """Tests for supplying custom A, b, C, d."""

    def test_custom_A_preserved(self):  # noqa: N802
        """Supplied A is not overwritten by __post_init__."""
        X = np.eye(3)  # noqa: N806
        A = np.ones((3, 2))  # noqa: N806
        b = np.ones(2)
        p = Problem(X, A=A, b=b)
        np.testing.assert_array_equal(p.A, A)

    def test_custom_C_preserved(self):  # noqa: N802
        """Supplied C is not overwritten by __post_init__."""
        X = np.eye(3)  # noqa: N806
        C = np.zeros((3, 1))  # noqa: N806
        d = np.array([0.5])
        p = Problem(X, C=C, d=d)
        np.testing.assert_array_equal(p.C, C)

    def test_m_reflects_custom_A(self):  # noqa: N802
        """M equals the number of columns of the supplied A."""
        X = np.eye(4)  # noqa: N806
        A = np.ones((4, 2))  # noqa: N806
        p = Problem(X, A=A, b=np.ones(2))
        assert p._m == 2


class TestProblemKkt:
    """Tests for Problem._kkt."""

    def test_default_shape(self, problem_small):
        """KKT matrix shape is (N+1, N+1) for the default budget constraint."""
        K, rhs = problem_small._kkt()  # noqa: N806
        assert K.shape == (problem_small.n + 1, problem_small.n + 1)
        assert rhs.shape == (problem_small.n + 1,)

    def test_symmetry(self, problem_small):
        """KKT matrix is symmetric."""
        K, _ = problem_small._kkt()  # noqa: N806
        np.testing.assert_allclose(K, K.T)

    def test_hessian_block_psd(self, problem_small):
        """Top-left (N, N) Hessian block is positive semi-definite."""
        K, _ = problem_small._kkt()  # noqa: N806
        n = problem_small.n
        np.testing.assert_allclose(K[:n, :n], K[:n, :n].T)
        assert np.all(np.linalg.eigvalsh(K[:n, :n]) >= -1e-10)

    def test_constraint_block(self, problem_small):
        """Off-diagonal constraint block equals A (ones column)."""
        K, _ = problem_small._kkt()  # noqa: N806
        n = problem_small.n
        np.testing.assert_array_equal(K[:n, n:], problem_small.A)

    def test_rhs_primal_zeros_no_return(self, problem_small):
        """Primal RHS block is zero when rho=0."""
        _, rhs = problem_small._kkt()
        np.testing.assert_array_equal(rhs[: problem_small.n], 0.0)

    def test_rhs_dual_equals_b(self, problem_small):
        """Dual RHS block equals b."""
        _, rhs = problem_small._kkt()
        np.testing.assert_array_equal(rhs[problem_small.n :], problem_small.b)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS block equals rho * mu."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = Problem(X, rho=0.5, mu=mu)._kkt()
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)

    def test_gamma_shifts_hessian(self):
        """Gamma adds 2*gamma*I to the Hessian block."""
        X = np.eye(3)  # noqa: N806
        K_base, _ = Problem(X)._kkt()  # noqa: N806
        K_reg, _ = Problem(X, gamma=1.0)._kkt()  # noqa: N806
        diff = K_reg[:3, :3] - K_base[:3, :3]
        np.testing.assert_allclose(diff, 2.0 * np.eye(3))

    def test_active_inequality_extends_system(self):
        """Pinning one inequality grows the KKT system by one row/col."""
        X = np.eye(3)  # noqa: N806
        p = Problem(X)
        K_base, _ = p._kkt()  # noqa: N806
        active = np.array([True, False, False])
        K_active, _ = p._kkt(active=active)  # noqa: N806
        assert K_active.shape == (K_base.shape[0] + 1, K_base.shape[1] + 1)


class TestProblemKktOperator:
    """Tests for Problem._kkt_operator."""

    def test_default_active_none(self, problem_small):
        """kkt_operator() with no argument uses active=None (all-False mask)."""
        op, rhs = problem_small._kkt_operator()
        assert op.shape == (problem_small.n + 1, problem_small.n + 1)
        assert rhs.shape == (problem_small.n + 1,)

    def test_matvec_matches_explicit_kkt(self, problem_small):
        """Matrix-free matvec agrees with the explicit KKT matrix."""
        op, _ = problem_small._kkt_operator()
        K, _ = problem_small._kkt()  # noqa: N806
        x = np.ones(problem_small.n + 1)
        np.testing.assert_allclose(op @ x, K @ x, atol=1e-10)

    def test_rhs_matches_kkt_rhs(self, problem_small):
        """RHS from kkt_operator matches RHS from kkt."""
        _, rhs_op = problem_small._kkt_operator()
        _, rhs_kkt = problem_small._kkt()
        np.testing.assert_allclose(rhs_op, rhs_kkt)

    def test_active_inequality_extends_system(self):
        """Pinning one inequality grows the operator size by one."""
        X = np.eye(3)  # noqa: N806
        p = Problem(X)
        op_base, _ = p._kkt_operator()
        active = np.array([True, False, False])
        op_active, _ = p._kkt_operator(active=active)
        assert op_active.shape == (op_base.shape[0] + 1, op_base.shape[1] + 1)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS block equals rho * mu."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = Problem(X, rho=0.5, mu=mu)._kkt_operator()
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)


class TestProblemNullSpaceOperator:
    """Tests for Problem._null_space_operator."""

    def test_default_active_none(self, problem_small):
        """null_space_operator() with no argument uses active=None (all-False mask)."""
        op, _rhs, w0, _P = problem_small._null_space_operator()  # noqa: N806
        assert op is not None
        assert w0.shape == (problem_small.n,)

    def test_particular_solution_satisfies_constraints(self, problem_small):
        """w0 from null_space_operator satisfies A^T w0 = b."""
        _, _, w0, _ = problem_small._null_space_operator()
        np.testing.assert_allclose(problem_small.A.T @ w0, problem_small.b, atol=1e-10)

    def test_null_space_basis_orthogonal_to_constraints(self, problem_small):
        """P spans null(A^T): A^T P = 0."""
        _, _, _, P = problem_small._null_space_operator()  # noqa: N806
        np.testing.assert_allclose(problem_small.A.T @ P, 0.0, atol=1e-10)

    def test_fully_determined_returns_none_op(self):
        """Returns (None, None, w0, None) when constraints fully determine w."""
        X = np.eye(2)  # noqa: N806
        p = Problem(X)
        active = np.ones(p.C.shape[1], dtype=bool)
        op, rhs, w0, P = p._null_space_operator(active=active)  # noqa: N806
        assert op is None
        assert rhs is None
        assert P is None
        assert w0.shape == (p.n,)


class TestProblemConstraintActiveSet:
    """Tests for Problem._constraint_active_set."""

    def test_no_violations_single_call(self):
        """solve_fn is called exactly once when the first solution is feasible."""
        X = np.eye(3)  # noqa: N806
        p = Problem(X)
        calls = []

        def solve_fn(active):
            """Record calls and return a feasible solution."""
            calls.append(active.copy())
            return np.array([1 / 3, 1 / 3, 1 / 3]), 1

        p._constraint_active_set(solve_fn)
        assert len(calls) == 1

    def test_iters_accumulated(self):
        """Total iters is the sum of step_iters across all active-set calls."""
        X = np.eye(2)  # noqa: N806
        p = Problem(X)
        call_count = [0]

        def solve_fn(active):
            """Return a solution that violates one constraint on the first call."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 1.1]), 3  # violates w[0] >= 0
            return np.array([0.0, 1.0]), 2

        _, total = p._constraint_active_set(solve_fn)
        assert total == 5

    def test_active_mask_starts_empty(self):
        """The first call to solve_fn receives an all-False active mask."""
        X = np.eye(2)  # noqa: N806
        p = Problem(X)
        first_active = []

        def solve_fn(active):
            """Capture the first active mask and return a feasible solution."""
            if not first_active:
                first_active.append(active.copy())
            return np.array([0.5, 0.5]), 1

        p._constraint_active_set(solve_fn)
        assert not first_active[0].any()

    def test_violated_constraint_promoted(self):
        """A constraint violated in round 1 is active in round 2."""
        X = np.eye(2)  # noqa: N806
        p = Problem(X)
        active_masks = []

        def solve_fn(active):
            """Return a solution violating constraint 0 on the first call."""
            active_masks.append(active.copy())
            if len(active_masks) == 1:
                return np.array([-0.1, 1.1]), 1
            return np.array([0.0, 1.0]), 1

        p._constraint_active_set(solve_fn)
        assert len(active_masks) == 2
        # constraint 0 corresponds to -w[0] <= 0, i.e. w[0] >= 0
        assert active_masks[1][0]
