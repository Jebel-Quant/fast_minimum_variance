"""Tests for fast_minimum_variance.api."""

import numpy as np

from fast_minimum_variance.api import API


class TestAPIDefaults:
    """Tests for default field initialisation in __post_init__."""

    def test_n(self, api):
        """N equals the number of columns in X."""
        assert api.n == api.X.shape[1]

    def test_m_default(self, api):
        """M equals 1 for the default budget constraint."""
        assert api.m == 1

    def test_A_default_shape(self, api):  # noqa: N802
        """Default A is ones vector of shape (N, 1)."""
        assert api.A.shape == (api.n, 1)
        np.testing.assert_array_equal(api.A, 1.0)

    def test_b_default(self, api):
        """Default b is [1.0]."""
        np.testing.assert_array_equal(api.b, [1.0])

    def test_C_default_shape(self, api):  # noqa: N802
        """Default C is -eye(N), encoding long-only constraints."""
        np.testing.assert_array_equal(api.C, -np.eye(api.n))

    def test_d_default(self, api):
        """Default d is zeros(N)."""
        np.testing.assert_array_equal(api.d, np.zeros(api.n))

    def test_rho_default(self, api):
        """Default rho is 0.0."""
        assert api.rho == 0.0

    def test_mu_default(self, api):
        """Default mu is None."""
        assert api.mu is None

    def test_gamma_default(self, api):
        """Default gamma is 0.0."""
        assert api.gamma == 0.0


class TestAPICustomConstraints:
    """Tests for supplying custom A, b, C, d."""

    def test_custom_A_preserved(self):  # noqa: N802
        """Supplied A is not overwritten by __post_init__."""
        X = np.eye(3)  # noqa: N806
        A = np.ones((3, 2))  # noqa: N806
        b = np.ones(2)
        api = API(X, A=A, b=b)
        np.testing.assert_array_equal(api.A, A)

    def test_custom_C_preserved(self):  # noqa: N802
        """Supplied C is not overwritten by __post_init__."""
        X = np.eye(3)  # noqa: N806
        C = np.zeros((3, 1))  # noqa: N806
        d = np.array([0.5])
        api = API(X, C=C, d=d)
        np.testing.assert_array_equal(api.C, C)

    def test_m_reflects_custom_A(self):  # noqa: N802
        """M equals the number of columns of the supplied A."""
        X = np.eye(4)  # noqa: N806
        A = np.ones((4, 2))  # noqa: N806
        api = API(X, A=A, b=np.ones(2))
        assert api.m == 2


class TestAPIKkt:
    """Tests for API.kkt."""

    def test_default_shape(self, api_small):
        """KKT matrix shape is (N+1, N+1) for the default budget constraint."""
        K, rhs = api_small.kkt()  # noqa: N806
        assert K.shape == (api_small.n + 1, api_small.n + 1)
        assert rhs.shape == (api_small.n + 1,)

    def test_symmetry(self, api_small):
        """KKT matrix is symmetric."""
        K, _ = api_small.kkt()  # noqa: N806
        np.testing.assert_allclose(K, K.T)

    def test_hessian_block_psd(self, api_small):
        """Top-left (N, N) Hessian block is positive semi-definite."""
        K, _ = api_small.kkt()  # noqa: N806
        n = api_small.n
        np.testing.assert_allclose(K[:n, :n], K[:n, :n].T)
        assert np.all(np.linalg.eigvalsh(K[:n, :n]) >= -1e-10)

    def test_constraint_block(self, api_small):
        """Off-diagonal constraint block equals A (ones column)."""
        K, _ = api_small.kkt()  # noqa: N806
        n = api_small.n
        np.testing.assert_array_equal(K[:n, n:], api_small.A)

    def test_rhs_primal_zeros_no_return(self, api_small):
        """Primal RHS block is zero when rho=0."""
        _, rhs = api_small.kkt()
        np.testing.assert_array_equal(rhs[: api_small.n], 0.0)

    def test_rhs_dual_equals_b(self, api_small):
        """Dual RHS block equals b."""
        _, rhs = api_small.kkt()
        np.testing.assert_array_equal(rhs[api_small.n :], api_small.b)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS block equals rho * mu."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = API(X, rho=0.5, mu=mu).kkt()
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)

    def test_gamma_shifts_hessian(self):
        """Gamma adds 2*gamma*I to the Hessian block."""
        X = np.eye(3)  # noqa: N806
        K_base, _ = API(X).kkt()  # noqa: N806
        K_reg, _ = API(X, gamma=1.0).kkt()  # noqa: N806
        diff = K_reg[:3, :3] - K_base[:3, :3]
        np.testing.assert_allclose(diff, 2.0 * np.eye(3))

    def test_active_inequality_extends_system(self):
        """Pinning one inequality grows the KKT system by one row/col."""
        X = np.eye(3)  # noqa: N806
        api = API(X)
        K_base, _ = api.kkt()  # noqa: N806
        active = np.array([True, False, False])
        K_active, _ = api.kkt(active=active)  # noqa: N806
        assert K_active.shape == (K_base.shape[0] + 1, K_base.shape[1] + 1)


class TestAPIKktOperator:
    """Tests for API.kkt_operator."""

    def test_default_active_none(self, api_small):
        """kkt_operator() with no argument uses active=None (all-False mask)."""
        op, rhs = api_small.kkt_operator()
        assert op.shape == (api_small.n + 1, api_small.n + 1)
        assert rhs.shape == (api_small.n + 1,)

    def test_matvec_matches_explicit_kkt(self, api_small):
        """Matrix-free matvec agrees with the explicit KKT matrix."""
        op, _ = api_small.kkt_operator()
        K, _ = api_small.kkt()  # noqa: N806
        x = np.ones(api_small.n + 1)
        np.testing.assert_allclose(op @ x, K @ x, atol=1e-10)

    def test_rhs_matches_kkt_rhs(self, api_small):
        """RHS from kkt_operator matches RHS from kkt."""
        _, rhs_op = api_small.kkt_operator()
        _, rhs_kkt = api_small.kkt()
        np.testing.assert_allclose(rhs_op, rhs_kkt)

    def test_active_inequality_extends_system(self):
        """Pinning one inequality grows the operator size by one."""
        X = np.eye(3)  # noqa: N806
        api = API(X)
        op_base, _ = api.kkt_operator()
        active = np.array([True, False, False])
        op_active, _ = api.kkt_operator(active=active)
        assert op_active.shape == (op_base.shape[0] + 1, op_base.shape[1] + 1)

    def test_rhs_with_return_term(self):
        """With rho > 0 the primal RHS block equals rho * mu."""
        X = np.eye(3)  # noqa: N806
        mu = np.array([1.0, 2.0, 3.0])
        _, rhs = API(X, rho=0.5, mu=mu).kkt_operator()
        np.testing.assert_allclose(rhs[:3], 0.5 * mu)


class TestAPINullSpaceOperator:
    """Tests for API.null_space_operator."""

    def test_default_active_none(self, api_small):
        """null_space_operator() with no argument uses active=None (all-False mask)."""
        op, _rhs, w0, _P = api_small.null_space_operator()  # noqa: N806
        assert op is not None
        assert w0.shape == (api_small.n,)

    def test_particular_solution_satisfies_constraints(self, api_small):
        """w0 from null_space_operator satisfies A^T w0 = b."""
        _, _, w0, _ = api_small.null_space_operator()
        np.testing.assert_allclose(api_small.A.T @ w0, api_small.b, atol=1e-10)

    def test_null_space_basis_orthogonal_to_constraints(self, api_small):
        """P spans null(A^T): A^T P = 0."""
        _, _, _, P = api_small.null_space_operator()  # noqa: N806
        np.testing.assert_allclose(api_small.A.T @ P, 0.0, atol=1e-10)

    def test_fully_determined_returns_none_op(self):
        """Returns (None, None, w0, None) when constraints fully determine w."""
        X = np.eye(2)  # noqa: N806
        api = API(X)
        active = np.ones(api.C.shape[1], dtype=bool)
        op, rhs, w0, P = api.null_space_operator(active=active)  # noqa: N806
        assert op is None
        assert rhs is None
        assert P is None
        assert w0.shape == (api.n,)


class TestAPIConstraintActiveSet:
    """Tests for API.constraint_active_set."""

    def test_no_violations_single_call(self):
        """solve_fn is called exactly once when the first solution is feasible."""
        X = np.eye(3)  # noqa: N806
        api = API(X)
        calls = []

        def solve_fn(active):
            """Record calls and return a feasible solution."""
            calls.append(active.copy())
            return np.array([1 / 3, 1 / 3, 1 / 3]), 1

        api.constraint_active_set(solve_fn)
        assert len(calls) == 1

    def test_iters_accumulated(self):
        """Total iters is the sum of step_iters across all active-set calls."""
        X = np.eye(2)  # noqa: N806
        api = API(X)
        call_count = [0]

        def solve_fn(active):
            """Return a solution that violates one constraint on the first call."""
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([-0.1, 1.1]), 3  # violates w[0] >= 0
            return np.array([0.0, 1.0]), 2

        _, total = api.constraint_active_set(solve_fn)
        assert total == 5

    def test_active_mask_starts_empty(self):
        """The first call to solve_fn receives an all-False active mask."""
        X = np.eye(2)  # noqa: N806
        api = API(X)
        first_active = []

        def solve_fn(active):
            """Capture the first active mask and return a feasible solution."""
            if not first_active:
                first_active.append(active.copy())
            return np.array([0.5, 0.5]), 1

        api.constraint_active_set(solve_fn)
        assert not first_active[0].any()

    def test_violated_constraint_promoted(self):
        """A constraint violated in round 1 is active in round 2."""
        X = np.eye(2)  # noqa: N806
        # C.T @ w <= d  →  -I w <= 0  →  w >= 0 (long-only)
        api = API(X)
        active_masks = []

        def solve_fn(active):
            """Return a solution violating constraint 0 on the first call."""
            active_masks.append(active.copy())
            if len(active_masks) == 1:
                return np.array([-0.1, 1.1]), 1
            return np.array([0.0, 1.0]), 1

        api.constraint_active_set(solve_fn)
        assert len(active_masks) == 2
        # constraint 0 corresponds to -w[0] <= 0, i.e. w[0] >= 0
        assert active_masks[1][0]
