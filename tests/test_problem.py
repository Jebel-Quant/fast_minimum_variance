"""Tests for fast_minimum_variance.problem (Problem class)."""

import numpy as np
import pytest

from fast_minimum_variance.problem import _Problem as Problem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_returns(T, N, seed=42):  # noqa: N803
    """Generate a T x N matrix of i.i.d. standard normal returns."""
    return np.random.default_rng(seed).standard_normal((T, N))


@pytest.fixture(scope="session")
def X():  # noqa: N802
    """Return matrix of shape (200, 10) with a fixed seed."""
    return make_returns(T=200, N=10, seed=42)


@pytest.fixture(scope="session")
def X_small():  # noqa: N802
    """Return matrix of shape (100, 3) for fast solver tests."""
    return make_returns(T=100, N=3, seed=0)


@pytest.fixture(scope="session")
def problem(X):  # noqa: N803
    """Problem instance wrapping the session-scoped return matrix."""
    return Problem(X)


@pytest.fixture(scope="session")
def problem_small(X_small):  # noqa: N803
    """Problem instance wrapping the small session-scoped return matrix."""
    return Problem(X_small)


# ---------------------------------------------------------------------------
# TestProblemDefaults
# ---------------------------------------------------------------------------


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

    def test_alpha_default(self, problem):
        """Default alpha is 0.0."""
        assert problem.alpha == 0.0


# ---------------------------------------------------------------------------
# TestProblemCustomConstraints
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestProblemKkt
# ---------------------------------------------------------------------------


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

    def test_alpha_shifts_hessian(self):
        """Alpha adds a ridge term to the Hessian block."""
        X2 = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # noqa: N806
        K_base, _ = Problem(X2)._kkt()  # noqa: N806
        K_reg, _ = Problem(X2, alpha=0.5)._kkt()  # noqa: N806
        ridge = 0.5 * np.einsum("ti,ti->", X2, X2) / X2.shape[1]
        expected_diff = 2 * (0.5 * (X2.T @ X2) - X2.T @ X2) + 2 * ridge * np.eye(X2.shape[1])
        diff = K_reg[:2, :2] - K_base[:2, :2]
        np.testing.assert_allclose(diff, expected_diff, atol=1e-12)

    def test_active_inequality_extends_system(self):
        """Pinning one inequality grows the KKT system by one row/col."""
        X = np.eye(3)  # noqa: N806
        p = Problem(X)
        K_base, _ = p._kkt()  # noqa: N806
        active = np.array([True, False, False])
        K_active, _ = p._kkt(active=active)  # noqa: N806
        assert K_active.shape == (K_base.shape[0] + 1, K_base.shape[1] + 1)


# ---------------------------------------------------------------------------
# TestProblemKktOperator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestProblemConstraintActiveSet
# ---------------------------------------------------------------------------


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
                return np.array([-0.1, 1.1]), 3
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
        assert active_masks[1][0]

    def test_first_call_receives_empty_active_mask(self):
        """solve_fn is called with all-False active mask on the first iteration."""
        p = Problem(np.eye(3), C=-np.eye(3), d=np.zeros(3))
        first_active = []

        def solve_fn(active):
            """Capture the first active mask and return a feasible solution."""
            if not first_active:
                first_active.append(active.copy())
            return np.array([0.5, 0.3, 0.2]), 1

        p._constraint_active_set(solve_fn)
        assert not first_active[0].any()
        assert first_active[0].shape == (3,)

    def test_returns_w_and_total_iters(self):
        """Returns (w, total_iters) tuple."""
        p = Problem(np.eye(2), C=-np.eye(2), d=np.zeros(2))
        w_expected = np.array([0.6, 0.4])
        result_w, result_iters = p._constraint_active_set(lambda _: (w_expected, 7))
        np.testing.assert_array_equal(result_w, w_expected)
        assert result_iters == 7

    def test_single_iteration_when_no_violations(self):
        """Loop exits after one call when the solution satisfies all constraints."""
        p = Problem(np.eye(3), C=-np.eye(3), d=np.zeros(3))
        call_count = [0]

        def solve_fn(active):
            """Count calls and return a feasible solution."""
            call_count[0] += 1
            return np.array([0.5, 0.3, 0.2]), 1

        p._constraint_active_set(solve_fn)
        assert call_count[0] == 1

    def test_promotes_violated_constraint(self):
        """Second call receives an active mask with the violated constraint promoted."""
        p = Problem(np.eye(3), C=-np.eye(3), d=np.zeros(3))
        w1 = np.array([0.7, -0.1, 0.4])
        w2 = np.array([0.6, 0.0, 0.4])
        responses = iter([(w1, 2), (w2, 3)])
        masks = []

        def solve_fn(active):
            """Record active masks and return preset solutions."""
            masks.append(active.copy())
            return next(responses)

        p._constraint_active_set(solve_fn)
        assert not masks[1][0]
        assert masks[1][1]
        assert not masks[1][2]

    def test_accumulates_iters_across_steps(self):
        """total_iters sums step_iters from all active-set iterations."""
        p = Problem(np.eye(2), C=-np.eye(2), d=np.zeros(2))
        w1 = np.array([0.8, -0.2])
        w2 = np.array([0.6, 0.4])
        responses = iter([(w1, 10), (w2, 5)])
        _, total_iters = p._constraint_active_set(lambda _: next(responses))
        assert total_iters == 15

    def test_multiple_violations_promoted_simultaneously(self):
        """All violated constraints are promoted in a single outer iteration."""
        p = Problem(np.eye(4), C=-np.eye(4), d=np.zeros(4))
        w1 = np.array([0.5, -0.1, -0.2, 0.8])
        w2 = np.array([0.5, 0.0, 0.0, 0.5])
        responses = iter([(w1, 1), (w2, 1)])
        masks = []

        def solve_fn(active):
            """Record active masks and return preset solutions."""
            masks.append(active.copy())
            return next(responses)

        p._constraint_active_set(solve_fn)
        assert not masks[1][0]
        assert masks[1][1]
        assert masks[1][2]
        assert not masks[1][3]

    def test_general_inequality_constraint(self):
        """Works with non-identity C: upper bound on first weight."""
        C = np.array([[1.0], [0.0]])  # noqa: N806
        d = np.array([0.3])
        p = Problem(np.eye(2), C=C, d=d)
        w1 = np.array([0.5, 0.5])
        w2 = np.array([0.3, 0.7])
        responses = iter([(w1, 1), (w2, 1)])
        call_count = [0]

        def solve_fn(active):
            """Count calls and return preset solutions."""
            call_count[0] += 1
            return next(responses)

        result_w, _ = p._constraint_active_set(solve_fn)
        np.testing.assert_array_equal(result_w, w2)
        assert call_count[0] == 2

    def test_active_mask_grows_monotonically(self):
        """Once promoted, a constraint stays active in subsequent iterations."""
        p = Problem(np.eye(3), C=-np.eye(3), d=np.zeros(3))
        w1 = np.array([0.7, -0.1, 0.4])
        w2 = np.array([0.8, 0.0, -0.1])
        w3 = np.array([0.6, 0.0, 0.4])
        responses = iter([(w1, 1), (w2, 1), (w3, 1)])
        captured = []

        def solve_fn(active):
            """Capture active masks and return preset solutions."""
            captured.append(active.copy())
            return next(responses)

        p._constraint_active_set(solve_fn)
        assert captured[1][1]
        assert not captured[1][2]
        assert captured[2][1]
        assert captured[2][2]

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_default_long_only_no_violations(self, n):
        """Equal-weight portfolio satisfies the default long-only constraints."""
        p = Problem(np.eye(n))
        w_eq = np.ones(n) / n
        call_count = [0]

        def solve_fn(active):
            """Return the equal-weight portfolio."""
            call_count[0] += 1
            return w_eq, 1

        p._constraint_active_set(solve_fn)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TestSolveKkt
# ---------------------------------------------------------------------------


class TestSolveKkt:
    """Tests for Problem.solve_kkt."""

    def test_iters(self, problem):
        """Direct KKT solver always returns iters==1."""
        _, iters = problem.solve_kkt()
        assert iters == 1

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_kkt()
        assert w.shape == (problem.n,)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_kkt()
        assert np.all(w >= -1e-10)

    def test_trivial_case(self):
        """With a single asset the full weight goes to it."""
        R = np.ones((10, 1))  # noqa: N806
        w, _ = Problem(X=R).solve_kkt()
        np.testing.assert_allclose(w, [1.0], atol=1e-10)

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        R = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(R).solve_kkt()
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= -1e-10)

    def test_return_term_tilts_weights(self):
        """With rho > 0 the solution tilts toward the highest-return asset."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))  # noqa: N806
        mu = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        w_mv, _ = Problem(X).solve_kkt()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_kkt()
        assert w_mk[4] > w_mv[4]

    def test_all_inequalities_active_break(self):
        """Active-set loop terminates when every inequality constraint is binding."""
        X = np.eye(2)  # noqa: N806
        C = np.array([[1.0], [0.0]])  # noqa: N806
        d = np.array([0.3])
        w, _ = Problem(X, C=C, d=d).solve_kkt()
        np.testing.assert_allclose(w[0], 0.3, atol=1e-8)
        np.testing.assert_allclose(w[1], 0.7, atol=1e-8)

    def test_active_set_drops_asset(self):
        """Active-set iteration drops a high-variance correlated asset."""
        X = np.array(  # noqa: N806
            [
                [0.1, 0.0, 5.0],
                [-0.1, 0.0, -5.0],
                [0.0, 0.1, 0.1],
                [0.0, -0.1, -0.1],
            ]
        )
        w, _ = Problem(X).solve_kkt()
        assert w[2] == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(w[:2], [0.5, 0.5], atol=1e-8)


# ---------------------------------------------------------------------------
# TestSolveCvxpy
# ---------------------------------------------------------------------------


class TestSolveCvxpy:
    """Tests for Problem.solve_cvxpy."""

    def test_shape(self, problem):
        """Output weight vector has shape (N,)."""
        w, _ = problem.solve_cvxpy()
        assert w.shape == (problem.X.shape[1],)

    def test_weights_sum_to_one(self, problem):
        """Weights sum to 1."""
        w, _ = problem.solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, problem):
        """All weights are non-negative."""
        w, _ = problem.solve_cvxpy()
        assert np.all(w >= -1e-6)

    def test_close_to_kkt(self, problem_small):
        """CVXPY solution is close to the exact KKT solution."""
        w_kkt, _ = problem_small.solve_kkt()
        w_cvxpy, _ = problem_small.solve_cvxpy()
        np.testing.assert_allclose(w_cvxpy, w_kkt, atol=1e-4)

    def test_objective_no_worse_than_equal_weight(self, X, problem):  # noqa: N803
        """Optimal portfolio has variance <= equal-weight portfolio."""
        w_opt, _ = problem.solve_cvxpy()
        N = X.shape[1]  # noqa: N806
        w_eq = np.ones(N) / N
        assert np.linalg.norm(X @ w_opt) <= np.linalg.norm(X @ w_eq) + 1e-6

    def test_return_term_tilts_weights(self, X, problem):  # noqa: N803
        """With rho > 0 the solution tilts toward the highest-return asset."""
        mu = np.zeros(X.shape[1])
        mu[-1] = 1.0
        w_mv, _ = problem.solve_cvxpy()
        w_mk, _ = Problem(X, rho=1.0, mu=mu).solve_cvxpy()
        assert w_mk[-1] > w_mv[-1]

    @pytest.mark.parametrize("N", [2, 5, 15])
    def test_various_sizes(self, N):  # noqa: N803
        """Solver works across a range of asset counts."""
        rng = np.random.default_rng(N)
        X = rng.standard_normal((200, N))  # noqa: N806
        w, _ = Problem(X).solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)

    def test_alpha_branch(self):
        """Alpha != 0 enters the Ledoit-Wolf ridge regularisation branch."""
        X = np.eye(3)  # noqa: N806
        w, _ = Problem(X, alpha=0.1).solve_cvxpy()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_import_error_without_cvxpy(self, monkeypatch):
        """solve_cvxpy raises ImportError when cvxpy is unavailable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("no cvxpy")  # noqa: TRY003
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="cvxpy is required"):
            Problem(np.eye(3)).solve_cvxpy()

    def test_runtime_error_on_infeasible(self):
        """solve_cvxpy raises RuntimeError when the problem is infeasible."""
        X = np.eye(2)  # noqa: N806
        C = np.eye(2)  # noqa: N806
        d = np.array([-1.0, -1.0])
        with pytest.raises(RuntimeError, match="CVXPY solver failed"):
            Problem(X, C=C, d=d).solve_cvxpy()
