"""Tests for API.constraint_active_set."""

import numpy as np
import pytest

from fast_minimum_variance.api import API


def _api(C, d):  # noqa: N803
    """Build a minimal API with identity X and custom C, d."""
    n = C.shape[0]
    return API(np.eye(n), C=C, d=d)


class TestConstraintActiveSet:
    """Tests for API.constraint_active_set."""

    def test_first_call_receives_empty_active_mask(self):
        """solve_fn is called with all-False active mask on the first iteration."""
        api = _api(-np.eye(3), np.zeros(3))
        first_active = []

        def solve_fn(active):
            """Capture the first active mask and return a feasible solution."""
            if not first_active:
                first_active.append(active.copy())
            return np.array([0.5, 0.3, 0.2]), 1

        api.constraint_active_set(solve_fn)

        assert not first_active[0].any()
        assert first_active[0].shape == (3,)

    def test_returns_w_and_total_iters(self):
        """Returns (w, total_iters) tuple."""
        api = _api(-np.eye(2), np.zeros(2))
        w_expected = np.array([0.6, 0.4])

        result_w, result_iters = api.constraint_active_set(lambda _: (w_expected, 7))

        np.testing.assert_array_equal(result_w, w_expected)
        assert result_iters == 7

    def test_single_iteration_when_no_violations(self):
        """Loop exits after one call when the solution satisfies all constraints."""
        api = _api(-np.eye(3), np.zeros(3))
        call_count = [0]

        def solve_fn(active):
            """Count calls and return a feasible solution."""
            call_count[0] += 1
            return np.array([0.5, 0.3, 0.2]), 1

        api.constraint_active_set(solve_fn)

        assert call_count[0] == 1

    def test_promotes_violated_constraint(self):
        """Second call receives an active mask with the violated constraint promoted."""
        api = _api(-np.eye(3), np.zeros(3))
        w1 = np.array([0.7, -0.1, 0.4])  # constraint 1 violated (w[1] < 0)
        w2 = np.array([0.6, 0.0, 0.4])
        responses = iter([(w1, 2), (w2, 3)])
        masks = []

        def solve_fn(active):
            """Record active masks and return preset solutions."""
            masks.append(active.copy())
            return next(responses)

        api.constraint_active_set(solve_fn)

        assert not masks[1][0]
        assert masks[1][1]
        assert not masks[1][2]

    def test_accumulates_iters_across_steps(self):
        """total_iters sums step_iters from all active-set iterations."""
        api = _api(-np.eye(2), np.zeros(2))
        w1 = np.array([0.8, -0.2])
        w2 = np.array([0.6, 0.4])
        responses = iter([(w1, 10), (w2, 5)])

        _, total_iters = api.constraint_active_set(lambda _: next(responses))

        assert total_iters == 15

    def test_multiple_violations_promoted_simultaneously(self):
        """All violated constraints are promoted in a single outer iteration."""
        api = _api(-np.eye(4), np.zeros(4))
        w1 = np.array([0.5, -0.1, -0.2, 0.8])  # constraints 1 and 2 violated
        w2 = np.array([0.5, 0.0, 0.0, 0.5])
        responses = iter([(w1, 1), (w2, 1)])
        masks = []

        def solve_fn(active):
            """Record active masks and return preset solutions."""
            masks.append(active.copy())
            return next(responses)

        api.constraint_active_set(solve_fn)

        assert not masks[1][0]
        assert masks[1][1]
        assert masks[1][2]
        assert not masks[1][3]

    def test_general_inequality_constraint(self):
        """Works with non-identity C: upper bound on first weight."""
        C = np.array([[1.0], [0.0]])  # noqa: N806  # w[0] <= 0.3
        d = np.array([0.3])
        api = _api(C, d)
        w1 = np.array([0.5, 0.5])  # violates w[0] <= 0.3
        w2 = np.array([0.3, 0.7])
        responses = iter([(w1, 1), (w2, 1)])
        call_count = [0]

        def solve_fn(active):
            """Count calls and return preset solutions."""
            call_count[0] += 1
            return next(responses)

        result_w, _ = api.constraint_active_set(solve_fn)

        np.testing.assert_array_equal(result_w, w2)
        assert call_count[0] == 2

    def test_active_mask_grows_monotonically(self):
        """Once promoted, a constraint stays active in subsequent iterations."""
        api = _api(-np.eye(3), np.zeros(3))
        w1 = np.array([0.7, -0.1, 0.4])  # constraint 1 violated
        w2 = np.array([0.8, 0.0, -0.1])  # constraint 2 violated, 1 still pinned
        w3 = np.array([0.6, 0.0, 0.4])
        responses = iter([(w1, 1), (w2, 1), (w3, 1)])
        captured = []

        def solve_fn(active):
            """Capture active masks and return preset solutions."""
            captured.append(active.copy())
            return next(responses)

        api.constraint_active_set(solve_fn)

        assert captured[1][1]
        assert not captured[1][2]
        assert captured[2][1]
        assert captured[2][2]

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_default_long_only_no_violations(self, n):
        """Equal-weight portfolio satisfies the default long-only constraints."""
        api = API(np.eye(n))
        w_eq = np.ones(n) / n
        call_count = [0]

        def solve_fn(active):
            """Return the equal-weight portfolio."""
            call_count[0] += 1
            return w_eq, 1

        api.constraint_active_set(solve_fn)
        assert call_count[0] == 1
