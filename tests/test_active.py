"""Tests for fast_minimum_variance.active."""

import numpy as np

from fast_minimum_variance.active import constraint_active_set


class TestConstraintActiveSet:
    """Tests for constraint_active_set."""

    def test_first_call_receives_empty_active_mask(self, mocker):
        """solve_fn is called with all-False active mask on the first iteration."""
        C = -np.eye(3)  # noqa: N806
        d = np.zeros(3)
        solve_fn = mocker.Mock(return_value=(np.array([0.5, 0.3, 0.2]), 1))

        constraint_active_set(C, d, solve_fn)

        active = solve_fn.call_args_list[0][0][0]
        assert not active.any()
        assert active.shape == (3,)

    def test_returns_w_and_total_iters(self, mocker):
        """Returns (w, total_iters) tuple."""
        C = -np.eye(2)  # noqa: N806
        d = np.zeros(2)
        w = np.array([0.6, 0.4])
        solve_fn = mocker.Mock(return_value=(w, 7))

        result_w, result_iters = constraint_active_set(C, d, solve_fn)

        np.testing.assert_array_equal(result_w, w)
        assert result_iters == 7

    def test_single_iteration_when_no_violations(self, mocker):
        """Loop exits after one call when the unconstrained solution satisfies all constraints."""
        C = -np.eye(3)  # noqa: N806
        d = np.zeros(3)
        solve_fn = mocker.Mock(return_value=(np.array([0.5, 0.3, 0.2]), 1))

        constraint_active_set(C, d, solve_fn)

        assert solve_fn.call_count == 1

    def test_promotes_violated_constraint(self, mocker):
        """Second call receives an active mask with the violated constraint promoted."""
        C = -np.eye(3)  # noqa: N806
        d = np.zeros(3)
        w1 = np.array([0.7, -0.1, 0.4])  # constraint 1 violated (w[1] < 0)
        w2 = np.array([0.6, 0.0, 0.4])  # all satisfied

        solve_fn = mocker.Mock(side_effect=[(w1, 2), (w2, 3)])
        constraint_active_set(C, d, solve_fn)

        active = solve_fn.call_args_list[1][0][0]
        assert not active[0]
        assert active[1]
        assert not active[2]

    def test_accumulates_iters_across_steps(self, mocker):
        """total_iters sums step_iters from all active-set iterations."""
        C = -np.eye(2)  # noqa: N806
        d = np.zeros(2)
        w1 = np.array([0.8, -0.2])
        w2 = np.array([0.6, 0.4])

        solve_fn = mocker.Mock(side_effect=[(w1, 10), (w2, 5)])
        _, total_iters = constraint_active_set(C, d, solve_fn)

        assert total_iters == 15

    def test_multiple_violations_promoted_simultaneously(self, mocker):
        """All violated constraints are promoted in a single outer iteration."""
        C = -np.eye(4)  # noqa: N806
        d = np.zeros(4)
        w1 = np.array([0.5, -0.1, -0.2, 0.8])  # constraints 1 and 2 violated
        w2 = np.array([0.5, 0.0, 0.0, 0.5])

        solve_fn = mocker.Mock(side_effect=[(w1, 1), (w2, 1)])
        constraint_active_set(C, d, solve_fn)

        active = solve_fn.call_args_list[1][0][0]
        assert not active[0]
        assert active[1]
        assert active[2]
        assert not active[3]

    def test_general_inequality_constraint(self, mocker):
        """Works with non-identity C: upper bound on first weight."""
        C = np.array([[1.0], [0.0]])  # noqa: N806  # C.T @ w <= d means w[0] <= 0.3
        d = np.array([0.3])
        w1 = np.array([0.5, 0.5])  # violates w[0] <= 0.3
        w2 = np.array([0.3, 0.7])  # satisfies

        solve_fn = mocker.Mock(side_effect=[(w1, 1), (w2, 1)])
        result_w, _ = constraint_active_set(C, d, solve_fn)

        np.testing.assert_array_equal(result_w, w2)
        assert solve_fn.call_count == 2

    def test_active_mask_grows_monotonically(self):
        """Once promoted, a constraint stays active in subsequent iterations."""
        C = -np.eye(3)  # noqa: N806
        d = np.zeros(3)
        w1 = np.array([0.7, -0.1, 0.4])  # constraint 1 violated
        w2 = np.array([0.8, 0.0, -0.1])  # constraint 2 violated (1 still pinned)
        w3 = np.array([0.6, 0.0, 0.4])  # all satisfied

        # Capture copies at call time — the active array is mutated in-place
        # so call_args_list entries would all reflect the final state.
        captured = []
        responses = iter([(w1, 1), (w2, 1), (w3, 1)])

        def side_effect(active):
            captured.append(active.copy())
            return next(responses)

        constraint_active_set(C, d, side_effect)

        assert captured[1][1]
        assert not captured[1][2]
        assert captured[2][1]
        assert captured[2][2]
