"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(scope="session")
def resource_dir() -> Path:
    """Return the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def reference_weights() -> Callable[[object], np.ndarray]:
    """Return an independent long-only min-var oracle (SLSQP), for cross-validation.

    Solves the same objective as ``_MinVarProblem`` — ``(1-alpha)||Xw||^2/T +
    alpha*w^T T0 w - rho*mu^T w`` subject to ``Bw = c`` (or the budget) and
    ``w >= 0`` — with SciPy's SLSQP, sharing no code with the library's
    active-set solvers. It replaces the former CVXPY reference and agrees with
    the direct KKT solve to ~1e-7 on the covered cases.
    """

    def _reference(prob: object) -> np.ndarray:
        x = prob.X  # ty:ignore[unresolved-attribute]
        t, n = x.shape
        alpha = prob.alpha  # ty:ignore[unresolved-attribute]

        if prob.target_lr is not None:  # ty:ignore[unresolved-attribute]
            bar_lam, u_k, delta_k = prob.target_lr  # ty:ignore[unresolved-attribute]

            def target_quad(w: np.ndarray) -> float:
                return float(w @ (bar_lam * w + u_k @ (delta_k * (u_k.T @ w))))

            has_target = True
        elif prob.target is not None:  # ty:ignore[unresolved-attribute]
            target = prob.target  # ty:ignore[unresolved-attribute]

            def target_quad(w: np.ndarray) -> float:
                return float(w @ (target @ w))

            has_target = True
        else:
            has_target = False

        rho = prob.rho  # ty:ignore[unresolved-attribute]
        mu = prob.mu  # ty:ignore[unresolved-attribute]

        def objective(w: np.ndarray) -> float:
            data = float((x @ w) @ (x @ w)) / t
            value = (1.0 - alpha) * data + alpha * target_quad(w) if has_target else data
            if rho != 0.0 and mu is not None:
                value = value - rho * float(mu @ w)
            return value

        if prob.B is not None:  # ty:ignore[unresolved-attribute]
            b_mat, c_vec = prob.B, prob.c  # ty:ignore[unresolved-attribute]
            constraints = [
                {"type": "eq", "fun": (lambda w, i=i: float(b_mat[i] @ w - c_vec[i]))} for i in range(b_mat.shape[0])
            ]
        else:
            constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

        res = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result: np.ndarray = res.x
        return result

    return _reference
