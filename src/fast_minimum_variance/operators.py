"""Matrix-free linear-operator layer for the minimum-variance solver.

Builds the active-set system operator ``Sigma`` as a composition of cvx-linalg
operators and runs conjugate gradients over it restricted to the active set,
without ever forming ``Sigma_a`` explicitly. Kept separate from the solver so the
linear-algebra concern is independently testable and the solver class stays
focused on the primal-dual loop.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from cvx.linalg import DenseOperator, FactorOperator, GramOperator, SumOperator
from scipy.sparse.linalg import LinearOperator, cg


def build_system_operator(
    x: np.ndarray,
    alpha: float,
    target: np.ndarray | None,
    target_lr: tuple[float, np.ndarray, np.ndarray] | None,
    t: int,
) -> SumOperator:
    """Build ``Sigma = (1-alpha)/T * X^T X + alpha * T0`` as a cvx-linalg operator.

    A :class:`~cvx.linalg.SumOperator` of the data Gram term and, when present,
    the target term (a :class:`~cvx.linalg.FactorOperator` for a low-rank RMT
    target, else a :class:`~cvx.linalg.DenseOperator`). The full-universe
    operators are sliced to the active set later; nothing is formed at ``n x n``.
    Without a target the data term carries the full weight.
    """
    has_target = target_lr is not None or target is not None
    c_data = (1.0 - alpha) if has_target else 1.0
    terms: list[tuple[float, Any]] = [(c_data / t, GramOperator(x))]
    if target_lr is not None:
        bar_lam, u_k, delta_k = target_lr
        terms.append((alpha, FactorOperator(np.full(u_k.shape[0], bar_lam), u_k, np.diag(delta_k))))
    elif target is not None:
        terms.append((alpha, DenseOperator(target)))
    return SumOperator(terms)


def restricted_matvec(sigma: SumOperator, active_idx: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
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


def cg_solve_reduced(
    sigma: SumOperator,
    active: np.ndarray,
    b_a: np.ndarray,
    mu_active: np.ndarray | None,
    p: int,
    x0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Run matrix-free CG on the reduced SPD system; return ``(V_eq, v_mu, iters)``.

    Solves ``Sigma_a V = B_a^T`` (``p`` right-hand sides) and, when ``mu_active``
    is given, ``Sigma_a v_mu = mu_a``, restricting ``sigma`` to the active set
    once so each matvec is ``O(n_a T)`` rather than ``O(n T)`` and ``Sigma_a`` is
    never formed explicitly. ``iters`` is the total CG matvec count.
    """
    n_a = int(active.sum())
    active_idx = np.flatnonzero(active)
    free_matvec = restricted_matvec(sigma, active_idx)
    count = [0]

    def matvec(v: np.ndarray) -> np.ndarray:
        """Apply Sigma_a to v via the pre-sliced free-block operator."""
        count[0] += 1
        return free_matvec(v)

    op = LinearOperator((n_a, n_a), matvec=matvec, dtype=np.float64)  # ty:ignore[missing-argument, parameter-already-assigned, unknown-argument]
    # x0 approximates the final w, which is proportional to the single solve
    # column only in the budget case; skip the guess for p > 1.
    guess = x0 if p == 1 else None
    v_eq = np.column_stack([cg(op, b_a[j], x0=guess)[0] for j in range(p)])
    v_mu = cg(op, mu_active, x0=guess)[0] if mu_active is not None else None
    return v_eq, v_mu, count[0]
