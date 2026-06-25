"""Shrinkage intensity and target utilities (LW, OAS, RMT, constant-correlation)."""

import numpy as np
from sklearn.covariance import ledoit_wolf, oas


def lw_alpha_and_target(X: np.ndarray) -> tuple[float, np.ndarray]:  # noqa: N803
    """Return (alpha_lw, target) for LW scaled-identity shrinkage via sklearn.

    X must be column-demeaned.  The shrinkage target is bar_lambda * I where
    bar_lambda = ||X||_F^2 / (n * T) is the average per-asset variance
    (equivalently, the mean eigenvalue of the sample covariance X^T X / T).
    """
    _, alpha = ledoit_wolf(X, assume_centered=True)
    n = X.shape[1]
    T = X.shape[0]  # noqa: N806
    bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (n * T)
    cov = (X.transpose() @ X) / T
    bar_lam_cov = np.linalg.trace(cov) / n
    assert np.isclose(bar_lam, bar_lam_cov)  # noqa: S101

    delta2 = np.linalg.norm(cov - bar_lam * np.eye(n), "fro") ** 2
    norms_sq = np.sum(X**2, axis=1)
    beta2 = (np.sum(norms_sq**2) - 2 * np.sum(X * (X @ cov)) + T * np.linalg.norm(cov, "fro") ** 2) / T**2
    alpha_manual = min(1.0, beta2 / delta2)
    assert np.isclose(alpha_manual, alpha)  # noqa: S101

    return float(alpha), bar_lam * np.eye(n)


def lw_alpha_and_target_hard(X: np.ndarray, alpha: float = 0.5) -> tuple[float, np.ndarray]:  # noqa: N803
    """Return (alpha, target) for scaled-identity shrinkage with a fixed alpha.

    X must be column-demeaned.  The shrinkage target is bar_lambda * I where
    bar_lambda = ||X||_F^2 / (n * T) is the average per-asset variance
    (equivalently, the mean eigenvalue of the sample covariance X^T X / T).
    """
    # _, alpha = ledoit_wolf(X, assume_centered=True)
    n = X.shape[1]
    T = X.shape[0]  # noqa: N806
    bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (n * T)

    return alpha, bar_lam * np.eye(n)


def oas_alpha_and_target(X: np.ndarray) -> tuple[float, np.ndarray]:  # noqa: N803
    """Return (alpha_oas, target) for OAS scaled-identity shrinkage via sklearn.

    Uses the same bar_lambda * I target as LW but the Oracle Approximating
    Shrinkage formula (Chen et al. 2010), which has lower MSE when n/T is
    non-negligible.
    """
    _, alpha = oas(X, assume_centered=True)
    n = X.shape[1]
    bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (n * X.shape[0])
    return float(alpha), bar_lam * np.eye(n)


def cc_target(X: np.ndarray) -> tuple[np.ndarray, float]:  # noqa: N803
    """Constant-correlation shrinkage target (Ledoit-Wolf 2004 JoPM).

    T0_ij = rho_bar * sigma_i * sigma_j  (i != j)
    T0_ii = sigma_i^2

    where sigma_i = sqrt(Sigma_ii) is the per-asset sample standard deviation
    and rho_bar is the mean off-diagonal sample correlation coefficient.
    Always PSD for 0 < rho_bar < 1.

    Returns (target, rho_bar).
    """
    T, n = X.shape  # noqa: N806
    cov = (X.T @ X) / T
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    rho_bar = float(corr.sum()) / (n * (n - 1))
    target = rho_bar * np.outer(std, std)
    np.fill_diagonal(target, std**2)
    return target, rho_bar


def lw_alpha_for_target(X: np.ndarray, target: np.ndarray) -> float:  # noqa: N803
    """LW oracle alpha for an arbitrary SPD shrinkage target T0.

    alpha* = min(1, beta2 / delta2) where
      delta2 = ||S - T0||_F^2   (distance of sample cov from target)
      beta2  = (1/T^2) sum_t ||x_t x_t' - S||_F^2  (noise in S)
    """
    T, _n = X.shape  # noqa: N806
    cov = (X.T @ X) / T
    delta2 = float(np.linalg.norm(cov - target, "fro") ** 2)
    norms_sq = np.sum(X**2, axis=1)
    beta2 = float((np.sum(norms_sq**2) - 2 * np.sum(X * (X @ cov)) + T * np.linalg.norm(cov, "fro") ** 2) / T**2)
    return min(1.0, beta2 / delta2)


def rmt_target_and_alpha(
    X: np.ndarray,  # noqa: N803
) -> tuple[np.ndarray, tuple[float, np.ndarray, np.ndarray], int, float]:
    """RMT-clipped shrinkage target with alpha=1.

    Eigenvalues of the sample covariance above the Marchenko-Pastur bulk edge
    are kept as-is (signal); all others are clipped to bar_lambda (noise floor).
    The resulting target has lambda_min = bar_lambda (same as the scaled-identity
    target) so it provides equally effective lambda_min lifting at any alpha, while
    being 9x closer to the sample covariance in Frobenius norm than bar_lambda * I.

    T0 = bar_lambda * I + U_k @ diag(lambda_k - bar_lambda) @ U_k^T

    where (U_k, lambda_k) are the k eigenpairs of S = X^T X / T whose eigenvalues
    exceed the MP upper edge bar_lambda * (1 + sqrt(n/T))^2.

    Returns (target, lr_factors, k, 1.0).  alpha=1 means the system matrix is
    T0^RMT directly; the _kkt_step Woodbury path applies this in O(n_a k + k^3).
    """
    T, n = X.shape  # noqa: N806
    cov = (X.T @ X) / T
    bar_lam = np.trace(cov) / n
    mp_upper = bar_lam * (1.0 + np.sqrt(n / T)) ** 2

    eigs, vecs = np.linalg.eigh(cov)  # ascending order
    signal = eigs > mp_upper
    k = int(signal.sum())
    eigs_k = eigs[signal]
    vecs_k = vecs[:, signal]

    delta_k = eigs_k - bar_lam  # (k,) eigenvalue excesses
    target = bar_lam * np.eye(n) + vecs_k @ np.diag(delta_k) @ vecs_k.T
    lr_factors = (float(bar_lam), vecs_k, delta_k)  # for O(nk) matvec
    return target, lr_factors, k, 1.0
