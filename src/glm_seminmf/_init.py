"""Warm starts: truncated SVD, quick internal NMF, or random.

All run in numpy/scipy on CPU — initialization is a small fraction of fit
cost. Both data-driven inits work on ``Y = log1p(X / exp(b))``, the log1p of
exposure-normalized counts (spec section 3), which preserves sparsity.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds


def _normalized_log_counts(X, b: np.ndarray):
    """``log1p(X * exp(-b))`` per sample; sparse in, sparse out."""
    inv = np.exp(-b)
    if sp.issparse(X):
        Y = X.multiply(inv[None, :]).tocsc()
        Y.data = np.log1p(Y.data)
        return Y
    return np.log1p(X * inv[None, :])


def _svd_init(Y, k: int, rng: np.random.Generator):
    """Truncated SVD of row-centered Y without densifying: implicit centering."""
    p, n = Y.shape
    row_means = np.asarray(Y.mean(axis=1)).ravel()

    if sp.issparse(Y):
        ones_n = np.ones(n)

        def mv(v):
            return Y @ v - row_means * v.sum()

        def rmv(u):
            return Y.T @ u - ones_n * (row_means @ u)

        op = LinearOperator((p, n), matvec=mv, rmatvec=rmv)
        U, S, Vt = svds(op, k=min(k, min(p, n) - 1), v0=rng.standard_normal(min(p, n)))
    else:
        Yc = Y - row_means[:, None]
        U, S, Vt = svds(Yc, k=min(k, min(p, n) - 1), v0=rng.standard_normal(min(p, n)))
    order = np.argsort(S)[::-1]
    U, S, Vt = U[:, order], S[order], Vt[order]

    # Orient each component so the usage side carries more positive than
    # negative mass, then keep the positive part as the non-negative G start.
    VS = Vt.T * S[None, :]
    for j in range(VS.shape[1]):
        if np.linalg.norm(np.minimum(VS[:, j], 0)) > np.linalg.norm(np.maximum(VS[:, j], 0)):
            VS[:, j] *= -1.0
            U[:, j] *= -1.0
    F0 = np.hstack([U, rng.normal(0.0, 0.01, size=(p, k - U.shape[1]))]) if U.shape[1] < k else U
    G0 = np.maximum(VS, 0.0)
    if G0.shape[1] < k:
        G0 = np.hstack([G0, np.abs(rng.normal(0.0, 0.1, size=(Y.shape[1], k - G0.shape[1])))])
    return F0, G0


def _nmf_init(Y, k: int, rng: np.random.Generator, n_iter: int = 80):
    """Multiplicative-update NMF on Y; sparse-safe (Y never densified)."""
    p, n = Y.shape
    scale = np.sqrt(max(Y.mean() if not sp.issparse(Y) else Y.sum() / (p * n), 1e-8) / k)
    W = rng.uniform(1e-3, 1.0, size=(p, k)) * scale
    H = rng.uniform(1e-3, 1.0, size=(k, n)) * scale
    eps = 1e-10
    for _ in range(n_iter):
        W *= (Y @ H.T) / (W @ (H @ H.T) + eps)
        H *= (W.T @ Y) / ((W.T @ W) @ H + eps)
    return W, H.T


def initialize(X, b: np.ndarray, k: int, method, random_state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(F0, G0, a0)``; F0 is p x k signed, G0 is n x k non-negative.

    ``method`` is ``"svd"`` | ``"nmf"`` | ``"random"`` or a ``(F0, G0)`` tuple.
    """
    p, n = X.shape
    rng = np.random.default_rng(random_state)

    mean_norm = np.asarray((X @ np.exp(-b)) if sp.issparse(X) else X @ np.exp(-b)).ravel() / n
    a0 = np.log(mean_norm + 1e-8)

    if isinstance(method, tuple):
        F0, G0 = (np.asarray(m, dtype=np.float64) for m in method)
        if F0.shape != (p, k) or G0.shape != (n, k):
            raise ValueError(
                f"init tuple shapes {F0.shape}, {G0.shape} do not match (p={p}, k={k}), (n={n}, k={k})"
            )
        if G0.min() < 0:
            raise ValueError("init G0 must be non-negative")
        return F0, G0, a0

    if k == 0:
        return np.zeros((p, 0)), np.zeros((n, 0)), a0
    if method == "random":
        return rng.normal(0.0, 0.1, (p, k)), np.abs(rng.normal(0.0, 0.5, (n, k))), a0

    Y = _normalized_log_counts(X, b)
    if method == "svd":
        F0, G0 = _svd_init(Y, k, rng)
    elif method == "nmf":
        F0, G0 = _nmf_init(Y, k, rng)
    else:
        raise ValueError(f"unknown init method: {method!r}")
    return F0, G0, a0
