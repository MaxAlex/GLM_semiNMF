"""Warm starts: truncated SVD, quick internal NMF, or random.

All run in numpy/scipy on CPU — initialization is a small fraction of fit
cost. Both data-driven inits work on ``Y = log1p(X / exp(b))``, the log1p of
exposure-normalized counts (spec section 3), which preserves sparsity.

When covariates are supplied, Y is first regressed on (centered) Z and the
factor warm start is computed on the residual, with the OLS coefficients
returned as the warm start for gamma — otherwise the initial factors absorb
covariate structure and the fitted gamma never fully displaces it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds

# Below this many matrix elements Y is densified for initialization, which
# makes the sparse and dense input paths bitwise identical (spec test 11).
_DENSIFY_ELEMENTS = 50_000_000


def _normalized_log_counts(X, b: np.ndarray):
    """``log1p(X * exp(-b))`` per sample; sparse in, sparse out."""
    inv = np.exp(-b)
    if sp.issparse(X):
        Y = X.multiply(inv[None, :]).tocsc()
        Y.data = np.log1p(Y.data)
        if Y.shape[0] * Y.shape[1] <= _DENSIFY_ELEMENTS:
            return Y.toarray()
        return Y
    return np.log1p(X * inv[None, :])


def _svd_init(Y, k: int, rng: np.random.Generator, gamma0, Zc):
    """Truncated SVD of row-centered, covariate-residualized Y. Sparse Y is
    handled through an implicit LinearOperator (never densified)."""
    p, n = Y.shape
    row_means = np.asarray(Y.mean(axis=1)).ravel()
    kk = min(k, min(p, n) - 1)
    v0 = rng.standard_normal(min(p, n))

    if sp.issparse(Y):
        ones_n = np.ones(n)

        def mv(v):
            # svds/ARPACK does not guarantee 1-D matvec inputs — scipy's
            # LinearOperator wrapper passes through whatever shape ARPACK
            # hands it, including (n, 1). Without ravel, `Y @ v` stays
            # (p, 1) while `row_means * v.sum()` is (p,), and the subtraction
            # silently broadcasts to (p, p) instead of raising — caught by
            # scipy's own downstream reshape check, not here.
            v = np.ravel(v)
            out = np.ravel(Y @ v) - row_means * v.sum()
            if gamma0 is not None:
                out = out - gamma0 @ (Zc.T @ v)
            return out

        def rmv(u):
            u = np.ravel(u)
            out = np.ravel(Y.T @ u) - ones_n * (row_means @ u)
            if gamma0 is not None:
                out = out - Zc @ (gamma0.T @ u)
            return out

        U, S, Vt = svds(LinearOperator((p, n), matvec=mv, rmatvec=rmv), k=kk, v0=v0)
    else:
        Yc = Y - row_means[:, None]
        if gamma0 is not None:
            Yc = Yc - gamma0 @ Zc.T
        U, S, Vt = svds(Yc, k=kk, v0=v0)
    order = np.argsort(S)[::-1]
    U, S, Vt = U[:, order], S[order], Vt[order]

    # Orient each component so the usage side carries more positive than
    # negative mass, then keep the positive part as the non-negative G start.
    VS = Vt.T * S[None, :]
    for j in range(VS.shape[1]):
        if np.linalg.norm(np.minimum(VS[:, j], 0)) > np.linalg.norm(np.maximum(VS[:, j], 0)):
            VS[:, j] *= -1.0
            U[:, j] *= -1.0
    F0, G0 = U, np.maximum(VS, 0.0)
    if kk < k:  # pad if the matrix could not support k components
        F0 = np.hstack([F0, rng.normal(0.0, 0.01, size=(p, k - kk))])
        G0 = np.hstack([G0, np.abs(rng.normal(0.0, 0.1, size=(n, k - kk)))])
    return F0, G0


def _nmf_init(Y, k: int, rng: np.random.Generator, gamma0, Zc, n_iter: int = 80):
    """Multiplicative-update NMF; covariate-residualized only when dense."""
    if gamma0 is not None and not sp.issparse(Y):
        Y = np.maximum(Y - gamma0 @ Zc.T, 0.0)
    p, n = Y.shape
    mean = Y.sum() / (p * n) if sp.issparse(Y) else Y.mean()
    scale = np.sqrt(max(mean, 1e-8) / k)
    W = rng.uniform(1e-3, 1.0, size=(p, k)) * scale
    H = rng.uniform(1e-3, 1.0, size=(k, n)) * scale
    eps = 1e-10
    for _ in range(n_iter):
        W *= (Y @ H.T) / (W @ (H @ H.T) + eps)
        H *= (W.T @ Y) / ((W.T @ W) @ H + eps)
    return W, H.T


def initialize(
    X, b: np.ndarray, k: int, method, random_state, Z: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(F0, G0, a0, gamma0)``; F0 is p x k signed, G0 non-negative.

    ``method`` is ``"svd"`` | ``"nmf"`` | ``"random"`` or a ``(F0, G0)`` tuple.
    ``gamma0`` is the covariate warm start (None when Z is None).
    """
    p, n = X.shape
    rng = np.random.default_rng(random_state)

    mean_norm = np.asarray(X @ np.exp(-b)).ravel() / n
    a0 = np.log(mean_norm + 1e-8)

    if isinstance(method, tuple):
        F0, G0 = (np.asarray(m, dtype=np.float64) for m in method)
        if F0.shape != (p, k) or G0.shape != (n, k):
            raise ValueError(
                f"init tuple shapes {F0.shape}, {G0.shape} do not match (p={p}, k={k}), (n={n}, k={k})"
            )
        if G0.size and G0.min() < 0:
            raise ValueError("init G0 must be non-negative")
        return F0, G0, a0, None if Z is None else np.zeros((p, Z.shape[1]))

    Y = gamma0 = Zc = None
    if Z is not None:
        Y = _normalized_log_counts(X, b)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        ZtZ = Zc.T @ Zc + 1e-8 * np.eye(Z.shape[1])
        YZ = np.asarray(Y @ Zc)
        gamma0 = YZ @ np.linalg.inv(ZtZ)

    if k == 0:
        return np.zeros((p, 0)), np.zeros((n, 0)), a0, gamma0
    if method == "random":
        return rng.normal(0.0, 0.1, (p, k)), np.abs(rng.normal(0.0, 0.5, (n, k))), a0, gamma0

    if Y is None:
        Y = _normalized_log_counts(X, b)
    if method == "svd":
        F0, G0 = _svd_init(Y, k, rng, gamma0, Zc)
    elif method == "nmf":
        F0, G0 = _nmf_init(Y, k, rng, gamma0, Zc)
    else:
        raise ValueError(f"unknown init method: {method!r}")
    return F0, G0, a0, gamma0
