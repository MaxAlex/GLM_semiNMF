"""Input validation and covariate encoding.

The core accepts ``X`` as scipy CSR/CSC or dense ndarray, oriented
features x samples, non-negative counts. ``Z`` is samples x covariates;
a pandas DataFrame is one-hot encoded with a dropped reference level per
categorical column.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Sparse inputs at most this many elements are densified up front: the fitting
# engine would hold them dense on the device anyway, and a single code path
# makes sparse and dense inputs bitwise identical (spec test 11). Larger
# sparse inputs stay sparse and may differ from dense in the last float digits.
_DENSIFY_ELEMENTS = 50_000_000

# A non-integer matrix whose largest entry is below this is almost certainly normalized
# (log1p-CPM tops out around 10-12); real count data, ambient-corrected or not, goes far
# higher. Only used to reject obviously-normalized input — see validate_X.
_NORMALIZED_MAX = 30.0


def validate_X(X) -> tuple[sp.csc_matrix | np.ndarray, int, int]:
    """Validate counts and return ``(X, p, n)`` with sparse input as CSC
    (small sparse inputs are densified — see ``_DENSIFY_ELEMENTS``).

    CSC because fitting streams over contiguous sample (column) chunks.

    **Non-integer counts are accepted.** The NB2 likelihood is defined for any real
    ``x >= 0``: the only ``x``-dependent gamma term is ``lgamma(x + 1)``, the continuous
    extension of ``log(x!)``, and the deviance uses ``xlogy(x, x)``, which is likewise
    continuous and zero at ``x = 0``. Nothing in the fitter counts events. This matters for
    ambient-corrected input — CellBender emits non-integer posterior means, and rounding
    them would discard the correction's precision at low expression, where it does the most
    work.

    Negatives still raise: they are outside the NB support and are almost always a sign that
    something has been centred or residualised upstream.

    Values that look *normalized* rather than merely fractional still raise, because the
    model does its own normalization through the exposure term and silently fitting
    log1p-CPM would be a quiet, hard-to-notice error. The test is deliberately narrow — a
    non-integer matrix whose maximum is below ``_NORMALIZED_MAX`` — so ambient-corrected
    counts, which keep count-scale magnitudes, pass.
    """
    if sp.issparse(X):
        X = X.tocsc()
        data = X.data
    elif isinstance(X, np.ndarray):
        data = X
    else:
        raise TypeError(
            f"X must be a scipy sparse matrix or numpy ndarray, got {type(X).__name__}. "
            "For labeled containers use glm_seminmf.adapters."
        )
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (features x samples), got shape {X.shape}")
    if data.size and data.min() < 0:
        raise ValueError("X must contain non-negative counts")
    if data.size and not np.issubdtype(data.dtype, np.integer):
        if np.any(data != np.floor(data)) and data.max() < _NORMALIZED_MAX:
            raise ValueError(
                f"X looks normalized: non-integer values with a maximum of {data.max():.3g}, "
                f"below the {_NORMALIZED_MAX} threshold for count-scale data. The model does "
                "its own normalization via the exposure term — pass counts, not log1p/CPM. "
                "Non-integer counts themselves are fine (e.g. CellBender output); it is the "
                "small dynamic range that looks wrong here."
            )
    p, n = X.shape
    if sp.issparse(X) and p * n <= _DENSIFY_ELEMENTS:
        # C order: CSC.toarray() yields F-order, whose different BLAS
        # accumulation order would break bitwise sparse/dense agreement.
        X = np.ascontiguousarray(X.toarray())
    return X, p, n


def encode_Z(Z, n: int, columns: list[str] | None = None) -> tuple[np.ndarray | None, list[str]]:
    """Return ``(Z_array, column_names)`` with categoricals one-hot encoded.

    Accepts an ndarray (used as-is) or a pandas DataFrame whose object /
    categorical / boolean columns are expanded to indicators with the first
    level dropped as reference. Pass the ``columns`` recorded at fit time to
    reproduce the same encoding for new samples (missing levels become 0).
    """
    if Z is None:
        if columns:
            raise ValueError("model was fitted with covariates; Z is required")
        return None, []
    try:
        import pandas as pd

        is_df = isinstance(Z, pd.DataFrame)
    except ImportError:  # pragma: no cover
        is_df = False
    if is_df:
        Z = pd.get_dummies(Z, drop_first=True, dtype=float)
        if columns is not None:
            Z = Z.reindex(columns=columns, fill_value=0.0)
        names = [str(c) for c in Z.columns]
        Z = Z.to_numpy(dtype=np.float64)
    else:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        names = columns if columns is not None else [f"z{j}" for j in range(Z.shape[1])]
        if columns is not None and Z.shape[1] != len(columns):
            raise ValueError(f"Z has {Z.shape[1]} columns; model was fitted with {len(columns)}")
    if Z.shape[0] != n:
        raise ValueError(f"Z has {Z.shape[0]} rows but X has {n} samples")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains non-finite values")
    return Z, names
