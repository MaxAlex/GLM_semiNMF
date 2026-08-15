"""Input validation and covariate encoding.

The core accepts ``X`` as scipy CSR/CSC or dense ndarray, oriented
features x samples, raw integer counts only. ``Z`` is samples x covariates;
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


def validate_X(X) -> tuple[sp.csc_matrix | np.ndarray, int, int]:
    """Validate counts and return ``(X, p, n)`` with sparse input as CSC
    (small sparse inputs are densified — see ``_DENSIFY_ELEMENTS``).

    CSC because fitting streams over contiguous sample (column) chunks.
    Raises on non-integer or negative values instead of silently rounding.
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
    if data.size and not np.issubdtype(data.dtype, np.integer):
        if np.any(data != np.floor(data)):
            raise ValueError(
                "X must contain raw integer counts; found non-integer values. "
                "The model does its own normalization via the exposure term — "
                "pass unnormalized counts."
            )
    if data.size and data.min() < 0:
        raise ValueError("X must contain non-negative counts")
    p, n = X.shape
    if sp.issparse(X) and p * n <= _DENSIFY_ELEMENTS:
        X = X.toarray()
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
