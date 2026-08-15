"""Thin conveniences for labeled containers (pandas, AnnData).

Optional layer: the core API accepts only ndarrays / scipy sparse and knows
nothing about these containers or any domain-specific conventions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .model import NBGLMSemiNMF

__all__ = ["fit_dataframe", "fit_anndata"]


def fit_dataframe(
    model: NBGLMSemiNMF,
    X: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on a features x samples DataFrame of counts.

    ``covariates`` is a samples x q DataFrame (categoricals one-hot encoded).
    Returns ``(loadings, usages)`` as labeled DataFrames; all fitted
    attributes remain on ``model``.
    """
    if covariates is not None and not isinstance(covariates, pd.DataFrame):
        raise TypeError("covariates must be a samples x q DataFrame")
    model.fit(X.to_numpy(), Z=covariates)
    cols = [f"factor_{j}" for j in range(model.F_.shape[1])]
    loadings = pd.DataFrame(model.F_, index=X.index, columns=cols)
    usages = pd.DataFrame(model.G_, index=X.columns, columns=cols)
    return loadings, usages


def fit_anndata(
    model: NBGLMSemiNMF,
    adata,
    layer: str | None = None,
    covariates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on an AnnData object (samples x features orientation, as is its
    convention; transposed internally). Requires the ``anndata`` extra.

    ``layer`` selects ``adata.layers[layer]`` (default ``adata.X``) — pass the
    layer holding raw integer counts. ``covariates`` names columns of
    ``adata.obs`` to use as the design matrix (categoricals one-hot encoded).
    Returns ``(loadings, usages)`` indexed by ``var_names`` / ``obs_names``.
    """
    mat = adata.layers[layer] if layer is not None else adata.X
    X = mat.T if not isinstance(mat, np.ndarray) else mat.T
    Z = adata.obs[list(covariates)] if covariates else None
    model.fit(X, Z=Z)
    cols = [f"factor_{j}" for j in range(model.F_.shape[1])]
    loadings = pd.DataFrame(model.F_, index=adata.var_names, columns=cols)
    usages = pd.DataFrame(model.G_, index=adata.obs_names, columns=cols)
    return loadings, usages
