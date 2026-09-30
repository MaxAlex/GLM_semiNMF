"""Model-independent NB scoring for benchmark comparisons.

Fit the calibration null once on training counts, then freeze its theta and
mean parameters across candidates. This is separate from each model's public
deviance_explained_, which deliberately uses its own fitted dispersion.
"""
import copy

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from glm_seminmf import NBGLMSemiNMF


def fit_common_null(X, exposure, Z=None, *, device='cpu', max_iter=300, max_seconds=30):
    return NBGLMSemiNMF(0, dispersion='shared', l1_G=0., exposure=exposure,
                       max_iter=max_iter, max_seconds=max_seconds, device=device,
                       random_state=0).fit(X, Z=Z)


def predictor(model, exposure, usages=None, Z=None):
    eta = model.a_[:, None] + np.asarray(exposure)[None, :]
    if Z is not None and model.gamma_ is not None:
        eta = eta + model.gamma_ @ np.asarray(Z).T
    if usages is not None:
        eta = eta + model.F_ @ usages.T
    return eta


def subset_for_transform(model, genes):
    """Frozen feature subset with original loading scale and usage penalty.

    Used only to infer activities from observed genes before scoring distinct
    held-out genes. Never renormalize the restricted loading columns.
    """
    result = copy.copy(model)
    for name in ('F_', 'a_', 'theta_'):
        setattr(result, name, getattr(model, name)[genes].copy())
    result.gamma_ = None if model.gamma_ is None else model.gamma_[genes].copy()
    result.l1_G = float(model.l1_G_)
    return result


def score_eta(X, eta, null_eta, theta):
    """Exact NB likelihood/deviance at a common theta; no predictor clipping."""
    x = X.toarray() if sp.issparse(X) else np.asarray(X)
    theta = np.asarray(theta)[:, None]
    lt = np.log(theta)
    mean_nll = lambda e: theta * np.logaddexp(0., e-lt) + x * np.logaddexp(0., lt-e)
    fitted, null = mean_nll(eta), mean_nll(null_eta)
    # At x=0 the saturated mean-dependent NLL is exactly zero.
    saturated_eta = np.log(np.maximum(x, np.finfo(float).tiny))
    saturated = np.where(x == 0, 0., mean_nll(saturated_eta))
    d_model, d_null = 2 * (fitted-saturated).sum(), 2 * (null-saturated).sum()
    constants = -gammaln(x+theta) + gammaln(theta) + gammaln(x+1.)
    log_p0 = -theta * np.logaddexp(0., eta-lt)
    return dict(common_theta_deviance_explained=float(1-d_model/d_null) if d_null > 0 else None,
                nll_per_entry=float((fitted+constants).mean()),
                observed_zero_fraction=float((x == 0).mean()),
                predicted_zero_fraction=float(np.exp(log_p0).mean()),
                zero_brier=float(np.square((x == 0)-np.exp(log_p0)).mean()))
