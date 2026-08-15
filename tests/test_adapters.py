"""Adapter layer: labeled containers in, labeled factors out."""

import numpy as np
import pandas as pd

from conftest import quick_model
from glm_seminmf import simulate_nb_seminmf
from glm_seminmf.adapters import fit_dataframe


def test_fit_dataframe_labels():
    sim = simulate_nb_seminmf(p=120, n=150, k=2, random_state=13)
    X = pd.DataFrame(
        sim.X,
        index=[f"feat{i}" for i in range(120)],
        columns=[f"s{j}" for j in range(150)],
    )
    m = quick_model(2, max_iter=40)
    loadings, usages = fit_dataframe(m, X)
    assert list(loadings.index) == list(X.index)
    assert list(usages.index) == list(X.columns)
    np.testing.assert_array_equal(loadings.to_numpy(), m.F_)
    np.testing.assert_array_equal(usages.to_numpy(), m.G_)


def test_fit_dataframe_with_covariates():
    sim = simulate_nb_seminmf(p=120, n=150, k=2, n_batches=2, random_state=14)
    X = pd.DataFrame(sim.X)
    cov = pd.DataFrame({"batch": np.where(sim.batch == 1, "b1", "b0")})
    m = quick_model(2, max_iter=40)
    fit_dataframe(m, X, covariates=cov)
    assert m.gamma_ is not None and m.gamma_.shape == (120, 1)
