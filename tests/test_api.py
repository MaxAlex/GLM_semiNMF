"""API contract: fitted attributes, shapes, ordering, transform, G_raw_."""

import numpy as np
import pytest

from conftest import quick_model
from glm_seminmf import simulate_nb_seminmf

P, N, K = 250, 350, 3


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_nb_seminmf(p=P, n=N, k=K, random_state=12)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(K).fit(sim.X)
    return sim, m


def test_attribute_shapes(fitted):
    sim, m = fitted
    assert m.F_.shape == (P, K)
    assert m.G_.shape == (N, K)
    assert m.G_raw_.shape == (N, K)
    assert m.a_.shape == (P,)
    assert m.b_.shape == (N,)
    assert m.theta_.shape == (P,)
    assert m.gamma_ is None
    assert m.loss_.ndim == 1 and len(m.loss_) == m.n_iter_
    assert isinstance(m.converged_, bool)
    assert 0.0 < m.deviance_explained_ <= 1.0


def test_F_columns_unit_norm(fitted):
    _, m = fitted
    np.testing.assert_allclose(np.linalg.norm(m.F_, axis=0), 1.0, rtol=1e-5)


def test_G_nonneg_with_exact_zeros(fitted):
    _, m = fitted
    assert (m.G_ >= 0).all()
    assert (m.G_ == 0).any(), "snap-to-zero should produce exact zeros"


def test_G_raw_continuous_tie_free(fitted):
    """Downstream rank statistics need a tie-free continuous variable."""
    _, m = fitted
    vals = m.G_raw_.ravel()
    assert np.unique(vals).size > 0.99 * vals.size


def test_factors_ordered_by_deviance_explained(fitted):
    _, m = fitted
    dev = m.component_stats_["deviance_explained"].to_numpy()
    assert (np.diff(dev) <= 1e-12).all(), dev


def test_component_stats_columns(fitted):
    _, m = fitted
    expected = {
        "deviance_explained",
        "usage_frac_nonzero",
        "usage_mean",
        "neg_loading_mass",
        "n_features_above_threshold",
        "degenerate",
        "duplicate_of",
    }
    assert expected <= set(m.component_stats_.columns)
    assert len(m.component_stats_) == K


def test_transform_matches_fit_usages(fitted):
    sim, m = fitted
    G2 = m.transform(sim.X)
    assert G2.shape == (N, K)
    for j in range(K):
        r = np.corrcoef(m.G_[:, j], G2[:, j])[0, 1]
        assert r > 0.98, f"factor {j}: {r:.4f}"


def test_fit_transform_returns_G(fitted):
    sim, _ = fitted
    m = quick_model(K, max_iter=30)
    G = m.fit_transform(sim.X)
    np.testing.assert_array_equal(G, m.G_)


def test_transform_before_fit_raises():
    m = quick_model(2)
    with pytest.raises(RuntimeError, match="before fit"):
        m.transform(np.zeros((5, 4), dtype=int))


def test_wrong_feature_count_raises(fitted):
    _, m = fitted
    with pytest.raises(ValueError, match="features"):
        m.transform(np.zeros((P + 1, 10), dtype=int))
