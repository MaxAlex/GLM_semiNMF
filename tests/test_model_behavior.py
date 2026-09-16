"""Spec tests 5-9: null, exposure invariance, covariate absorption,
separation control, rank recovery."""

import warnings

import numpy as np
import pytest

from conftest import match_factors, quick_model
from glm_seminmf import simulate_nb_seminmf


def test_null_data_explains_nothing():
    """Test 5: on k_true = 0 data, fitted factors explain negligible deviance."""
    sim = simulate_nb_seminmf(p=300, n=500, k=0, random_state=5)
    m = quick_model(3).fit(sim.X)
    # ~1.6% of entries worth of free parameters absorb a comparable fraction
    # of noise deviance; real factors in the recovery tests explain 5-50x this.
    assert m.deviance_explained_ < 0.05, m.deviance_explained_
    assert (m.component_stats_["deviance_explained"] < 0.05).all()


def test_exposure_invariance():
    """Test 6: scaling a subset of samples' exposure and resampling leaves G
    approximately unchanged. Catches exposure/offset bugs."""
    base = dict(p=300, n=500, k=3, random_state=6)
    sim1 = simulate_nb_seminmf(**base)

    # identical latent draw, but half the samples get 4x exposure
    sim2 = simulate_nb_seminmf(**base)
    rng = np.random.default_rng(60)
    scaled = rng.random(500) < 0.5
    mu_scale = np.ones(500)
    mu_scale[scaled] = 4.0
    lam = rng.gamma(sim2.theta[:, None], np.exp(
        np.clip(sim2.a[:, None] + sim2.b[None, :] + np.log(mu_scale)[None, :] + sim2.F @ sim2.G.T, -30, 14)
    ) / sim2.theta[:, None])
    X2 = rng.poisson(lam).astype(np.int64)

    m1 = quick_model(3).fit(sim1.X)
    m2 = quick_model(3).fit(X2)

    corr, cols = match_factors(m1.F_, m2.F_)
    assert np.abs(corr).mean() > 0.85
    for j1, j2 in enumerate(cols):
        g1, g2 = m1.G_[:, j1], m2.G_[:, j2]
        active = (g1 > 0) & (g2 > 0)
        if active.sum() < 20:
            continue
        r = np.corrcoef(g1[active], g2[active])[0, 1]
        assert r > 0.7, f"factor {j1}: usage corr {r:.3f}"
        # no systematic usage difference between scaled and unscaled samples
        ratio = np.log(g2[active] + 1e-9) - np.log(g1[active] + 1e-9)
        gap = abs(np.median(ratio[scaled[active]]) - np.median(ratio[~scaled[active]]))
        assert gap < 0.35, f"factor {j1}: exposure leaked into usage, gap {gap:.3f}"


def test_covariate_absorption():
    """Test 7: with Z supplied no factor tracks batch; without Z one does."""
    sim = simulate_nb_seminmf(
        p=300, n=600, k=3, n_batches=2, batch_strength=1.5, random_state=7
    )
    is_b1 = (sim.batch == 1).astype(float)

    def max_batch_corr(m):
        cors = []
        for j in range(m.G_.shape[1]):
            g = m.G_raw_[:, j]
            if np.std(g) < 1e-12:
                continue
            cors.append(abs(np.corrcoef(g, is_b1)[0, 1]))
        return max(cors)

    m_with = quick_model(3).fit(sim.X, Z=sim.Z)
    m_without = quick_model(3).fit(sim.X)
    assert max_batch_corr(m_with) < 0.4, max_batch_corr(m_with)
    assert max_batch_corr(m_without) > 0.6, max_batch_corr(m_without)
    # and gamma should track the true batch coefficients
    r = np.corrcoef(m_with.gamma_[:, 0], sim.gamma[:, 0])[0, 1]
    assert r > 0.8, f"gamma corr {r:.3f}"


def test_separation_bounded_and_reports_status():
    """Test 8: a feature structurally zero wherever a factor is active drives
    its loading towards separation. The accepted contribution stays bounded;
    a deterministic case in test_optimizer checks actual safeguard activation."""
    sim = simulate_nb_seminmf(p=200, n=400, k=3, random_state=8)
    X = sim.X.copy()
    f0 = int(np.argmax(np.abs(sim.F[:, 0]) > 0))
    X[f0, sim.G[:, 0] > np.median(sim.G[sim.G[:, 0] > 0, 0])] = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = quick_model(3, l1_F=0.0, max_iter=300).fit(X)
    assert np.isfinite(m.F_).all()
    # the factor contribution to the log-mean is bounded by the clip
    contrib = np.abs(m.F_) * m.G_.max(axis=0, initial=0.0)[None, :]
    assert contrib.max() <= 15.0 * (1 + 1e-4), contrib.max()
    messages = [str(w.message) for w in caught]
    # This finite sample need not reach the safeguard with a correct optimizer.
    # A deterministic diverging G-only case separately tests actual activation.
    if m.stop_reason_ == "safeguard_hit":
        assert any("clip" in msg for msg in messages), messages
        assert not m.converged_
    elif m.converged_:
        assert m.stationarity_["passed"]


def test_overcomplete_fit_retains_components_and_recovers_signal():
    """Overfitting k may split factors; do not silently drop the extra columns.

    Recovery is empirical. Nonnegativity does not guarantee automatic rank
    selection, so diagnostic flag correctness is tested separately below.
    """
    k_true = 3
    sim = simulate_nb_seminmf(p=300, n=500, k=k_true, random_state=9)
    m = quick_model(6).fit(sim.X)
    corr, _ = match_factors(sim.F, m.F_)
    assert np.abs(corr).mean() > 0.8, f"real factors fragmented: {np.abs(corr).round(3)}"
    assert m.F_.shape[1] == m.G_.shape[1] == len(m.component_stats_) == 6
    np.testing.assert_array_equal(m.component_stats_["degenerate"],
                                  (m.G_ > 0).mean(axis=0) < 1e-3)


def test_known_degenerate_and_duplicate_factors_are_flagged_not_dropped():
    rng = np.random.default_rng(10)
    f = rng.normal(size=(12, 1))
    f /= np.linalg.norm(f)
    fixed = np.column_stack([f, f, np.zeros_like(f)])
    x = rng.poisson(3, size=(12, 20))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = quick_model(3, dispersion=3., max_iter=100).fit(x, F_fixed=fixed)
    np.testing.assert_array_equal(m.F_, fixed)
    assert m.component_stats_.loc[2, "degenerate"]
    assert m.component_stats_.loc[0, "duplicate_of"] == 1
    assert m.component_stats_.loc[1, "duplicate_of"] == 0
    assert any("degenerate or duplicated" in str(w.message) for w in caught)
    assert m.G_.shape[1] == 3
