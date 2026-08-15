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


def test_separation_bounded_and_warns():
    """Test 8: a feature structurally zero wherever a factor is active drives
    its loading to -inf unpenalized; the clip must bound it and warn at
    lambda = 0."""
    sim = simulate_nb_seminmf(p=200, n=400, k=3, random_state=8)
    X = sim.X.copy()
    f0 = int(np.argmax(np.abs(sim.F[:, 0]) > 0))
    X[f0, sim.G[:, 0] > np.median(sim.G[sim.G[:, 0] > 0, 0])] = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = quick_model(3, l1_F=0.0, max_iter=300).fit(X)
    assert np.isfinite(m.F_).all()
    assert np.abs(m.F_).max() <= 15.0 + 1e-9
    messages = [str(w.message) for w in caught]
    assert any("clip" in msg for msg in messages), messages


def test_rank_recovery_surplus_factors_flagged():
    """Test 9: fitting k > k_true leaves surplus factors visibly degenerate /
    near-zero rather than fragmenting real factors."""
    k_true = 3
    sim = simulate_nb_seminmf(p=300, n=500, k=k_true, random_state=9)
    m = quick_model(6).fit(sim.X)

    corr, _ = match_factors(sim.F, m.F_)
    assert np.abs(corr).mean() > 0.8, f"real factors fragmented: {np.abs(corr).round(3)}"

    stats = m.component_stats_
    dev = stats["deviance_explained"].to_numpy()
    weak = (
        stats["degenerate"].to_numpy()
        | (stats["duplicate_of"].to_numpy() >= 0)
        | (dev < 0.05 * max(dev.max(), 1e-12))
    )
    assert weak.sum() >= 6 - k_true - 1, f"surplus not flagged:\n{stats.round(3)}"
