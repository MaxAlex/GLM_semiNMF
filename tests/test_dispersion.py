"""Dispersion estimation: parameter-space boundary handling and trend fitting.

The moment estimate alpha = (ssr - s_mu) / s_mu2 leaves the parameter space
whenever a feature's residual spread is at or below its Poisson expectation.
Clamping those features to the alpha floor makes them theta ~ 1e4, and NB
curvature carries theta directly, so a handful is enough to stall the
mean-model solver. These tests pin the boundary behavior.
"""

import numpy as np
import pytest
import torch

from conftest import match_factors, quick_model
from glm_seminmf import simulate_nb_seminmf
from glm_seminmf._dispersion import _ALPHA_MIN, dispersion_from_moments

THETA_CEILING = 1.0 / _ALPHA_MIN


def moments(p=200, n=100, slope=-0.3, unusable=()):
    """Self-consistent moment accumulators on a known log-alpha/log-mean trend.

    Features listed in ``unusable`` are made underdispersed (excess <= 0), the
    case the raw estimator cannot represent.
    """
    s_mu = torch.tensor(np.geomspace(50.0, 5000.0, p), dtype=torch.float64)
    s_mu2 = s_mu**2 / n
    centered = torch.log(s_mu) - torch.log(s_mu).mean()
    alpha_true = torch.exp(-1.0 + slope * centered)
    ssr = s_mu + alpha_true * s_mu2
    for i in unusable:
        ssr[i] = s_mu[i] * 0.9  # residual spread below the Poisson expectation
    return ssr, s_mu, s_mu2, alpha_true


def test_recovers_a_known_trend_when_every_feature_is_usable():
    ssr, s_mu, s_mu2, alpha_true = moments()
    theta = torch.exp(dispersion_from_moments(ssr, s_mu, s_mu2, "trend"))
    torch.testing.assert_close(theta, 1.0 / alpha_true, rtol=0.05, atol=0.0)


def test_underdispersed_features_take_the_trend_not_the_alpha_floor():
    bad = list(range(0, 200, 20))  # 10 features spread across the mean range
    ssr, s_mu, s_mu2, alpha_true = moments(unusable=bad)
    theta = torch.exp(dispersion_from_moments(ssr, s_mu, s_mu2, "trend"))

    idx = torch.tensor(bad)
    assert (theta[idx] < 0.05 * THETA_CEILING).all(), theta[idx]
    # They land on the trend, which at these means is the truth to within noise.
    torch.testing.assert_close(theta[idx], 1.0 / alpha_true[idx], rtol=0.25, atol=0.0)
    # Usable neighbours are unaffected by their presence.
    good = torch.tensor([i for i in range(200) if i not in bad])
    torch.testing.assert_close(theta[good], 1.0 / alpha_true[good], rtol=0.05, atol=0.0)


def test_unusable_features_do_not_steer_the_trend():
    """Enough boundary features to beat the 1%/99% winsorization on their own."""
    bad = list(range(0, 200, 4))  # 25% of features
    ssr, s_mu, s_mu2, alpha_true = moments(unusable=bad)
    theta = torch.exp(dispersion_from_moments(ssr, s_mu, s_mu2, "trend"))
    good = torch.tensor([i for i in range(200) if i not in bad])
    torch.testing.assert_close(theta[good], 1.0 / alpha_true[good], rtol=0.05, atol=0.0)
    assert theta.max() < 0.05 * THETA_CEILING


def test_pooled_fallback_when_too_few_features_support_a_trend():
    ssr, s_mu, s_mu2, _ = moments(p=12, unusable=range(8))  # 4 usable, need 10
    theta = torch.exp(dispersion_from_moments(ssr, s_mu, s_mu2, "trend"))
    assert torch.isfinite(theta).all() and (theta > 0).all()
    # Every boundary feature collapses onto the same pooled value.
    assert theta[:8].std() < 1e-9 * theta[:8].mean()
    assert theta.max() < 0.05 * THETA_CEILING


def test_feature_mode_stays_raw_by_contract():
    """'feature' is documented as the unshrunk estimate; it keeps the clamp."""
    ssr, s_mu, s_mu2, _ = moments(unusable=[3])
    theta = torch.exp(dispersion_from_moments(ssr, s_mu, s_mu2, "feature"))
    assert theta[3] == pytest.approx(THETA_CEILING)


@pytest.mark.parametrize("mode", ["trend", "feature", "shared"])
def test_estimates_stay_inside_the_clamp_for_degenerate_moments(mode):
    zero = torch.zeros(20, dtype=torch.float64)
    theta = torch.exp(dispersion_from_moments(zero, zero, zero, mode))
    assert torch.isfinite(theta).all()
    assert (theta >= 1.0 / 1e3).all() and (theta <= THETA_CEILING).all()


def test_estimated_dispersion_certifies_and_has_no_runaway_tail():
    """End-to-end: the boundary features used to become theta ~ 1e4, which
    inflated curvature (h = (theta + X) * sigmoid * sigmoid) enough to stop the
    mean model certifying at all."""
    sim = simulate_nb_seminmf(p=300, n=500, k=3, random_state=6)
    m = quick_model(3, max_iter=1000).fit(sim.X)
    assert m.converged_, (m.stop_reason_, m.stationarity_)
    assert m.theta_.max() < 20 * sim.theta.max(), m.theta_.max()
    corr, _ = match_factors(sim.F, m.F_)
    assert np.abs(corr).mean() > 0.85
