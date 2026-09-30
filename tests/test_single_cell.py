"""Sparse-count alternatives and gene-subset exposure contracts."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from glm_seminmf import NBGLMSemiNMF
from glm_seminmf._init import initialize
from glm_seminmf._fitting import FitConfig, physical_derivatives, information_usage_penalty
from glm_seminmf._dispersion import dispersion_from_moments
from glm_seminmf.adapters import fit_anndata
from test_optimizer import problem


@pytest.mark.parametrize('active', [('G_raw',), ('F', 'a', 'gamma'), ('a',), ('b',)])
def test_active_derivatives_match_full_audit(active):
    state, data = problem(chunk=3)
    cfg = FitConfig(lam_G=.2, lam_G2=.3)
    grad, curv = physical_derivatives(state, data, cfg)
    subset, metric = physical_derivatives(state, data, cfg, active=active)
    assert set(subset) == set(active)
    for name in active:
        torch.testing.assert_close(subset[name], grad[name], rtol=0, atol=0)
        torch.testing.assert_close(metric[name], curv[name], rtol=0, atol=0)


def test_information_penalty_tracks_depth_and_ignores_zero_loading_rows():
    state, data = problem(covariates=False)
    state.a.fill_(-4)
    low = information_usage_penalty(state, data)
    state.a.add_(2)
    assert information_usage_penalty(state, data) > low
    state.a.sub_(2)
    # Adding uninvolved genes changes p but cannot change score information.
    state.F = torch.cat([state.F, torch.zeros_like(state.F)])
    state.a = torch.cat([state.a, state.a])
    state.log_theta = torch.cat([state.log_theta, state.log_theta])
    assert information_usage_penalty(state, data) == pytest.approx(low)


def test_information_penalty_frozen_for_transform():
    rng = np.random.default_rng(4)
    x = rng.poisson(.1, (30, 40))
    model = NBGLMSemiNMF(2, l1_G='information', dispersion=10., max_iter=5,
                        device='cpu', random_state=3).fit(x)
    lam = model.l1_G_
    model.transform(x)
    assert model.l1_G_ == lam > 0
    assert model.transform_objective_components_['usage_l1'] == pytest.approx(
        lam * model.transform_raw_.sum())


def test_external_exposure_survives_gene_subset_and_adapter():
    rng = np.random.default_rng(2)
    full = rng.poisson(2, (40, 25))
    offsets = np.log(full.sum(0) / np.median(full.sum(0)))
    adata = SimpleNamespace(X=sp.csr_matrix(full[:12].T),
                            obs=pd.DataFrame({'log_exposure': offsets}),
                            obs_names=[f'c{i}' for i in range(25)],
                            var_names=[f'g{i}' for i in range(12)])
    model = NBGLMSemiNMF(1, dispersion=10., max_iter=2, device='cpu', random_state=1)
    fit_anndata(model, adata, exposure_key='log_exposure')
    np.testing.assert_array_equal(model.b_, offsets)
    with pytest.raises(ValueError, match='external exposure'):
        model.transform(full[:12])
    model.transform(full[:12], exposure=offsets)
    np.testing.assert_array_equal(model.transform_b_, offsets)


def test_pearson_dense_sparse_streamed_and_covariates(monkeypatch):
    import glm_seminmf._init as init
    rng = np.random.default_rng(6)
    z = np.tile([0., 1.], 30)[:, None]
    b = rng.normal(0, .4, 60)
    x = rng.poisson(np.exp(-2 + b + rng.normal(0, .5, (35, 1)) + .4*z.T))
    dense = initialize(x, b, 3, 'pearson', 2, z, batch_size=17)
    sparse = initialize(sp.csc_matrix(x), b, 3, 'pearson', 2, z, batch_size=17)
    for a, c in zip(dense, sparse):
        np.testing.assert_allclose(a, c, atol=1e-10)
    monkeypatch.setattr(init, '_DENSIFY_ELEMENTS', 0)
    streamed = initialize(sp.csc_matrix(x), b, 3, 'pearson', 2, z, batch_size=17)
    np.testing.assert_allclose(dense[0] @ dense[1].T, streamed[0] @ streamed[1].T, atol=1e-8)
    assert np.isfinite(streamed[0]).all() and (streamed[1] >= 0).all()
    assert streamed[3].shape == (35, 1)


def test_pooled_trend_recovers_constant_dispersion_across_mean_range():
    from test_dispersion import moments
    ssr, sm, sm2, alpha = moments(slope=0.)
    theta = dispersion_from_moments(ssr, sm, sm2, 'trend_pooled').exp()
    torch.testing.assert_close(theta, 1/alpha, atol=1e-12, rtol=1e-12)


def test_pooled_trend_does_not_select_positive_noise_at_low_means():
    rng = np.random.default_rng(0)
    mu = np.exp(rng.uniform(np.log(.005), np.log(5), (1000, 1)))
    exposure = np.exp(rng.normal(0, .7, 500))
    mean = mu*exposure
    x = rng.poisson(rng.gamma(10., mean/10.))
    fitted = x.sum(1)[:, None]/exposure.sum()*exposure
    moments = [torch.tensor(v) for v in (((x-fitted)**2).sum(1), fitted.sum(1), (fitted**2).sum(1))]
    pooled = dispersion_from_moments(*moments, 'trend_pooled')
    original = dispersion_from_moments(*moments, 'trend')
    low = mu[:, 0] < .05
    error = lambda th: np.sqrt(np.mean((th.numpy()[low]-np.log(10.))**2))
    assert error(pooled) < .3
    assert error(pooled) < .2*error(original)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pooled_dispersion_reproducible_and_handles_empty_counts(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    zero = torch.zeros(60, dtype=torch.float64, device=device)
    a = dispersion_from_moments(zero, zero, zero, 'trend_pooled')
    b = dispersion_from_moments(zero, zero, zero, 'trend_pooled')
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


def test_rare_program_not_marked_collapsed():
    # Use a supplied start and no updates to isolate diagnostic semantics.
    x = np.ones((4, 2001), dtype=int)
    f = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    g = np.zeros((2001, 2)); g[0, 0] = 1
    m = NBGLMSemiNMF(2, init=(f, g), max_iter=0, dispersion=10., device='cpu').fit(x, F_fixed=f)
    assert bool(m.component_stats_.iloc[0]['rare'])
    assert not bool(m.component_stats_.iloc[0]['degenerate'])
    assert bool(m.component_stats_.iloc[1]['degenerate'])
    assert not bool(m.component_stats_.iloc[1]['rare'])


def test_reused_steps_preserve_objective_and_frozen_block_contract():
    from glm_seminmf._fitting import run_transform
    state, data = problem(covariates=False)
    state.b_trainable = False
    other = copy.deepcopy(state)
    cfg = dict(update_theta=False, lam_G=.1, lam_G2=.7, max_iter=500, stationarity_tol=1e-6)
    reference = run_transform(state, data, FitConfig(**cfg))
    reused = run_transform(other, data, FitConfig(**cfg, reuse_step_size=True))
    assert reference.converged and reused.converged
    assert reused.final_objective == pytest.approx(reference.final_objective, abs=1e-8)
    assert np.all(np.diff(reused.losses) <= 1e-10)
