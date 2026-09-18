"""Objective, boundary KKT, checkpoint, and frozen-parameter regressions."""
import copy
import time

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

import glm_seminmf._fitting as ft
from glm_seminmf import NBGLMSemiNMF
from glm_seminmf._fitting import (
    DataSource, FitConfig, FitState, full_objective, objective_components,
    physical_derivatives, rescale_columns, run_fit, run_transform,
    shift_baseline, sphere_l1_prox, stationarity,
)
from glm_seminmf._likelihood import nb_nll


def problem(chunk=None, *, covariates=True):
    rng = np.random.default_rng(3)
    p, n, k = 5, 7, 2
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    f = t(rng.normal(size=(p, k)))
    f /= f.norm(dim=0)
    z = rng.normal(size=(n, 2)) if covariates else None
    state = FitState(f, t(rng.uniform(.2, 1, (n, k))), t(rng.normal(size=p)),
                     t(rng.normal(size=n) * .1),
                     t(rng.normal(size=(p, 2)) * .1) if covariates else None,
                     t(np.log(np.arange(p) + 2)), True)
    x = rng.poisson(2, size=(p, n))
    return state, DataSource(x, z, torch.device('cpu'), torch.float64, chunk)


@pytest.mark.parametrize('penalties', [(0.3, 0.2, 0.), (0., 0., .4), (.3, .2, .4)])
def test_physical_gradients_and_chunking(penalties):
    cfg = FitConfig(lam=penalties[0], lam_G=penalties[1], lam_G2=penalties[2])
    state, data = problem()
    chunk_state, chunk_data = problem(2)
    grad, _ = physical_derivatives(state, data, cfg)
    other, _ = physical_derivatives(chunk_state, chunk_data, cfg)
    for name in grad:
        torch.testing.assert_close(grad[name], other[name], rtol=1e-12, atol=1e-12)
    obj = lambda: full_objective(state, data, *penalties)
    assert obj() == pytest.approx(full_objective(chunk_state, chunk_data, *penalties), abs=1e-12)
    for name, g in grad.items():
        prm = getattr(state, name)
        for i in range(min(5, prm.numel())):
            value = prm.flatten()[i].item()
            prm.flatten()[i] = value + 1e-6
            hi = obj()
            prm.flatten()[i] = value - 1e-6
            lo = obj()
            prm.flatten()[i] = value
            expected = g.flatten()[i].item() + (cfg.lam * np.sign(value) if name == 'F' else 0)
            assert (hi - lo) / 2e-6 == pytest.approx(expected, abs=2e-7)


def test_normalized_coordinate_chain_rule():
    state, data = problem()
    raw = (2 * state.F).clone().requires_grad_()
    f = raw / raw.norm(dim=0)
    loss = sum(nb_nll(x, state.a[:, None] + state.b[s:e] + f @ state.G()[s:e].T +
                      state.gamma @ z.T, state.log_theta[:, None]).sum()
               for s, e, x, z in data) + .3 * f.abs().sum()
    loss.backward()
    cfg = FitConfig(lam=.3)
    grad, _ = physical_derivatives(state, data, cfg)
    v = grad['F'] + .3 * state.F.sign()
    expected = (v - state.F * (v * state.F).sum(dim=0)) / 2
    torch.testing.assert_close(raw.grad, expected)


def test_sphere_prox_including_all_thresholded_case():
    angles = torch.linspace(-np.pi, np.pi, 100001, dtype=torch.float64)
    candidates = torch.stack([angles.cos(), angles.sin()])
    for z, threshold in [(torch.tensor([[1.], [.5]], dtype=torch.float64), .2),
                         (torch.tensor([[.1], [-.2]], dtype=torch.float64), 2.),
                         (torch.zeros(2, 1, dtype=torch.float64), 1.)]:
        result = sphere_l1_prox(z, threshold)
        loss = .5 * (result - z).square().sum() + threshold * result.abs().sum()
        grid_loss = .5 * (candidates - z).square().sum(dim=0) + threshold * candidates.abs().sum(dim=0)
        assert loss <= grid_loss.min() + 1e-9
        assert result.norm() == pytest.approx(1.)


def test_scale_and_shift_have_distinct_penalty_effects():
    state, data = problem()
    state.G_raw -= state.G_raw.min(dim=0).values
    state.F *= 2
    before_eta = state.eta(0, data.n, data.Z).clone()
    ridge = state.G_raw.square().sum().item()
    rescale_columns(state)
    torch.testing.assert_close(state.eta(0, data.n, data.Z), before_eta)
    assert state.G_raw.square().sum().item() == pytest.approx(4 * ridge)
    state.G_raw += 2
    eta = state.eta(0, data.n, data.Z).clone()
    before = full_objective(state, data, .1, .2, .3)
    shift_baseline(state)
    torch.testing.assert_close(state.eta(0, data.n, data.Z), eta)
    assert full_objective(state, data, .1, .2, .3) < before


@pytest.mark.parametrize('start', [0., 1e-100])
def test_near_zero_activation_and_independent_bounded_oracle(start):
    state, data = problem(covariates=False)
    state.G_raw.fill_(start)
    state.b_trainable = False
    frozen = {n: getattr(state, n).clone() for n in ('F', 'a', 'b', 'log_theta')}
    cfg = FitConfig(lam_G=.1, lam_G2=.7, update_theta=False, max_iter=600,
                    stationarity_tol=2e-7)
    # Independent NumPy NB objective and gradient, not the production evaluator.
    x = data._resident.numpy()
    f, a, b, theta = [frozen[n].numpy() for n in ('F', 'a', 'b', 'log_theta')]
    theta = np.exp(theta)[:, None]
    def objective(v):
        g = v.reshape(state.G_raw.shape)
        eta = a[:, None] + b + f @ g.T
        delta = eta - np.log(theta)
        value = (theta * np.logaddexp(0, delta) + x * np.logaddexp(0, -delta)).sum()
        value += .1 * g.sum() + .35 * (g*g).sum()
        r = theta / (1 + np.exp(-delta)) - x / (1 + np.exp(delta))
        return value, (r.T @ f + .1 + .7 * g).ravel()
    ref = minimize(objective, np.full(state.G_raw.numel(), start), jac=True,
                   bounds=[(0, None)] * state.G_raw.numel(), method='L-BFGS-B',
                   options=dict(ftol=1e-15, gtol=1e-9, maxiter=1000))
    result = run_transform(state, data, cfg)
    assert result.converged, result
    assert state.G_raw.max() > .01
    assert objective(state.G_raw.numpy().ravel())[0] == pytest.approx(ref.fun, abs=1e-8)
    assert result.stationarity['max_residual'] < 2e-7
    for n, value in frozen.items():
        assert torch.equal(getattr(state, n), value)


def stationary_problem():
    state = FitState(torch.ones(1, 1, dtype=torch.float64),
                     torch.zeros(2, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                     torch.zeros(2, dtype=torch.float64), None, torch.zeros(1, dtype=torch.float64), False)
    data = DataSource(np.ones((1, 2)), None, 'cpu', torch.float64, None)
    return state, data


def test_stationary_initial_state_is_success_with_zero_iterations():
    state, data = stationary_problem()
    result = run_fit(state, data, FitConfig(update_theta=False))
    assert result.converged and result.n_iter == 0
    assert result.initial_objective == result.final_objective
    assert len(result.losses) == 1


def test_exact_l1_zero_loading_subgradient():
    state, data = stationary_problem()
    state.F = torch.tensor([[1.], [0.]], dtype=torch.float64)
    state.a = torch.zeros(2, dtype=torch.float64)
    state.log_theta = torch.zeros(2, dtype=torch.float64)
    data = DataSource(np.ones((2, 2)), None, 'cpu', torch.float64, None)
    assert stationarity(state, data, FitConfig(lam=2))['passed']


@pytest.mark.parametrize('exit_kind', ['max_iter', 'timeout', 'line_search_failed'])
def test_initial_checkpoint_survives_budget_and_failed_step(exit_kind):
    state, data = problem()
    original = copy.deepcopy(state)
    cfg = FitConfig(update_theta=False, max_iter=0 if exit_kind == 'max_iter' else 3,
                    max_linesearch=0 if exit_kind == 'line_search_failed' else 30,
                    deadline=time.monotonic()-1 if exit_kind == 'timeout' else None)
    result = run_fit(state, data, cfg)
    assert result.stop_reason == exit_kind
    assert not result.converged and result.n_iter == 0
    assert result.initial_objective == result.final_objective
    assert torch.equal(state.F, original.F)
    assert torch.equal(state.G_raw, original.G_raw)


def test_best_restore_rechecks_residual_and_counts_iterations(monkeypatch):
    import glm_seminmf._fitting as ft
    state, data = stationary_problem()
    state.a.fill_(.5)
    cfg = FitConfig(update_theta=False, max_iter=1, inner_steps=1)
    original_step = ft._step
    def worse_step(state, data, cfg, active, info=None):
        state.a.add_(10)
        return 'accepted'
    monkeypatch.setattr(ft, '_step', worse_step)
    result = run_fit(state, data, cfg)
    assert result.n_iter == 1 and len(result.losses) == 2
    assert result.best_iteration == 0
    assert result.final_objective == result.initial_objective
    assert not result.converged
    assert result.stationarity['max_residual'] > .1


def test_dispersion_transition_and_restoration(monkeypatch):
    import glm_seminmf._fitting as ft
    state, data = problem()
    result = run_fit(state, data, FitConfig(max_iter=8, theta_every=1,
                                         theta_max_updates=2, inner_steps=1))
    assert result.theta_updates <= 2
    assert result.dispersion_status == 'estimated_frozen'
    assert any(event['event'] == 'freeze_restore' for event in result.history)
    audit = stationarity(state, data, FitConfig())
    assert result.stationarity['max_residual'] == audit['max_residual']
    assert result.final_objective <= result.initial_objective + 1e-10
    assert len(result.losses) == result.n_iter + 1


def test_fixed_loadings_theta_and_transform_frozen():
    rng = np.random.default_rng(5)
    x = rng.poisson(2, (5, 9))
    f = rng.normal(size=(5, 2)) * .2
    theta = np.arange(5) + 2.
    model = NBGLMSemiNMF(2, l2_G=.2, dtype='float32', max_iter=20).fit(x, F_fixed=f, theta_fixed=theta)
    np.testing.assert_array_equal(model.F_, f)
    np.testing.assert_allclose(model.theta_, theta, rtol=1e-14)
    saved = [v.copy() for v in (model.F_, model.a_, model.theta_, model.b_)]
    model.transform(x)
    for before, after in zip(saved, (model.F_, model.a_, model.theta_, model.b_)):
        np.testing.assert_array_equal(before, after)
    assert model.export_objective_change_ == 0


def test_fixed_scalar_dispersion_and_active_exposure_without_factors():
    x = np.array([[1, 4, 10], [2, 3, 12]])
    model = NBGLMSemiNMF(0, dispersion=7., exposure='fit', max_iter=100).fit(x)
    np.testing.assert_allclose(model.theta_, 7.)
    assert 'b' in model.stationarity_['blocks']
    assert model.stationarity_['blocks']['b'] < .001


@pytest.mark.parametrize('kwargs', [{'l2_G': -1}, {'l1_F': float('nan')},
                                   {'dispersion': 0}, {'stationarity_tol': 0},
                                   {'max_seconds': -1}, {'inner_steps': 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        NBGLMSemiNMF(0, **kwargs).fit(np.ones((2, 3)))


def test_small_joint_fit_certifies_returned_exact_objective():
    from glm_seminmf import simulate_nb_seminmf
    sim = simulate_nb_seminmf(p=8, n=15, k=1, random_state=10)
    m = NBGLMSemiNMF(1, l1_F=.1, l1_G=.1, l2_G=.1, dispersion=sim.theta,
                    exposure=sim.b, random_state=0, max_iter=500).fit(sim.X)
    assert m.converged_, (m.stop_reason_, m.stationarity_)
    assert m.stationarity_['max_residual'] <= m.stationarity_tol
    assert np.all(np.diff(m.loss_) <= 1e-9)
    eta = m.a_[:,None] + m.b_ + m.F_ @ m.G_.T
    from scipy.stats import nbinom
    theta = m.theta_[:, None]
    independent = -nbinom.logpmf(sim.X, theta, theta/(theta+np.exp(eta))).sum()
    independent += .1 * abs(m.F_).sum() + .1 * m.G_.sum() + .05 * (m.G_**2).sum()
    assert m.final_objective_ == pytest.approx(independent, abs=1e-9)


def separating_problem():
    """X=0 for the only feature, whose only loading is negative: the likelihood
    improves monotonically as the usage grows, with no finite minimizer."""
    state = FitState(-torch.ones(1, 1, dtype=torch.float64),
                     torch.zeros(1, 1, dtype=torch.float64),
                     torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                     None, torch.zeros(1, dtype=torch.float64), False)
    return state, DataSource(np.zeros((1, 1)), None, 'cpu', torch.float64, None)


def test_separation_self_limits_at_the_residual_tolerance_and_is_reported():
    """NB improvement from eta -> -inf saturates exponentially, so an absolute
    residual tolerance is met at a finite usage. The stationarity claim is real
    at that tolerance; the large contribution is reported, never clipped."""
    state, data = separating_problem()
    result = run_transform(state, data, FitConfig(update_theta=False, max_iter=200))
    assert result.stop_reason == 'stationary'
    assert result.stationarity['max_contribution'] == pytest.approx(state.G_raw.item())
    assert state.G_raw.item() > 5  # ran well past any legitimate starting scale
    # Tightening the tolerance moves the stopping point further out: this is an
    # asymptote, not a minimum, which is why a fixed level cannot detect it.
    state2, data2 = separating_problem()
    tight = run_transform(state2, data2, FitConfig(update_theta=False, max_iter=200,
                                                   stationarity_tol=1e-9))
    assert state2.G_raw.item() > state.G_raw.item() + 1


def test_sustained_contribution_growth_stops_the_fit(monkeypatch):
    """Divergence is detected by sustained growth across iterations. The
    trigger is lowered and the tolerance tightened so the usage keeps growing
    instead of meeting the residual test first."""
    state, data = separating_problem()
    monkeypatch.setattr(ft, 'CONTRIBUTION_TRIGGER', .5)
    result = run_transform(state, data, FitConfig(update_theta=False, max_iter=200,
                                                  stationarity_tol=1e-12,
                                                  safeguard_patience=3))
    assert result.stop_reason == 'safeguard_hit'
    assert not result.converged
    assert result.stationarity['safeguard_active']
    event = [e for e in result.history if e['event'] == 'safeguard_divergence']
    assert len(event) == 1 and event[0]['contribution'] >= 2 * event[0]['from_contribution']
    assert event[0]['over_iterations'] == 3
    assert result.final_objective <= result.initial_objective


def test_transient_overshoot_above_trigger_is_not_fatal(monkeypatch):
    """Regression for the seed-6 failure: an ordinary fit whose contribution
    sits above the trigger must still be optimized and certified, not aborted.
    The trigger is lowered so a small deterministic problem exceeds it."""
    state, data = problem(covariates=False)
    state.b_trainable = False
    monkeypatch.setattr(ft, 'CONTRIBUTION_TRIGGER', 0.05)
    cfg = FitConfig(lam_G=.1, lam_G2=.7, update_theta=False, max_iter=600,
                    stationarity_tol=2e-7)
    result = run_transform(state, data, cfg)
    assert result.stationarity['max_contribution'] > 0.05
    assert result.stationarity['safeguard_active']
    assert result.converged, result
    assert result.stop_reason == 'stationary'


def test_step_does_not_reject_trials_on_contribution_level(monkeypatch):
    """The level test must not enter the line search: with the trigger below
    the current state, a step still has to be proposed and accepted."""
    state, data = problem(covariates=False)
    monkeypatch.setattr(ft, 'CONTRIBUTION_TRIGGER', 1e-12)
    before = state.F.clone()
    assert ft._step(state, data, FitConfig(lam_G=.1), ['G_raw']) == 'accepted'
    assert ft._step(state, data, FitConfig(lam=.1), ['F', 'a']) == 'accepted'
    assert not torch.equal(state.F, before)


def test_timeout_skips_optional_finalization_and_returns_diagnostics():
    m = NBGLMSemiNMF(0, dispersion=2., max_seconds=1e-8).fit(np.ones((2, 3)))
    assert m.timed_out_ and not m.converged_
    assert m.n_iter_ == 0 and m.stop_reason_ == 'timeout'
    assert m.finalization_timed_out_
    assert np.isnan(m.deviance_explained_)
    assert np.isfinite(m.final_objective_)


def test_float32_audit_uses_exported_values_and_does_not_relax_tolerance():
    state, data = problem()
    state32 = FitState(**{n: (v.float() if torch.is_tensor(v) else v) for n,v in vars(state).items()})
    cfg = FitConfig(stationarity_tol=1e-7)
    got = stationarity(state32, data, cfg)
    state64 = FitState(**{n: (v.double() if torch.is_tensor(v) else v) for n,v in vars(state32).items()})
    ref = stationarity(state64, data, cfg)
    assert got['blocks'] == ref['blocks']
    assert got['tolerance'] == 1e-7
    assert not got['passed']


def test_transform_covariates_required_and_frozen():
    rng = np.random.default_rng(3)
    x = rng.poisson(2, (5, 8))
    z = rng.normal(size=(8, 1))
    m = NBGLMSemiNMF(1, dispersion=3., max_iter=20).fit(x, Z=z)
    gamma = m.gamma_.copy()
    m.transform(x, Z=z)
    np.testing.assert_array_equal(m.gamma_, gamma)
    with pytest.raises(ValueError, match='Z.*required|Z must be supplied'):
        m.transform(x)


def test_stable_objective_difference_resolves_sub_ulp_improvement():
    from glm_seminmf._fitting import objective_difference
    state, data = stationary_problem()
    # At a nearly stationary state, retain an improvement that is tiny
    # compared with the full objective, without loss-dependent slack.
    state.a.fill_(1e-6)
    cfg = FitConfig(update_theta=False)
    old = {'a': state.a.clone()}
    grad, _ = physical_derivatives(state, data, cfg)
    state.a -= grad['a'] * 1e-3
    difference = objective_difference(state, data, cfg, old)
    assert difference < 0
    eps = (state.a - old['a']).item()
    expected = grad['a'].item() * eps + .5 * eps**2
    assert difference == pytest.approx(expected, rel=1e-5, abs=1e-24)


def test_stable_difference_matches_full_objective_for_all_blocks():
    from glm_seminmf._fitting import objective_difference, _snapshot
    state, data = problem(2)
    cfg = FitConfig(lam=.3, lam_G=.2, lam_G2=.1)
    old = _snapshot(state)
    before = full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
    state.F *= .9
    state.G_raw += .02
    state.a -= .03
    state.b += .01
    state.gamma *= 1.1
    diff = objective_difference(state, data, cfg, old)
    assert diff == pytest.approx(full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)-before, abs=1e-12)


def test_float32_and_float64_usage_predictions_agree_with_declared_tolerance():
    outputs = []
    for dtype in (torch.float32, torch.float64):
        state, data = problem(2, covariates=False)
        state = FitState(**{n: (v.to(dtype) if torch.is_tensor(v) else v)
                            for n, v in vars(state).items()})
        state.b_trainable = False
        result = run_transform(state, data, FitConfig(lam_G=.1, lam_G2=.7,
                               update_theta=False, max_iter=500, stationarity_tol=1e-4))
        assert result.converged, result.stationarity
        outputs.append(state.eta(0, data.n, None).double().numpy())
    np.testing.assert_allclose(*outputs, atol=2e-4, rtol=2e-4)


def test_rejected_nonfinite_proposal_restores_parameters(monkeypatch):
    import glm_seminmf._fitting as ft
    state, data = problem()
    original = copy.deepcopy(state)
    def broken_derivatives(state, data, cfg, **kwargs):
        grad, curv = saved(state, data, cfg, **kwargs)
        if not kwargs.get('diagnostics'):
            grad['G_raw'].fill_(float('nan'))
        return grad, curv
    saved = ft.physical_derivatives
    monkeypatch.setattr(ft, 'physical_derivatives', broken_derivatives)
    result = run_fit(state, data, FitConfig(update_theta=False, max_iter=2, max_linesearch=2))
    assert result.stop_reason == 'nonfinite'
    assert not result.converged
    assert torch.equal(state.G_raw, original.G_raw)
    assert torch.equal(state.F, original.F)


def test_numerical_floor_certifies_when_no_representable_step_helps():
    """An exhausted line search whose best trial cannot improve the objective by
    more than a few ulps has proved the iterate optimal for this arithmetic.
    float32 reaches that floor at an absolute residual far above any tolerance."""
    state, data = problem(covariates=False)
    state32 = FitState(**{n: (v.float() if torch.is_tensor(v) else v)
                          for n, v in vars(state).items()})
    state32.b_trainable = False
    cfg = FitConfig(lam=.3, lam_G=.1, update_theta=False, max_iter=4000,
                    stationarity_tol=1e-12)  # unreachable in float32
    result = run_transform(state32, data, cfg)
    assert result.stop_reason == 'numerically_stationary'
    assert result.converged
    floor = result.stationarity['numerical_floor']
    assert abs(floor['best_change']) <= floor['slack']
    # The certificate is numerical, not a relaxed tolerance: the residual is
    # still reported honestly and still exceeds what was asked for.
    assert result.stationarity['max_residual'] > cfg.stationarity_tol
    assert not result.stationarity['passed']


def test_numerical_floor_is_not_claimed_when_another_block_is_unconverged():
    """The certificate covers only the block the line search exhausted on."""
    state, data = problem(covariates=False)
    cfg = FitConfig(update_theta=False, max_iter=3, max_linesearch=0)
    result = run_fit(state, data, cfg)
    assert result.stop_reason == 'line_search_failed'
    assert not result.converged
    assert result.stationarity['numerical_floor'] is None
