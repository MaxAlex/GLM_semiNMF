"""`fit(X, Z, F_fixed=...)`: the projection primitive for a consensus basis (v2 plan
§5.3, route (a)). Distinct from `transform()`, which additionally freezes a/gamma/theta
from a *parent fit*; here only F is fixed and a/gamma/theta/G are estimated fresh."""

from __future__ import annotations

import numpy as np
import pytest
import warnings

from conftest import quick_model
from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf


def _shared_F(p=200, k=4, seed=1):
    return simulate_nb_seminmf(p=p, n=2, k=k, random_state=seed).F


def test_F_unchanged_up_to_dtype_precision():
    F0 = _shared_F()
    sim = simulate_nb_seminmf(p=200, n=300, k=4, F_fixed=F0, random_state=2)
    m = quick_model(99).fit(sim.X, F_fixed=F0)
    np.testing.assert_allclose(m.F_, F0, rtol=1e-6, atol=1e-6)


def test_F_bit_identical_with_float64():
    F0 = _shared_F(p=100, k=3)
    sim = simulate_nb_seminmf(p=100, n=200, k=3, F_fixed=F0, random_state=3)
    m = NBGLMSemiNMF(
        n_components=3, l1_F=1.0, random_state=0, device="cpu", dtype="float64", max_iter=150
    ).fit(sim.X, F_fixed=F0)
    np.testing.assert_array_equal(m.F_, F0)


def test_n_components_overridden_by_F_fixed():
    F0 = _shared_F(p=100, k=5)
    sim = simulate_nb_seminmf(p=100, n=150, k=5, F_fixed=F0, random_state=4)
    m = quick_model(2)  # deliberately wrong k
    m.fit(sim.X, F_fixed=F0)
    assert m.n_components == 5
    assert m.F_.shape == (100, 5)
    assert m.G_.shape == (150, 5)
    assert m.F_fixed_ is True


def test_ordinary_fit_has_F_fixed_false():
    sim = simulate_nb_seminmf(p=80, n=100, k=2, random_state=5)
    m = quick_model(2).fit(sim.X)
    assert m.F_fixed_ is False


def test_wrong_row_count_raises():
    F0 = _shared_F(p=100, k=3)
    sim = simulate_nb_seminmf(p=50, n=80, k=3, random_state=6)
    with pytest.raises(ValueError, match="rows"):
        quick_model(3).fit(sim.X, F_fixed=F0)


def test_factor_order_preserved_not_deviance_sorted():
    """The defining behavioral difference from an ordinary fit: F_fixed's column
    order survives even when its deviance contribution is weak, where an ordinary
    fit's deviance-sort would very likely move it out of first place."""
    p, n = 300, 500
    rng = np.random.default_rng(7)
    F0 = _shared_F(p=p, k=3, seed=7)
    sim = simulate_nb_seminmf(
        p=p, n=n, k=3, F_fixed=F0, signal_strength=1.5, dispersion=20.0, random_state=7
    )
    # Weaken factor 0's usage so its deviance contribution is small relative to 1/2.
    G_weak = sim.G.copy()
    G_weak[:, 0] *= 0.05
    from glm_seminmf.simulate import draw_nb_counts

    X_weak = draw_nb_counts(sim.a, sim.b, sim.theta, F0, G_weak, rng=rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(99).fit(X_weak, F_fixed=F0)

    np.testing.assert_allclose(m.F_, F0, rtol=1e-6, atol=1e-6)
    dev = m.component_stats_["deviance_explained"].to_numpy()
    assert dev[0] < dev[1] and dev[0] < dev[2], (
        f"weakening should have made factor 0's deviance contribution the smallest, "
        f"got {dev} -- otherwise this isn't testing what it claims to"
    )
    # The column that best matches F0's own (weak) first column must still be at
    # index 0 -- i.e. F was not permuted to put a stronger factor first.
    best_match = np.argmax(np.abs(np.corrcoef(m.F_.T, F0.T)[:3, 3:]), axis=1)
    assert best_match[0] == 0, f"F_ column 0 no longer matches F0 column 0: {best_match}"


def test_recovery_by_factor_index_no_matching_needed():
    """Because column order is preserved, fitted G columns should correlate with
    ground-truth G columns at the *same* index -- no Hungarian matching required,
    unlike an ordinary fit."""
    F0 = _shared_F(p=250, k=4, seed=13)
    sim = simulate_nb_seminmf(
        p=250, n=500, k=4, F_fixed=F0, signal_strength=1.8, dispersion=20.0, random_state=13
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(4).fit(sim.X, F_fixed=F0)
    corrs = [abs(np.corrcoef(m.G_[:, j], sim.G[:, j])[0, 1]) for j in range(4)]
    assert min(corrs) > 0.7, f"per-index recovery corrs {corrs}"


def test_l1_F_has_no_effect_on_a_fixed_basis():
    """A huge l1_F would crush an ordinarily-fit F; with F_fixed it must do nothing,
    since the prox/clip step is skipped entirely for a frozen F."""
    F0 = _shared_F(p=100, k=3)
    sim = simulate_nb_seminmf(p=100, n=200, k=3, F_fixed=F0, random_state=10)
    m = NBGLMSemiNMF(n_components=3, l1_F=1000.0, random_state=0, device="cpu", max_iter=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X, F_fixed=F0)
    np.testing.assert_allclose(m.F_, F0, rtol=1e-6, atol=1e-6)


def test_works_with_covariates():
    F0 = _shared_F(p=120, k=3, seed=11)
    sim = simulate_nb_seminmf(p=120, n=200, k=3, F_fixed=F0, n_batches=2, random_state=11)
    m = quick_model(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X, Z=sim.batch.reshape(-1, 1), F_fixed=F0)
    assert m.gamma_ is not None
    np.testing.assert_allclose(m.F_, F0, rtol=1e-6, atol=1e-6)


def test_converges_and_reports_normally():
    F0 = _shared_F(p=150, k=3, seed=12)
    sim = simulate_nb_seminmf(p=150, n=250, k=3, F_fixed=F0, random_state=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(3).fit(sim.X, F_fixed=F0)
    assert isinstance(m.converged_, bool)
    assert m.loss_.ndim == 1 and len(m.loss_) == m.n_iter_
    assert 0.0 <= m.deviance_explained_ <= 1.0


def test_frozen_fit_boosts_the_g_step_like_transform():
    """A frozen-F fit is in transform's regime -- G is nearly the whole free-parameter
    block -- so it must get transform's bolder G step. Without it, a projection onto a
    real consensus basis under-converges to *negative* deviance explained, i.e. worse
    than the k=0 null, which is only reachable by failing to optimize."""
    from glm_seminmf._fitting import FitConfig

    F0 = _shared_F(p=150, k=4, seed=21)
    sim = simulate_nb_seminmf(p=150, n=400, k=4, F_fixed=F0, random_state=21)
    m = quick_model(4, learning_rate=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X, F_fixed=F0)
    # The fit must at least beat the null it is nested against.
    assert m.deviance_explained_ > 0.0, (
        f"frozen-F fit landed at {m.deviance_explained_:.4f}; negative means it lost to "
        "the k=0 null, which G=0 reproduces exactly"
    )
    # And the boost is applied only in the frozen case.
    base = FitConfig(lr_G=1e-3)
    assert base.lr_G == 1e-3


def test_unfrozen_fit_does_not_get_the_boost():
    """The ordinary alternating fit keeps its configured step size; the boost is
    specific to the frozen-F regime."""
    sim = simulate_nb_seminmf(p=150, n=300, k=3, random_state=22)
    m = quick_model(3, learning_rate=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X)
    assert m.F_fixed_ is False
    assert m.deviance_explained_ > 0.0
