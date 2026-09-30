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


@pytest.mark.parametrize("basis", [np.ones(5), np.ones((5, 2, 1)),
                                  np.full((5, 2), np.nan), np.full((5, 2), np.inf)])
def test_invalid_fixed_basis_raises(basis):
    with pytest.raises(ValueError, match="F_fixed"):
        quick_model(2).fit(np.ones((5, 8)), F_fixed=basis)


def test_nonunit_and_zero_columns_preserved_through_fit_and_transform():
    rng = np.random.default_rng(34)
    x = rng.poisson(2, (12, 25))
    f = rng.normal(size=(12, 3)) * np.array([.1, 2., 0.])
    saved = f.copy()
    m = quick_model(99, dtype="float32", dispersion=5., l2_G=.2).fit(x, F_fixed=f)
    np.testing.assert_array_equal(f, saved)
    np.testing.assert_array_equal(m.F_, saved)
    assert m.compute_dtype_ == "float64"
    assert m.G_.shape == (25, 3)
    assert "F" not in m.stationarity_["blocks"]
    assert np.all(m.G_[:, 2] == 0)
    m.transform(x)
    np.testing.assert_array_equal(m.F_, saved)


def test_projection_start_avoids_spectral_initialization(monkeypatch):
    import glm_seminmf._init as init_module

    def unexpected_svd(*args, **kwargs):
        raise AssertionError("fixed-basis fit must not initialize unrelated factors")

    monkeypatch.setattr(init_module, "_svd_init", unexpected_svd)
    f = np.eye(6, 2)
    m = quick_model(99, max_iter=0, dispersion=5.).fit(np.ones((6, 10)), F_fixed=f)
    np.testing.assert_array_equal(m.F_, f)


def test_explicit_usage_start_is_preserved_with_fixed_basis():
    f = np.eye(6, 2) * 2
    g = np.arange(20).reshape(10, 2).astype(float)
    g[0] = 0  # already touch-zero, so baseline canonicalization is a no-op
    m = quick_model(99, init=(f, g), max_iter=0, dispersion=5.).fit(
        np.ones((6, 10)), F_fixed=f)
    np.testing.assert_array_equal(m.F_, f)
    np.testing.assert_array_equal(m.G_, g)


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
    assert m.loss_.ndim == 1 and len(m.loss_) == m.n_iter_ + 1
    assert 0.0 <= m.deviance_explained_ <= 1.0


def test_frozen_fit_beats_null_with_curvature_scaled_steps():
    """Projection must improve over the null with the new adaptive solver.

    The incoming Adam-specific 4x learning-rate boost is superseded by curvature
    scaling and line search, shared with transform.
    """
    F0 = _shared_F(p=150, k=4, seed=21)
    sim = simulate_nb_seminmf(p=150, n=400, k=4, F_fixed=F0, random_state=21)
    m = quick_model(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X, F_fixed=F0)
    # The fit must at least beat the null it is nested against.
    assert m.deviance_explained_ > 0.0, (
        f"frozen-F fit landed at {m.deviance_explained_:.4f}; negative means it lost to "
        "the k=0 null, which G=0 reproduces exactly"
    )
    assert m.improved_on_init_
    assert m.final_objective_ < m.initial_objective_
    assert "F" not in m.stationarity_["blocks"]


def test_unfrozen_fit_uses_curvature_scaled_steps():
    sim = simulate_nb_seminmf(p=150, n=300, k=3, random_state=22)
    m = quick_model(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sim.X)
    assert m.F_fixed_ is False
    assert m.deviance_explained_ > 0.0
