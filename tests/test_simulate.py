"""``F_fixed`` in ``simulate_nb_seminmf``: reusing one shared loading matrix
across many calls (one per donor, in a cohort-structure simulator) rather
than drawing a fresh one every call."""

import numpy as np
import pytest

from conftest import match_factors, quick_model
from glm_seminmf import simulate_nb_seminmf


def _shared_F(p=300, k=4, seed=0):
    return simulate_nb_seminmf(p=p, n=2, k=k, random_state=seed).F


def test_F_fixed_reused_bit_identical():
    """Two calls with the same F_fixed return exactly that matrix, not a
    fresh draw — the whole point of the parameter."""
    F0 = _shared_F()
    sim1 = simulate_nb_seminmf(p=300, n=50, k=4, F_fixed=F0, random_state=1)
    sim2 = simulate_nb_seminmf(p=300, n=80, k=4, F_fixed=F0, random_state=2)
    np.testing.assert_array_equal(sim1.F, F0)
    np.testing.assert_array_equal(sim2.F, F0)
    # Different donor draws (different n, different seed) still differ in G/X.
    assert sim1.G.shape[0] == 50 and sim2.G.shape[0] == 80
    assert not np.array_equal(sim1.X[:, :10], sim2.X[:, :10])


def test_F_fixed_infers_k_and_ignores_density_F():
    F0 = _shared_F(k=6)
    sim = simulate_nb_seminmf(p=300, n=50, k=999, density_F=0.9, F_fixed=F0, random_state=0)
    assert sim.F.shape[1] == 6
    assert sim.G.shape[1] == 6


def test_F_fixed_wrong_row_count_raises():
    F0 = _shared_F(p=300, k=4)
    with pytest.raises(ValueError, match="rows"):
        simulate_nb_seminmf(p=200, n=50, k=4, F_fixed=F0, random_state=0)


def test_F_fixed_all_zero_column_does_not_crash():
    F0 = _shared_F(p=100, k=3)
    F0 = F0.copy()
    F0[:, 1] = 0.0
    sim = simulate_nb_seminmf(p=100, n=30, k=3, F_fixed=F0, random_state=0)
    assert np.isfinite(sim.X).all()
    assert sim.G[:, 1].sum() >= 0.0  # column still well-formed, just carries no signal


def test_F_fixed_still_recoverable():
    """A model fit against F_fixed-generated data should still recover that
    shared basis — i.e. this isn't degenerate data, just a shared draw."""
    F0 = _shared_F(p=300, k=4, seed=7)
    sim = simulate_nb_seminmf(
        p=300, n=600, k=4, F_fixed=F0, signal_strength=1.5, dispersion=20.0, random_state=7
    )
    m = quick_model(4).fit(sim.X)
    corr, _ = match_factors(sim.F, m.F_)
    assert np.abs(corr).mean() > 0.8, f"mean |corr| {np.abs(corr).mean():.3f}"
