"""`improved_on_init_`: catch the case where the optimizer hands back its warm start.

The proximal solver prevents the incoming branch's Adam divergence failure.
Still report a lack of progress when returning an uncertified warm start, without
warning when initialization is already certified stationary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from conftest import quick_model
from glm_seminmf import simulate_nb_seminmf


def test_normal_fit_improves_on_init():
    sim = simulate_nb_seminmf(p=200, n=300, k=3, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(3).fit(sim.X)
    assert m.improved_on_init_ is True
    assert m.loss_.min() < m.loss_[0]


def test_zero_budget_is_flagged_and_warns():
    """Exercise the guard deterministically without requiring optimizer failure."""
    sim = simulate_nb_seminmf(p=80, n=100, k=2, random_state=0)
    m = quick_model(2, max_iter=0)
    with pytest.warns(RuntimeWarning, match="never improved on its initialization"):
        m.fit(sim.X)
    assert m.improved_on_init_ is False
    assert m.best_iteration_ == 0
    assert m.final_objective_ == m.initial_objective_


def test_sparse_counts_improve_with_default_proximal_steps():
    """The sparse fixture that broke Adam must improve with backtracking."""
    sim = simulate_nb_seminmf(
        p=800, n=600, k=5, baseline_log_mean=-3.0, baseline_spread=0.8,
        dispersion=5.0, random_state=0,
    )
    assert (sim.X == 0).mean() > 0.85
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = quick_model(5, l1_F=0.001 * 600, max_iter=60).fit(sim.X)
    assert m.improved_on_init_ is True
    assert m.final_objective_ < m.initial_objective_
    assert not any("never improved" in str(w.message) for w in caught)


def test_stationary_initial_state_does_not_warn():
    # k=0 and constant counts yield an already optimal fixed-theta intercept.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = quick_model(0, dispersion=5.).fit(np.full((3, 8), 2))
    assert m.converged_
    assert m.improved_on_init_ is False
    assert not any("never improved" in str(w.message) for w in caught)


def test_flag_is_set_for_f_fixed_fits_too():
    sim = simulate_nb_seminmf(p=150, n=200, k=3, random_state=2)
    F0 = simulate_nb_seminmf(p=150, n=2, k=3, random_state=2).F
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(3).fit(sim.X, F_fixed=F0)
    assert isinstance(m.improved_on_init_, bool)
