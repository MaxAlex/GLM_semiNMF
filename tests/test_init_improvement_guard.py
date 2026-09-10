"""`improved_on_init_`: catch the case where the optimizer hands back its warm start.

If the step size is too large for the data, the objective can rise immediately and
never recover, so the best iterate is iteration 0. run_fit's best-iterate restore then
returns the initialization, with a plausible deviance_explained and stable-looking
results across max_iter values. Nothing downstream can detect that, so the model
reports it.
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


def test_sparse_counts_at_default_step_size_are_flagged_and_warn():
    """The real failure mode: sparse counts, default learning rate.

    At ~92% zeros and ~70 counts per sample the default step size is too large for
    the likelihood's curvature — the objective rises immediately and never returns
    below its first value, so the "fit" is the warm start. This is the shape of real
    scRNA-seq data restricted to a few thousand selected genes, where a cell can carry
    only ~100 counts across the fitted gene set.
    """
    sim = simulate_nb_seminmf(
        p=800, n=600, k=5, baseline_log_mean=-3.0, baseline_spread=0.8,
        dispersion=5.0, random_state=0,
    )
    assert (sim.X == 0).mean() > 0.85, "fixture should be sparse enough to trigger this"

    m = quick_model(5, l1_F=0.001 * 600, max_iter=60)
    with pytest.warns(RuntimeWarning, match="never improved on its initialization"):
        m.fit(sim.X)
    assert m.improved_on_init_ is False
    # The tell: the returned state is the first iterate, so the minimum is at index 0.
    assert int(np.argmin(m.loss_[:-1])) == 0


def test_lower_learning_rate_recovers_on_the_same_sparse_data():
    """And the documented remedy actually works — otherwise the warning's advice
    ("lower learning_rate rather than raising max_iter") would be wrong."""
    sim = simulate_nb_seminmf(
        p=800, n=600, k=5, baseline_log_mean=-3.0, baseline_spread=0.8,
        dispersion=5.0, random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(5, l1_F=0.001 * 600, max_iter=200, learning_rate=1e-4).fit(sim.X)
    assert m.improved_on_init_ is True


def test_flag_is_set_for_f_fixed_fits_too():
    sim = simulate_nb_seminmf(p=150, n=200, k=3, random_state=2)
    F0 = simulate_nb_seminmf(p=150, n=2, k=3, random_state=2).F
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = quick_model(3).fit(sim.X, F_fixed=F0)
    assert isinstance(m.improved_on_init_, bool)
