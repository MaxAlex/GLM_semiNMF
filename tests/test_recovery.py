"""Spec tests 1-4: recovery, signed recovery, sign-ambiguity absence,
rotation stability. The correctness criterion is recovering the generating
factors, not loss decrease."""

import itertools

import numpy as np
import pytest

from conftest import match_factors, quick_model
from glm_seminmf import simulate_nb_seminmf

K = 4


@pytest.mark.parametrize(
    "dispersion,density_F",
    list(itertools.product([3.0, 20.0], [0.1, 0.3])),
)
def test_recovery_grid(dispersion, density_F):
    """Test 1: mean matched |loading corr| above threshold across a grid of
    noise (dispersion) and sparsity levels."""
    sim = simulate_nb_seminmf(
        p=300, n=500, k=K, dispersion=dispersion, density_F=density_F, random_state=1
    )
    m = quick_model(K).fit(sim.X)
    corr, _ = match_factors(sim.F, m.F_)
    assert np.abs(corr).mean() > 0.85, f"mean |corr| {np.abs(corr).mean():.3f}"


def test_signed_recovery():
    """Test 2 (strict): negative loadings recovered with correct sign and
    approximately correct magnitude. This is what distinguishes the model
    from non-negative factorization."""
    sim = simulate_nb_seminmf(
        p=300, n=600, k=K, negative_loading_fraction=0.4, random_state=2,
        baseline_log_mean=2.0, signal_strength=0.7,
    )
    # Keep suppressed features observable and condition on the generating
    # nuisance parameters: this tests signed-factor recovery, not MoM bias or
    # count-total exposure approximation. The low-count/default-nuisance
    # recovery grid above remains a separate empirical check.
    m = quick_model(K, dispersion=sim.theta, exposure=sim.b).fit(sim.X)
    assert m.converged_, m.stationarity_
    corr, cols = match_factors(sim.F, m.F_)
    assert (corr > 0.85).all(), f"signed corr {corr.round(3)}"

    for j_true, j_est in enumerate(cols):
        f_true, f_est = sim.F[:, j_true], m.F_[:, j_est]
        strong_neg = f_true < -3.0 / np.sqrt(sim.F.shape[0])
        assert strong_neg.any()
        # sign agreement on clearly negative true loadings
        frac_neg_recovered = (f_est[strong_neg] < 0).mean()
        assert frac_neg_recovered > 0.9, f"factor {j_true}: {frac_neg_recovered:.2f}"
        # magnitude approximately correct on the negative part (median ratio;
        # a regression slope is meaningless when the few strong-negative true
        # values are nearly identical)
        ratio = np.median(np.abs(f_est[strong_neg]) / np.abs(f_true[strong_neg]))
        assert 0.4 < ratio < 2.5, f"factor {j_true}: negative magnitude ratio {ratio:.2f}"


def test_no_sign_flips_across_restarts():
    """Test 3: whenever two restarts recover the same factor (high |corr|),
    the match is never sign-flipped; G >= 0 must eliminate the sign ambiguity.

    Weakly matched pairs (different local optima under poor inits) are not the
    same factor, so their sign carries no information; the assertion is over
    genuinely matched pairs, with a non-vacuity guard.
    """
    sim = simulate_nb_seminmf(p=250, n=400, k=K, negative_loading_fraction=0.35, random_state=3)
    fits = [quick_model(K, random_state=s).fit(sim.X) for s in range(3)]
    fits += [quick_model(K, random_state=s, init="random").fit(sim.X) for s in range(2)]
    matched_corrs = []
    for m1, m2 in itertools.combinations(fits, 2):
        corr, _ = match_factors(m1.F_, m2.F_)
        matched_corrs.extend(corr[np.abs(corr) > 0.8])
    matched_corrs = np.array(matched_corrs)
    assert len(matched_corrs) >= 10, "too few well-matched pairs to test sign stability"
    assert (matched_corrs > 0).all(), f"sign flip among matched factors: {matched_corrs.round(3)}"


def test_rotation_stability_across_seeds():
    """Test 4: across restarts on the same data, matched loading correlations
    exceed threshold. The distribution is reported via -rA output."""
    sim = simulate_nb_seminmf(p=250, n=400, k=K, random_state=4)
    fits = [quick_model(K, random_state=s).fit(sim.X) for s in range(4)]
    all_corrs = []
    for m1, m2 in itertools.combinations(fits, 2):
        corr, _ = match_factors(m1.F_, m2.F_)
        all_corrs.extend(np.abs(corr))
    all_corrs = np.array(all_corrs)
    print(
        f"\nrotation stability |corr|: min={all_corrs.min():.3f} "
        f"median={np.median(all_corrs):.3f} mean={all_corrs.mean():.3f}"
    )
    assert all_corrs.mean() > 0.9
    assert all_corrs.min() > 0.7
