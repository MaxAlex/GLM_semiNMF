"""Independent likelihood and upstream predictor checks for benchmark scoring."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import nbinom

# Benchmarks remain scripts, rather than part of the installed library API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from scoring import score_eta, subset_for_transform
from bench import _glmpca_eta


def test_common_theta_score_matches_independent_nb_logpmf():
    rng = np.random.default_rng(1)
    x = rng.poisson(.5, (12, 15))
    eta = rng.normal(-1, .5, x.shape)
    theta = np.exp(rng.normal(2, .2, 12))
    null_eta = np.full(x.shape, -1.)
    score = score_eta(x, eta, null_eta, theta)
    th = theta[:, None]
    expected = nbinom.logpmf(x, th, th/(th+np.exp(eta)))
    assert score['nll_per_entry'] == pytest.approx(-expected.mean(), abs=1e-12)
    assert score_eta(x, null_eta, null_eta, theta)['common_theta_deviance_explained'] == pytest.approx(0.)
    assert np.isfinite(score_eta(x, np.full(x.shape, -200.), null_eta, theta)['nll_per_entry'])


@pytest.mark.parametrize('explicit_size', [False, True])
def test_glmpca_predictor_matches_upstream_before_postprocessing(monkeypatch, explicit_size):
    upstream = pytest.importorskip('glmpca.glmpca')
    rng = np.random.default_rng(19)
    x = rng.poisson(3, (20, 30)).astype(float)
    size = np.exp(rng.normal(0, .3, 30)) if explicit_size else None
    original_init, original_ortho = upstream.glmpca_init, upstream.ortho
    captured = {}
    def capture_init(*args, **kwargs):
        result = original_init(*args, **kwargs)
        captured['rfunc'] = result['rfunc']
        return result
    def capture_ortho(U, V, A, X=None, G=None, Z=None):
        captured['eta'] = captured['rfunc'](np.hstack([X, U]), np.hstack([A, V])).copy()
        return original_ortho(U, V, A, X=X, G=G, Z=Z)
    monkeypatch.setattr(upstream, 'glmpca_init', capture_init)
    monkeypatch.setattr(upstream, 'ortho', capture_ortho)
    np.random.seed(0)
    result = upstream.glmpca(x, 2, fam='poi', sz=size, ctl={'maxIter': 5, 'eps': 1e-8})
    np.testing.assert_allclose(_glmpca_eta(result, x, size_factors=size), captured['eta'], atol=1e-12)
