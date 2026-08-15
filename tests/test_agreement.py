"""Spec test 11 and 5.4: sparse/dense agreement (exact), CPU/GPU agreement
(tolerance), bit-reproducibility under a fixed seed."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from conftest import match_factors, quick_model
from glm_seminmf import simulate_nb_seminmf


def _sim_sparse():
    sim = simulate_nb_seminmf(
        p=200, n=300, k=3, baseline_log_mean=-1.5, random_state=11
    )
    return sim, sp.csr_matrix(sim.X)


def test_sparse_dense_exact_agreement():
    sim, Xs = _sim_sparse()
    m_dense = quick_model(3, max_iter=60).fit(sim.X)
    m_sparse = quick_model(3, max_iter=60).fit(Xs)
    np.testing.assert_array_equal(m_dense.F_, m_sparse.F_)
    np.testing.assert_array_equal(m_dense.G_, m_sparse.G_)
    np.testing.assert_array_equal(m_dense.theta_, m_sparse.theta_)


def test_sparse_streaming_path_agrees(monkeypatch):
    """Force the large-sparse streaming path (no densification, no resident
    device copy) and check it matches the dense path to float tolerance."""
    import glm_seminmf._fitting as ft
    import glm_seminmf._init as init
    import glm_seminmf._inputs as inputs

    sim, Xs = _sim_sparse()
    m_dense = quick_model(3, max_iter=60).fit(sim.X)

    monkeypatch.setattr(inputs, "_DENSIFY_ELEMENTS", 0)
    monkeypatch.setattr(init, "_DENSIFY_ELEMENTS", 0)
    monkeypatch.setattr(ft, "_residency_limit", lambda device, dtype: 0)
    monkeypatch.setattr(ft, "_CHUNK_ELEMENTS", 200 * 64)
    m_sparse = quick_model(3, max_iter=60).fit(Xs)

    corr, _ = match_factors(m_dense.F_, m_sparse.F_)
    assert (np.abs(corr) > 0.999).all(), corr


def test_bit_reproducibility_same_seed():
    sim, _ = _sim_sparse()
    m1 = quick_model(3, max_iter=60, init="random").fit(sim.X)
    m2 = quick_model(3, max_iter=60, init="random").fit(sim.X)
    np.testing.assert_array_equal(m1.F_, m2.F_)
    np.testing.assert_array_equal(m1.G_raw_, m2.G_raw_)
    np.testing.assert_array_equal(m1.loss_, m2.loss_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_agreement():
    sim, _ = _sim_sparse()
    m_cpu = quick_model(3).fit(sim.X)
    m_gpu = quick_model(3, device="cuda").fit(sim.X)
    corr, _ = match_factors(m_cpu.F_, m_gpu.F_)
    assert (np.abs(corr) > 0.98).all(), corr.round(4)
    # final losses agree to float32 optimization tolerance
    rel = abs(m_cpu.loss_[-1] - m_gpu.loss_[-1]) / abs(m_cpu.loss_[-1])
    assert rel < 1e-3, rel


def test_chunked_matches_unchunked():
    """batch_size only bounds memory; results must be identical up to
    float accumulation order."""
    sim, _ = _sim_sparse()
    m_full = quick_model(3, max_iter=60).fit(sim.X)
    m_chunk = quick_model(3, max_iter=60, batch_size=64).fit(sim.X)
    corr, _ = match_factors(m_full.F_, m_chunk.F_)
    assert (np.abs(corr) > 0.999).all(), corr
