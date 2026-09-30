"""Sparse and implicit SVD must accept ARPACK's column-vector callbacks."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

import glm_seminmf._init as init_module


@pytest.mark.parametrize("implicit", [False, True])
def test_svd_callbacks_preserve_vector_shapes_and_residualization(monkeypatch, implicit):
    rng = np.random.default_rng(8)
    y = rng.normal(size=(12, 17))
    z = rng.normal(size=(17, 2))
    z -= z.mean(axis=0)
    gamma = rng.normal(size=(12, 2))
    centered = y - y.mean(axis=1, keepdims=True) - gamma @ z.T
    original = init_module.svds

    def checked_svds(operator, **kwargs):
        v, u = rng.normal(size=(17, 1)), rng.normal(size=(12, 1))
        np.testing.assert_allclose(operator.matvec(v), centered @ v, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvec(u), centered.T @ u, atol=1e-12)
        return original(operator, **kwargs)

    monkeypatch.setattr(init_module, "svds", checked_svds)
    source = aslinearoperator(y) if implicit else sp.csc_matrix(y)
    f, g = init_module._svd_init(source, 3, rng, gamma, z)
    assert f.shape == (12, 3) and g.shape == (17, 3)
    assert np.isfinite(f).all() and np.isfinite(g).all()
    assert (g >= 0).all()
