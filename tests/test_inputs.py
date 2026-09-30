"""Input validation: non-negative counts, covariate encoding."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from glm_seminmf._inputs import encode_Z, validate_X


def test_accepts_non_integer_counts():
    """Ambient correction (CellBender) emits fractional posterior means.

    The NB2 likelihood is defined for any real x >= 0 -- the only x-dependent gamma term is
    lgamma(x + 1) and the deviance uses xlogy -- so nothing in the fitter counts events.
    """
    X = np.random.default_rng(0).gamma(2.0, scale=60.0, size=(10, 8))
    validate_X(X)


def test_accepts_non_integer_sparse():
    X = sp.random(20, 15, density=0.3, random_state=0, format="csr") * 370.0
    validate_X(X)


def test_rejects_normalized_looking_input():
    """Fractional *and* small-ranged still raises: that is log1p/CPM, not counts."""
    rng = np.random.default_rng(0)
    X = np.log1p(rng.poisson(3.0, size=(50, 60)).astype(float))
    with pytest.raises(ValueError, match="looks normalized"):
        validate_X(X)


def test_non_integer_counts_fit_end_to_end():
    from glm_seminmf import NBGLMSemiNMF
    from glm_seminmf.simulate import simulate_nb_seminmf

    sim = simulate_nb_seminmf(p=120, n=300, k=3, random_state=0)
    X = np.asarray(sim.X.todense() if hasattr(sim.X, "todense") else sim.X, dtype=np.float64)
    model = NBGLMSemiNMF(n_components=3, max_iter=60, random_state=0).fit(X * 0.93 + 0.07)
    assert np.isfinite(model.F_).all() and np.isfinite(model.G_).all()
    assert (model.G_ >= 0).all()


def test_rejects_negative_counts():
    X = np.array([[1, 2], [-1, 0]])
    with pytest.raises(ValueError, match="non-negative"):
        validate_X(X)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("sparse", [False, True])
def test_rejects_nonfinite_counts(value, sparse):
    X = np.array([[1., 40.], [value, 0.]])
    if sparse:
        X = sp.csc_matrix(X)
    with pytest.raises(ValueError, match="finite counts"):
        validate_X(X)


def test_accepts_integer_valued_floats():
    X = np.array([[1.0, 2.0], [0.0, 5.0]])
    validate_X(X)  # whole-valued floats are integer counts


def test_small_sparse_densified():
    X = sp.random(20, 15, density=0.3, random_state=0, format="csr")
    X.data = np.round(X.data * 10)
    out, p, n = validate_X(X)
    # small sparse inputs are densified (single code path => bitwise
    # agreement with dense); large ones stay CSC — see _DENSIFY_ELEMENTS
    assert isinstance(out, np.ndarray) and out.flags["C_CONTIGUOUS"]
    assert (p, n) == (20, 15)


def test_large_sparse_stays_csc(monkeypatch):
    import glm_seminmf._inputs as inputs

    monkeypatch.setattr(inputs, "_DENSIFY_ELEMENTS", 0)
    X = sp.random(20, 15, density=0.3, random_state=0, format="csr")
    X.data = np.round(X.data * 10)
    out, _, _ = validate_X(X)
    assert sp.issparse(out) and out.format == "csc"


def test_encode_Z_dataframe_one_hot():
    df = pd.DataFrame({"batch": ["a", "b", "c", "a"], "depth": [0.1, 0.2, 0.3, 0.4]})
    Z, names = encode_Z(df, 4)
    assert Z.shape == (4, 3)  # depth + 2 batch indicators (reference dropped)
    assert any("batch" in nm for nm in names)


def test_encode_Z_reindexes_to_fit_columns():
    df_fit = pd.DataFrame({"batch": ["a", "b", "c"]})
    _, names = encode_Z(df_fit, 3)
    df_new = pd.DataFrame({"batch": ["a", "a", "b"]})  # level "c" absent
    Z, _ = encode_Z(df_new, 3, columns=names)
    assert Z.shape == (3, len(names))


def test_encode_Z_wrong_rows_raises():
    with pytest.raises(ValueError, match="rows"):
        encode_Z(np.zeros((5, 2)), 7)
