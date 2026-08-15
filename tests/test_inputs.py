"""Input validation: raw integer counts only, covariate encoding."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from glm_seminmf._inputs import encode_Z, validate_X


def test_rejects_non_integer_counts():
    X = np.random.default_rng(0).gamma(2.0, size=(10, 8))
    with pytest.raises(ValueError, match="integer"):
        validate_X(X)


def test_rejects_non_integer_sparse():
    X = sp.random(20, 15, density=0.3, random_state=0, format="csr") * 3.7
    with pytest.raises(ValueError, match="integer"):
        validate_X(X)


def test_rejects_negative_counts():
    X = np.array([[1, 2], [-1, 0]])
    with pytest.raises(ValueError, match="non-negative"):
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
