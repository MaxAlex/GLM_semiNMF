import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from glm_seminmf import NBGLMSemiNMF

# Suite runs many small fits across xdist workers; per-process BLAS/torch
# thread pools oversubscribe the box and slow everything down.
torch.set_num_threads(2)


def match_factors(F_true: np.ndarray, F_est: np.ndarray):
    """Hungarian-match estimated factors to true ones by |loading corr|.

    Returns (signed_corrs, est_indices) for the true factors in order.
    """
    kt, ke = F_true.shape[1], F_est.shape[1]
    C = np.corrcoef(F_true.T, F_est.T)[:kt, kt:]
    rows, cols = linear_sum_assignment(-np.abs(C))
    return C[rows, cols], cols


def quick_model(k: int, **kw) -> NBGLMSemiNMF:
    """Small-problem defaults used across the suite (CPU, fast, seeded)."""
    args = dict(
        n_components=k,
        l1_F=1.0,
        random_state=0,
        device="cpu",
        max_iter=400,
    )
    args.update(kw)
    return NBGLMSemiNMF(**args)


@pytest.fixture(autouse=True)
def _quiet_convergence_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*did not converge.*")
        yield
