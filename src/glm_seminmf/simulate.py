"""Synthetic data from the NB-GLM semi-NMF generative process (spec section 6).

Pure numpy; no torch dependency. The validation suite fits the model to draws
from this generator and asserts recovery of the generating factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SimulatedData", "draw_nb_counts", "simulate_nb_seminmf"]


@dataclass
class SimulatedData:
    """A draw from the generative model. Shapes: X (p, n), F (p, k) with unit-L2
    columns, G (n, k) non-negative, a (p,), b (n,), theta (p,), and optionally
    Z (n, q) with gamma (p, q)."""

    X: np.ndarray
    F: np.ndarray
    G: np.ndarray
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    Z: np.ndarray | None = None
    gamma: np.ndarray | None = None
    batch: np.ndarray | None = None


def simulate_nb_seminmf(
    p: int = 2000,
    n: int = 1000,
    k: int = 8,
    *,
    density_F: float = 0.15,
    density_G: float = 0.4,
    negative_loading_fraction: float = 0.3,
    signal_strength: float = 1.0,
    dispersion: float = 10.0,
    dispersion_spread: float = 0.5,
    baseline_log_mean: float = 0.7,
    baseline_spread: float = 1.0,
    exposure_sd: float = 0.5,
    n_batches: int = 0,
    batch_strength: float = 1.0,
    F_fixed: np.ndarray | None = None,
    random_state: int | None = None,
) -> SimulatedData:
    """Draw ``X ~ NB(mu, theta)`` with ``log mu = a + b + Z gamma + F G^T``.

    Parameters
    ----------
    p, n, k
        Features, samples, latent factors. ``k=0`` gives null data
        (intercepts, exposure, and covariates only). Ignored for ``k`` when
        ``F_fixed`` is given: ``k`` is then taken from ``F_fixed.shape[1]``.
    density_F, density_G
        Fraction of nonzero entries per loading column / usage column.
        ``density_F`` is ignored when ``F_fixed`` is given.
    negative_loading_fraction
        Probability that a nonzero loading entry is negative. This is the knob
        that distinguishes the signed model from plain NMF.
    signal_strength
        Per-factor scale: each usage column is rescaled so the standard
        deviation of its contribution ``F_k G_k^T`` over that factor's active
        (feature, sample) pairs equals this many log-units.
    dispersion, dispersion_spread
        Median NB dispersion theta and lognormal spread across features.
        Larger theta = closer to Poisson.
    baseline_log_mean, baseline_spread
        Mean and sd of per-feature intercepts ``a``.
    exposure_sd
        SD of per-sample log-exposure ``b`` (centered at 0).
    n_batches, batch_strength
        If ``n_batches >= 2``, samples get random batch labels, ``Z`` is the
        one-hot design with the first batch as dropped reference, and ~30% of
        features receive N(0, batch_strength) batch coefficients.
    random_state
        Seed for a ``numpy.random.default_rng``.
    """
    rng = np.random.default_rng(random_state)

    if F_fixed is not None:
        F_fixed = np.asarray(F_fixed, dtype=np.float64)
        if F_fixed.shape[0] != p:
            raise ValueError(f"F_fixed has {F_fixed.shape[0]} rows, expected p={p}")
        k = F_fixed.shape[1]

    a = rng.normal(baseline_log_mean, baseline_spread, size=p)
    b = rng.normal(0.0, exposure_sd, size=n)
    theta = np.exp(np.log(dispersion) + rng.normal(0.0, dispersion_spread, size=p))

    F = F_fixed.copy() if F_fixed is not None else np.zeros((p, k))
    G = np.zeros((n, k))
    for j in range(k):
        if F_fixed is None:
            nnz_f = max(2, int(round(density_F * p)))
            support = rng.choice(p, size=nnz_f, replace=False)
            mags = rng.gamma(shape=2.0, scale=0.5, size=nnz_f) + 0.1
            signs = np.where(rng.random(nnz_f) < negative_loading_fraction, -1.0, 1.0)
            F[support, j] = signs * mags
            F[:, j] /= np.linalg.norm(F[:, j])
        else:
            # Caller's loading, taken as-is (not re-normalized): a shared
            # basis reused across many simulate_nb_seminmf calls — e.g. one
            # call per donor in a cohort simulator — must stay bit-identical
            # across calls, which re-normalizing per call would not guarantee
            # if F_fixed were ever mutated in place upstream (it isn't here,
            # since we copied above, but the contract should not depend on
            # renormalization being a no-op).
            support = np.flatnonzero(F[:, j])
            if support.size == 0:
                support = np.arange(p)  # dense/all-zero column: treat every feature as support

        active = rng.random(n) < density_G
        if not active.any():
            active[rng.integers(n)] = True
        G[active, j] = rng.gamma(shape=2.0, scale=1.0, size=int(active.sum()))

        # Rescale usages so the factor's contribution to eta has the requested
        # spread over its active (feature, sample) pairs.
        contrib = np.outer(F[support, j], G[active, j])
        sd = contrib.std()
        if sd > 0:
            G[:, j] *= signal_strength / sd

    Z = gamma = batch = None
    if n_batches >= 2:
        batch = rng.integers(n_batches, size=n)
        Z = np.eye(n_batches)[batch][:, 1:]  # drop reference level
        gamma = np.zeros((p, n_batches - 1))
        affected = rng.random(p) < 0.3
        gamma[affected] = rng.normal(0.0, batch_strength, size=(int(affected.sum()), n_batches - 1))

    X = draw_nb_counts(a, b, theta, F, G, Z=Z, gamma=gamma, rng=rng)

    return SimulatedData(X=X, F=F, G=G, a=a, b=b, theta=theta, Z=Z, gamma=gamma, batch=batch)


def draw_nb_counts(
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    *,
    Z: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``X ~ NB(mu, theta)`` from ``log mu = a + b + Z gamma^T + F G^T``.

    The generative final step of :func:`simulate_nb_seminmf`, factored out so a caller
    building non-uniform cohort structure (donor-varying ``b``/``G``/``Z`` sharing one
    fixed ``a``/``theta``/``F`` across many draws — the case ``simulate_nb_seminmf``
    itself does not support, since it redraws ``a``/``theta`` fresh every call even with
    ``F_fixed``) can reuse the exact same NB draw rather than reimplementing it.

    ``a``: (p,), ``b``: (n,), ``theta``: (p,), ``F``: (p, k), ``G``: (n, k),
    ``Z``: (n, q) or None, ``gamma``: (p, q) or None matching ``Z``.
    """
    eta = a[:, None] + b[None, :]
    if Z is not None:
        if gamma is None:
            raise ValueError("Z given without gamma")
        eta = eta + gamma @ Z.T
    if F.shape[1] > 0:
        eta = eta + F @ G.T
    eta = np.clip(eta, -30.0, 12.0)  # keep counts finite and simulation cheap
    mu = np.exp(eta)
    lam = rng.gamma(shape=theta[:, None], scale=mu / theta[:, None])
    return rng.poisson(lam).astype(np.int64)
