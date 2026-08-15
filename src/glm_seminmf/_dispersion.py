"""Method-of-moments NB dispersion estimation with mean-dispersion trend shrinkage.

Estimation works from per-feature moment accumulators so callers can stream
over sample chunks without ever materializing a dense p x n mean matrix:

    ssr   = sum_s (x_fs - mu_fs)^2
    s_mu  = sum_s mu_fs
    s_mu2 = sum_s mu_fs^2

NB2 gives Var = mu + mu^2 / theta, so with alpha = 1/theta the moment estimate
per feature is  alpha = (ssr - s_mu) / s_mu2.
"""

from __future__ import annotations

import torch

# alpha = 1/theta clamp: theta confined to [1e-3, 1e4] (1e4 ~ effectively Poisson).
_ALPHA_MIN = 1e-4
_ALPHA_MAX = 1e3

# Shrinkage midpoint: a feature with ~this many total expected counts gets
# half-weight on its own estimate, half on the trend. Heuristic; low-count
# features are the ones whose raw moment estimate is noise (spec section 3).
_SHRINK_COUNTS = 50.0


def dispersion_from_moments(
    ssr: torch.Tensor,
    s_mu: torch.Tensor,
    s_mu2: torch.Tensor,
    mode: str | float,
) -> torch.Tensor:
    """Return per-feature ``log_theta`` of shape ``(p,)`` from moment accumulators.

    ``mode``: ``"trend"`` (per-feature, shrunk toward a quadratic
    log-alpha-vs-log-mean trend), ``"feature"`` (raw per-feature),
    ``"shared"`` (single pooled value), or a float (fixed theta).
    """
    if isinstance(mode, (int, float)):
        val = torch.as_tensor(float(mode), dtype=ssr.dtype, device=ssr.device)
        return torch.log(val).expand(ssr.shape[0]).clone()

    if mode == "shared":
        alpha = (ssr.sum() - s_mu.sum()) / s_mu2.sum().clamp(min=1e-12)
        alpha = alpha.clamp(_ALPHA_MIN, _ALPHA_MAX)
        return (-torch.log(alpha)).expand(ssr.shape[0]).clone()

    alpha_raw = ((ssr - s_mu) / s_mu2.clamp(min=1e-12)).clamp(_ALPHA_MIN, _ALPHA_MAX)
    if mode == "feature":
        return -torch.log(alpha_raw)
    if mode != "trend":
        raise ValueError(f"unknown dispersion mode: {mode!r}")

    log_alpha = torch.log(alpha_raw)
    log_mean = torch.log(s_mu.clamp(min=1e-12))

    # Quadratic trend of log-alpha on log-mean, fitted on winsorized responses
    # so separation-level outlier features don't steer the trend.
    lo, hi = torch.quantile(log_alpha, torch.tensor([0.01, 0.99], dtype=log_alpha.dtype, device=log_alpha.device))
    resp = log_alpha.clamp(lo, hi)
    m = log_mean - log_mean.mean()
    design = torch.stack([torch.ones_like(m), m, m * m], dim=1)
    # Normal equations on the tiny 3x3 system; torch.linalg.lstsq is avoided
    # because its first call in a process can differ bitwise from later calls,
    # breaking seed reproducibility (spec 5.4).
    ata = design.T @ design + 1e-10 * torch.eye(3, dtype=design.dtype, device=design.device)
    coef = torch.linalg.solve(ata, design.T @ resp)
    trend = design @ coef

    w = s_mu / (s_mu + _SHRINK_COUNTS)  # information-based shrink weight
    log_alpha_shrunk = w * log_alpha + (1.0 - w) * trend
    return -log_alpha_shrunk.clamp(
        torch.log(torch.tensor(_ALPHA_MIN, dtype=log_alpha.dtype, device=log_alpha.device)),
        torch.log(torch.tensor(_ALPHA_MAX, dtype=log_alpha.dtype, device=log_alpha.device)),
    )
