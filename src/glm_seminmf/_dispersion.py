"""Method-of-moments NB dispersion estimation with mean-dispersion trend shrinkage.

Estimation works from per-feature moment accumulators so callers can stream
over sample chunks without ever materializing a dense p x n mean matrix:

    ssr   = sum_s (x_fs - mu_fs)^2
    s_mu  = sum_s mu_fs
    s_mu2 = sum_s mu_fs^2

NB2 gives Var = mu + mu^2 / theta, so with alpha = 1/theta the moment estimate
per feature is  alpha = (ssr - s_mu) / s_mu2.

That estimate leaves the parameter space whenever a feature's residual spread
falls at or below its Poisson expectation, which happens by chance and does not
mean the feature is near-Poisson. Clamping such a feature to the alpha floor
turns it into theta ~ 1e4, and because NB curvature carries theta directly
(h = (theta + X) * sigmoid * sigmoid), a handful of them is enough to condition
the mean-model problem badly. Shrinkage weight is therefore set by how large a
feature's excess variance is relative to its own sampling scale, not by its
count total: a feature with plenty of counts but no resolvable excess takes the
trend, not the floor.
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

# Minimum number of features with a usable (positive) variance excess before a
# quadratic trend is worth fitting; below this the trend is the pooled estimate.
_MIN_TREND_FEATURES = 10


def _pooled_trend(ssr, s_mu, s_mu2):
    """Local moment pooling with conservative shrinkage toward a shared alpha.

    Pool *signed* excesses before enforcing positivity, avoiding selection of
    only noisy positive estimates at low means. Equal-population mean bins have
    at least 20 features (up to 20 bins). Uninformative bins borrow the global
    moment estimate; interpolation in log-alpha is flat beyond bin centers.

    The Poisson known-mean variance is a working weight, not a calibrated test
    for residuals from a fitted NB factor model. This remains an opt-in policy.
    """
    excess = ssr - s_mu
    pooled = (excess.sum() / s_mu2.sum().clamp(min=1e-12)).clamp(_ALPHA_MIN, _ALPHA_MAX)
    p = ssr.numel()
    n_bins = max(1, min(20, p // 20))
    log_mean = s_mu.clamp(min=1e-12).log()
    order = torch.argsort(log_mean, stable=True)
    def pool(value):
        # Fixed-order reductions avoid nondeterministic CUDA scatter atomics.
        return torch.stack([part.sum() for part in torch.tensor_split(value[order], n_bins)])
    centers = pool(log_mean) / pool(torch.ones_like(s_mu))
    denom = pool(s_mu2).clamp(min=1e-12)
    variance = (pool(s_mu) + 2 * denom) / denom.square()
    weight = pooled.square() / (pooled.square() + variance)
    bin_alpha = (weight * pool(excess) / denom + (1-weight) * pooled).clamp(_ALPHA_MIN, _ALPHA_MAX)
    if n_bins == 1:
        trend = bin_alpha.log().expand_as(log_mean)
    else:
        right = torch.searchsorted(centers.contiguous(), log_mean.contiguous()).clamp(1, n_bins-1)
        left = right-1
        frac = ((log_mean-centers[left]) / (centers[right]-centers[left]).clamp(min=1e-12)).clamp(0, 1)
        trend = (1-frac)*bin_alpha[left].log() + frac*bin_alpha[right].log()
    z = (excess / (s_mu+2*s_mu2).clamp(min=1e-12).sqrt()).clamp(min=0)
    w = (s_mu / (s_mu+_SHRINK_COUNTS)) * z.square() / (1+z.square())
    raw = (excess / s_mu2.clamp(min=1e-12)).clamp(_ALPHA_MIN, _ALPHA_MAX)
    return -(w*raw.log() + (1-w)*trend).clamp(min=pooled.new_tensor(_ALPHA_MIN).log(),
                                            max=pooled.new_tensor(_ALPHA_MAX).log())


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
    if mode == "trend_pooled":
        return _pooled_trend(ssr, s_mu, s_mu2)

    excess = ssr - s_mu
    alpha_raw = (excess / s_mu2.clamp(min=1e-12)).clamp(_ALPHA_MIN, _ALPHA_MAX)
    if mode == "feature":
        return -torch.log(alpha_raw)  # raw by contract, floor clamp included
    if mode != "trend":
        raise ValueError(f"unknown dispersion mode: {mode!r}")

    log_alpha = torch.log(alpha_raw)
    log_mean = torch.log(s_mu.clamp(min=1e-12))
    log_bounds = (float(torch.log(torch.tensor(_ALPHA_MIN))),
                  float(torch.log(torch.tensor(_ALPHA_MAX))))

    # A feature contributes to the trend only if its own estimate is inside the
    # parameter space; otherwise it sits at the clamp floor and drags the trend.
    usable = excess > 0
    m = log_mean - log_mean.mean()
    design = torch.stack([torch.ones_like(m), m, m * m], dim=1)
    if int(usable.sum()) >= _MIN_TREND_FEATURES:
        # Quadratic trend of log-alpha on log-mean, fitted on winsorized
        # responses so separation-level outlier features don't steer it.
        subset = log_alpha[usable]
        lo, hi = torch.quantile(subset, torch.tensor([0.01, 0.99], dtype=subset.dtype,
                                                     device=subset.device))
        resp = subset.clamp(lo, hi)
        d = design[usable]
        # Normal equations on the tiny 3x3 system; torch.linalg.lstsq is avoided
        # because its first call in a process can differ bitwise from later calls,
        # breaking seed reproducibility (spec 5.4).
        ata = d.T @ d + 1e-10 * torch.eye(3, dtype=d.dtype, device=d.device)
        trend = design @ torch.linalg.solve(ata, d.T @ resp)
    else:
        pooled = ((ssr.sum() - s_mu.sum()) / s_mu2.sum().clamp(min=1e-12)).clamp(
            _ALPHA_MIN, _ALPHA_MAX)
        trend = torch.log(pooled).expand_as(log_alpha)

    # Sampling scale of the excess: for counts near Poisson, var(ssr) is of
    # order 2 * sum(mu^2), so this is roughly a z statistic for "is there any
    # resolvable overdispersion here". It goes to zero at the boundary, which
    # is exactly where the moment estimate stops being usable.
    z = (excess / (2.0 * s_mu2).clamp(min=1e-12).sqrt()).clamp(min=0.0)
    w = (s_mu / (s_mu + _SHRINK_COUNTS)) * (z * z / (z * z + 1.0))
    return -(w * log_alpha + (1.0 - w) * trend).clamp(*log_bounds)
