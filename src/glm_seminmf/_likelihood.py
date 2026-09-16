"""Numerically stabilized negative-binomial (NB2) likelihood pieces.

All functions operate on torch tensors, are dtype/device agnostic, and follow
the model orientation: features x samples. ``eta`` is the linear predictor
(log mean); ``log_theta`` is the log of the per-feature dispersion, shaped to
broadcast against ``eta`` (typically ``(p, 1)``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as tf

# Linear-predictor clamp (spec 5.1): mu is confined to [exp(-30), exp(30)].
ETA_CLAMP = 30.0


def st_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp with straight-through gradient.

    Forward pass is ``clamp(x, lo, hi)``; the gradient is passed through as if
    the clamp were the identity. A hard clamp would zero the gradient of any
    entry outside the box, leaving it no signal to re-enter; straight-through
    keeps the (inward-pointing) NLL gradient alive.
    """
    return x + (x.clamp(lo, hi) - x).detach()


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Inverse softplus, ``log(expm1(y))``, stable at both ends. Requires y > 0."""
    tiny = torch.finfo(y.dtype).tiny
    return torch.where(
        y > 20.0,
        y + torch.log1p(-torch.exp(-y.clamp(min=1.0))),
        torch.log(torch.expm1(y.clamp(min=tiny, max=20.0))),
    )


def nb_nll(
    x: torch.Tensor,
    eta: torch.Tensor,
    log_theta: torch.Tensor,
    *,
    full: bool = False,
) -> torch.Tensor:
    """Elementwise NB2 negative log-likelihood at linear predictor ``eta``.

    Uses stable softplus forms without clamping the predictor. The backward
    derivative is the derivative of the reported objective, including tails.

    With ``full=False`` the ``lgamma`` terms (constant in ``eta``) are dropped:
    fine inside mean-parameter blocks, wrong for comparing losses across
    dispersion updates — use ``full=True`` for the reported loss trace.
    """
    theta = torch.exp(log_theta)
    # This exact stable expression has no straight-through clamp. Both tails
    # stay differentiable, and x=0 does not require evaluating exp(eta).
    delta = eta - log_theta
    nll = theta * tf.softplus(delta) + x * tf.softplus(-delta)
    if full:
        nll = nll - torch.lgamma(x + theta) + torch.lgamma(theta) + torch.lgamma(x + 1.0)
    return nll


def nb_deviance(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Elementwise NB2 deviance, ``2*(loglik_saturated - loglik_model)``.

    The ``lgamma`` terms cancel between saturated and fitted models, so this
    needs only means. ``theta`` broadcasts against ``x``/``mu``.
    """
    xlog = torch.xlogy(x, x) - torch.xlogy(x, mu)  # x*log(x/mu), zero at x == 0
    tail = (x + theta) * (torch.log(x + theta) - torch.log(mu + theta))
    return 2.0 * (xlog - tail)
