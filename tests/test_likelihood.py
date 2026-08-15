"""Spec test 10 (gradient check) plus likelihood-level correctness."""

import numpy as np
import scipy.stats
import torch

from glm_seminmf._likelihood import nb_deviance, nb_nll, softplus_inv


def test_nll_matches_scipy_nbinom():
    rng = np.random.default_rng(0)
    x = rng.poisson(5.0, size=(40, 30)).astype(np.float64)
    eta = rng.normal(1.0, 1.5, size=(40, 30))
    theta = np.exp(rng.normal(1.0, 1.0, size=(40, 1)))
    mu = np.exp(eta)
    # scipy NB parametrization: n = theta, p = theta/(theta+mu)
    ref = -scipy.stats.nbinom.logpmf(x, theta, theta / (theta + mu))
    got = nb_nll(
        torch.tensor(x), torch.tensor(eta), torch.tensor(np.log(theta)), full=True
    ).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-8)


def test_nll_extreme_eta_is_finite():
    x = torch.tensor([[0.0, 3.0, 1000.0]])
    for eta_val in (-200.0, -30.0, 0.0, 30.0, 200.0):
        eta = torch.full((1, 3), eta_val, requires_grad=True)
        nll = nb_nll(x, eta, torch.tensor([[0.5]]), full=True).sum()
        assert torch.isfinite(nll)
        nll.backward()
        assert torch.isfinite(eta.grad).all()


def test_gradient_finite_difference():
    """Finite-difference the analytic (autograd) gradient of each block."""
    torch.manual_seed(0)
    p, n, k = 8, 6, 2
    x = torch.poisson(torch.full((p, n), 4.0)).double()
    log_theta = torch.randn(p, 1).double() * 0.5

    params = {
        "F": torch.randn(p, k).double() * 0.3,
        "G_raw": torch.randn(n, k).double(),
        "a": torch.randn(p).double() * 0.5,
        "b": torch.randn(n).double() * 0.3,
    }

    def objective(pr):
        eta = (
            pr["a"].unsqueeze(1)
            + pr["b"].unsqueeze(0)
            + pr["F"] @ torch.nn.functional.softplus(pr["G_raw"]).T
        )
        return nb_nll(x, eta, log_theta).sum()

    for name in params:
        prm = params[name].clone().requires_grad_(True)
        pr = {**params, name: prm}
        objective(pr).backward()
        analytic = prm.grad.clone()

        eps = 1e-6
        flat = params[name].flatten()
        for idx in range(min(flat.numel(), 12)):
            for sign, store in ((1, "hi"), (-1, "lo")):
                pert = params[name].clone().flatten()
                pert[idx] += sign * eps
                pr = {**params, name: pert.view_as(params[name])}
                if sign == 1:
                    hi = objective(pr).item()
                else:
                    lo = objective(pr).item()
            fd = (hi - lo) / (2 * eps)
            np.testing.assert_allclose(
                analytic.flatten()[idx].item(), fd, rtol=1e-5, atol=1e-4,
                err_msg=f"block {name}, coordinate {idx}",
            )


def test_softplus_inv_roundtrip():
    y = torch.tensor([1e-8, 1e-3, 0.5, 5.0, 25.0, 500.0], dtype=torch.float64)
    back = torch.nn.functional.softplus(softplus_inv(y))
    np.testing.assert_allclose(back.numpy(), y.numpy(), rtol=1e-9)


def test_deviance_zero_at_saturation():
    x = torch.tensor([[0.0, 2.0, 7.0]])
    theta = torch.tensor([[3.0]])
    dev = nb_deviance(x, x.clamp(min=1e-12), theta)
    np.testing.assert_allclose(dev.numpy(), 0.0, atol=1e-6)
    # deviance grows as mu moves away from x
    worse = nb_deviance(x, x + 2.0, theta)
    assert (worse.numpy() > 0).all()
