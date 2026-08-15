"""Block-alternating optimization of the NB-GLM semi-NMF objective.

Data streams through sample (column) chunks so a dense p x n mean matrix is
never materialized; gradients are accumulated across chunks before each
optimizer step, so semantics are full-batch regardless of chunk size
(``batch_size`` controls memory, not stochasticity).

Blocks: G-block updates (G_raw [, b]) with everything else frozen; F-block
updates (F, a [, gamma]). The L1 penalty on F is applied as a proximal
soft-threshold scaled by Adam's per-coordinate step size; each factor's
contribution to the log-mean is hard-clipped at ``F_CLIP`` as the separation
backstop (spec 5.2). A small L1 on G (``FitConfig.lam_G``) anchors the
usage-baseline flat direction — see ``rescale_columns``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as tf

from ._dispersion import dispersion_from_moments
from ._likelihood import ETA_CLAMP, nb_deviance, nb_nll, softplus_inv, st_clamp

# Separation backstop (spec 5.2): a factor's contribution to the linear
# predictor is hard-clipped at |F_fk| * max_s G_sk <= F_CLIP (log-fold change
# at maximal activity). F columns are kept unit-L2 by rescaling, so a bound on
# raw |F| would be unreachable; the contribution bound is scale-invariant.
F_CLIP = 15.0
# Auto-chunking: ~this many matrix elements per streamed chunk (float32:
# ~128 MB forward, a few x that with autograd buffers).
_CHUNK_ELEMENTS = 32_000_000
# Fallback residency cap (elements) when device memory cannot be queried.
_RESIDENT_ELEMENTS = 200_000_000


def _residency_limit(device: torch.device, dtype: torch.dtype) -> int:
    """Max p*n elements to keep resident on the device. Streaming instead of
    residency costs a full host-to-device copy of X per gradient pass, so be
    as generous as free memory allows (chunked compute still bounds the
    autograd working set)."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    if device.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(device)
            return int(0.35 * free / itemsize)
        except Exception:  # pragma: no cover - driver-dependent
            return _RESIDENT_ELEMENTS
    return 1_000_000_000  # CPU: ~4 GB float32; chunk conversion dominates otherwise


class DataSource:
    """Iterates ``(start, end, x_chunk, z_chunk)`` over contiguous sample ranges.

    ``x_chunk`` is a dense (p, m) tensor on the compute device. Sparse X stays
    sparse on the host; only the current chunk is densified.
    """

    def __init__(self, X, Z: np.ndarray | None, device, dtype, batch_size: int | None):
        self.p, self.n = X.shape
        self.device, self.dtype = device, dtype
        auto = max(1, _CHUNK_ELEMENTS // max(self.p, 1))
        self.chunk = min(self.n, batch_size if batch_size else auto)
        self.Z = None if Z is None else torch.as_tensor(Z, device=device, dtype=dtype)
        self._sparse = sp.issparse(X)
        self._X_host = X
        self._resident = None
        if self.p * self.n <= _residency_limit(device, dtype):
            dense = X.toarray() if self._sparse else np.asarray(X)
            self._resident = torch.as_tensor(dense, device=device, dtype=dtype)
        elif not self._sparse:
            # Streaming path: stage the host copy in the compute dtype once so
            # each pass pays only the transfer, not an int -> float conversion.
            np_dtype = np.float32 if dtype == torch.float32 else np.float64
            if X.dtype != np_dtype:
                self._X_host = np.ascontiguousarray(X, dtype=np_dtype)

    def __iter__(self):
        for s in range(0, self.n, self.chunk):
            e = min(self.n, s + self.chunk)
            if self._resident is not None:
                x = self._resident[:, s:e]
            else:
                xc = self._X_host[:, s:e]
                xc = xc.toarray() if self._sparse else np.ascontiguousarray(xc)
                x = torch.as_tensor(xc, device=self.device, dtype=self.dtype)
            z = None if self.Z is None else self.Z[s:e]
            yield s, e, x, z



@dataclass
class FitState:
    """Current parameter tensors. G is represented pre-softplus (``softplus``
    mode) or directly with projection to the orthant (``projected`` mode)."""

    F: torch.Tensor  # (p, k) signed
    G_raw: torch.Tensor  # (n, k)
    a: torch.Tensor  # (p,)
    b: torch.Tensor  # (n,)
    gamma: torch.Tensor | None  # (p, q)
    log_theta: torch.Tensor  # (p,)
    b_trainable: bool
    g_param: str = "softplus"

    def G(self, rows: slice = slice(None)) -> torch.Tensor:
        g = self.G_raw[rows]
        return tf.softplus(g) if self.g_param == "softplus" else g

    def eta(self, s: int, e: int, z: torch.Tensor | None) -> torch.Tensor:
        eta = self.a.unsqueeze(1) + self.b[s:e].unsqueeze(0)
        if self.gamma is not None and z is not None:
            eta = eta + self.gamma @ z.T
        if self.F.shape[1] > 0:
            eta = eta + self.F @ self.G(slice(s, e)).T
        return eta

    def mu(self, s: int, e: int, z: torch.Tensor | None) -> torch.Tensor:
        return torch.exp(self.eta(s, e, z).clamp(-ETA_CLAMP, ETA_CLAMP))


@dataclass
class FitConfig:
    lam: float = 0.0
    # Small L1 on G (sum of usages). eta is invariant under G_k -> G_k + c
    # with a -> a - c F_k, so the likelihood alone leaves each usage column's
    # baseline free; this penalty smoothly selects the touch-zero
    # representative, keeps the orthant constraint active, and measurably
    # improves factor recovery. Scaled ~ p by the caller.
    lam_G: float = 0.0
    max_iter: int = 500
    tol: float = 1e-5
    algorithm: str = "adam"  # "adam" | "lbfgs" | "adam_joint"
    lr_G: float = 0.05
    lr_F: float = 0.05
    inner_steps: int = 5
    lbfgs_inner: int = 8
    theta_every: int = 10
    dispersion_mode: str | float = "trend"
    update_theta: bool = True
    lr_decay: float = 0.5
    lr_patience: int = 10
    verbose: bool = False


def _set_active(active: list[torch.Tensor], all_params: list[torch.Tensor]) -> None:
    for prm in all_params:
        prm.requires_grad_(any(prm is x for x in active))


def full_objective(state: FitState, data: DataSource, lam: float, lam_G: float = 0.0) -> float:
    """Penalized objective with the full NLL (lgamma terms included)."""
    total = 0.0
    with torch.no_grad():
        for s, e, x, z in data:
            total += nb_nll(x, state.eta(s, e, z), state.log_theta.unsqueeze(1), full=True).sum(
                dtype=torch.float64
            ).item()
        total += lam * state.F.abs().sum(dtype=torch.float64).item()
        if lam_G > 0 and state.F.shape[1] > 0:
            total += lam_G * state.G(slice(None)).sum(dtype=torch.float64).item()
    return total


def total_deviance(state: FitState, data: DataSource, drop_factor: int | None = None) -> float:
    """Summed NB deviance; optionally with one factor's contribution removed
    (no refit — a pure ablation of that factor's term in the predictor)."""
    total = 0.0
    theta = torch.exp(state.log_theta).unsqueeze(1)
    with torch.no_grad():
        for s, e, x, z in data:
            eta = state.eta(s, e, z)
            if drop_factor is not None:
                eta = eta - torch.outer(state.F[:, drop_factor], state.G(slice(s, e))[:, drop_factor])
            mu = torch.exp(eta.clamp(-ETA_CLAMP, ETA_CLAMP))
            total += nb_deviance(x, mu, theta).sum(dtype=torch.float64).item()
    return total


def refresh_dispersion(state: FitState, data: DataSource, mode: str | float) -> None:
    """Method-of-moments update of log_theta from current residuals."""
    if isinstance(mode, (int, float)):
        state.log_theta.fill_(float(np.log(mode)))
        return
    p = state.F.shape[0]
    kw = dict(dtype=state.F.dtype, device=state.F.device)
    ssr, s_mu, s_mu2 = torch.zeros(p, **kw), torch.zeros(p, **kw), torch.zeros(p, **kw)
    with torch.no_grad():
        for s, e, x, z in data:
            mu = state.mu(s, e, z)
            ssr += ((x - mu) ** 2).sum(dim=1)
            s_mu += mu.sum(dim=1)
            s_mu2 += (mu**2).sum(dim=1)
        state.log_theta.copy_(dispersion_from_moments(ssr, s_mu, s_mu2, mode))


def _accumulate_grads(state: FitState, data: DataSource, lam_G: float = 0.0) -> None:
    g_active = lam_G > 0 and state.F.shape[1] > 0 and state.G_raw.requires_grad
    for s, e, x, z in data:
        loss = nb_nll(x, state.eta(s, e, z), state.log_theta.unsqueeze(1)).sum()
        if g_active:
            loss = loss + lam_G * state.G(slice(s, e)).sum()
        loss.backward()


def _clip_bound(state: FitState) -> torch.Tensor:
    """Per-column bound on |F| so that |F_fk| * max_s G_sk <= F_CLIP."""
    if state.G_raw.shape[1] == 0:
        return torch.ones(0, dtype=state.F.dtype, device=state.F.device)
    gmax = state.G().max(dim=0).values.clamp(min=1e-12)
    return F_CLIP / gmax


def _prox_and_clip_F(state: FitState, opt: torch.optim.Adam | None, cfg: FitConfig) -> None:
    """Soft-threshold F by the L1 prox (scaled by Adam's per-coordinate
    effective step when available) and apply the separation contribution clip."""
    with torch.no_grad():
        if cfg.lam > 0:
            thr = None
            if opt is not None:
                stt = opt.state.get(state.F)
                if stt and "exp_avg_sq" in stt:
                    step = stt["step"]
                    step = step.item() if torch.is_tensor(step) else step
                    v_hat = stt["exp_avg_sq"] / (1.0 - 0.999**step)
                    denom = v_hat.sqrt() + 1e-8
                    # Floor at a fraction of the median: coordinates whose
                    # gradients vanish (weak factors) would otherwise get an
                    # exploding threshold that wipes the column and
                    # destabilizes the fit.
                    denom = torch.maximum(denom, 0.1 * denom.median())
                    thr = cfg.lr_F * cfg.lam / denom
            if thr is None:
                thr = cfg.lr_F * cfg.lam
            state.F.copy_(torch.sign(state.F) * (state.F.abs() - thr).clamp(min=0.0))
        bound = _clip_bound(state).unsqueeze(0)
        state.F.copy_(state.F.clamp(-bound, bound))


def _project_G(state: FitState) -> None:
    if state.g_param == "projected":
        with torch.no_grad():
            state.G_raw.clamp_(min=0.0)


def rescale_columns(state: FitState) -> None:
    """Canonicalize the per-factor ambiguities the likelihood cannot see.
    Must run before the convergence check (spec section 3).

    1. Shift: eta is invariant under ``G_k -> G_k + c, a -> a - c F_k``, so
       each usage column is pulled down to touch zero (offset pushed into
       ``a``). Without this the orthant constraint never binds — fitted G goes
       dense and rotational ambiguity partially returns, which measurably
       degrades factor recovery.
    2. Scale: unit-L2 columns of F, scale pushed into G.
    """
    if state.F.shape[1] == 0:
        return
    with torch.no_grad():
        G = state.G(slice(None))
        m = G.min(dim=0).values
        if (m > 1e-8).any():
            state.a.add_(state.F @ m)
            G = (G - m).clamp(min=1e-12)
            state.G_raw.copy_(softplus_inv(G) if state.g_param == "softplus" else G)
        norms = state.F.norm(dim=0)
        c = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
        state.F.div_(c)
        if state.g_param == "softplus":
            g = tf.softplus(state.G_raw) * c
            state.G_raw.copy_(softplus_inv(g.clamp(min=1e-12)))
        else:
            state.G_raw.mul_(c)


def _lbfgs_block(params: list[torch.Tensor], state: FitState, data: DataSource, cfg: FitConfig, is_F: bool) -> None:
    opt = torch.optim.LBFGS(
        params, lr=1.0, max_iter=cfg.lbfgs_inner, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad(set_to_none=False)
        total = 0.0
        for s, e, x, z in data:
            loss = nb_nll(x, state.eta(s, e, z), state.log_theta.unsqueeze(1)).sum()
            if is_F and cfg.lam > 0:
                # smooth |F| surrogate; the outer prox/clip still runs after
                loss = loss + cfg.lam * torch.sqrt(state.F**2 + 1e-8).sum()
            if not is_F and cfg.lam_G > 0 and state.F.shape[1] > 0:
                loss = loss + cfg.lam_G * state.G(slice(s, e)).sum()
            loss.backward()
            total += loss.detach().item()
        return total

    saved = [prm.detach().clone() for prm in params]
    opt.step(closure)
    # Strong-Wolfe line search can step to non-finite territory on this
    # objective; revert the block update rather than poisoning the state.
    with torch.no_grad():
        if any(not torch.isfinite(prm).all() for prm in params):
            for prm, old in zip(params, saved):
                prm.copy_(old)


def run_fit(state: FitState, data: DataSource, cfg: FitConfig):
    """Alternating minimization until convergence. Returns (losses, n_iter,
    converged, clip_hit)."""
    all_params = [state.F, state.a, state.G_raw, state.b] + (
        [state.gamma] if state.gamma is not None else []
    )
    params_G = [state.G_raw] + ([state.b] if state.b_trainable else [])
    params_F = [prm for prm in [state.F, state.a, state.gamma] if prm is not None]
    have_k = state.F.shape[1] > 0

    opt_G = torch.optim.Adam(params_G, lr=cfg.lr_G)
    opt_F = torch.optim.Adam(params_F, lr=cfg.lr_F)
    opt_J = torch.optim.Adam(params_F + params_G, lr=cfg.lr_F)

    theta_active = cfg.update_theta
    if theta_active:
        refresh_dispersion(state, data, cfg.dispersion_mode)

    losses: list[float] = []
    prev = None
    stall = 0
    converged = False
    best, best_it = np.inf, 0
    for it in range(cfg.max_iter):
        if cfg.algorithm == "adam_joint":
            _set_active(params_F + params_G, all_params)
            for _ in range(2 * cfg.inner_steps):
                opt_J.zero_grad(set_to_none=True)
                _accumulate_grads(state, data, cfg.lam_G)
                opt_J.step()
                _prox_and_clip_F(state, opt_J, cfg)
                _project_G(state)
        elif cfg.algorithm == "lbfgs":
            if have_k:
                _set_active(params_G, all_params)
                _lbfgs_block(params_G, state, data, cfg, is_F=False)
                _project_G(state)
            _set_active(params_F, all_params)
            _lbfgs_block(params_F, state, data, cfg, is_F=True)
            _prox_and_clip_F(state, None, cfg)
        elif cfg.algorithm == "adam":
            if have_k:
                _set_active(params_G, all_params)
                for _ in range(cfg.inner_steps):
                    opt_G.zero_grad(set_to_none=True)
                    _accumulate_grads(state, data, cfg.lam_G)
                    opt_G.step()
                    _project_G(state)
            _set_active(params_F, all_params)
            for _ in range(cfg.inner_steps):
                opt_F.zero_grad(set_to_none=True)
                _accumulate_grads(state, data)
                opt_F.step()
                _prox_and_clip_F(state, opt_F, cfg)
        else:
            raise ValueError(f"unknown algorithm: {cfg.algorithm!r}")

        rescale_columns(state)
        if theta_active and (it + 1) % cfg.theta_every == 0:
            refresh_dispersion(state, data, cfg.dispersion_mode)

        loss = full_objective(state, data, cfg.lam, cfg.lam_G)
        losses.append(loss)
        if cfg.verbose and (it % 10 == 0 or it == cfg.max_iter - 1):
            print(f"[glm_seminmf] iter {it:4d}  loss {loss:.6e}")

        rel = abs(prev - loss) / (abs(prev) + 1e-12) if prev is not None else np.inf
        if rel < cfg.tol:
            stall += 1
            if stall >= 3:
                if theta_active:
                    # Mean model has settled: one final dispersion refresh,
                    # then freeze theta so final convergence is judged at
                    # fixed dispersion (per-iteration re-estimation is
                    # unstable; spec section 3).
                    refresh_dispersion(state, data, cfg.dispersion_mode)
                    theta_active = False
                    prev, stall = None, 0
                    continue
                converged = True
                break
        else:
            stall = 0

        # Plateau-triggered learning-rate decay: full-batch Adam oscillates at
        # a fixed step size; geometric decay lets the tol criterion bind.
        if loss < best:
            best, best_it = loss, it
        elif it - best_it >= cfg.lr_patience:
            for opt in (opt_G, opt_F, opt_J):
                for grp in opt.param_groups:
                    grp["lr"] *= cfg.lr_decay
            best_it = it
        prev = loss

    _set_active([], all_params)
    clip_hit = False
    if have_k:
        with torch.no_grad():
            bound = _clip_bound(state).unsqueeze(0)
            clip_hit = bool((state.F.abs() >= bound * (1.0 - 1e-5)).any().item())
    return losses, len(losses), converged, clip_hit


def run_transform(state: FitState, data: DataSource, cfg: FitConfig):
    """Optimize the G-block only (G_raw [, b]) with F, a, gamma, theta frozen.
    Used by ``transform``. Returns (losses, converged)."""
    all_params = [state.F, state.a, state.G_raw, state.b] + (
        [state.gamma] if state.gamma is not None else []
    )
    params_G = [state.G_raw] + ([state.b] if state.b_trainable else [])
    _set_active(params_G, all_params)
    opt_G = torch.optim.Adam(params_G, lr=cfg.lr_G)
    losses: list[float] = []
    prev, stall, converged = None, 0, False
    best, best_it = np.inf, 0
    for it in range(cfg.max_iter):
        for _ in range(cfg.inner_steps):
            opt_G.zero_grad(set_to_none=True)
            _accumulate_grads(state, data, cfg.lam_G)
            opt_G.step()
            _project_G(state)
        loss = full_objective(state, data, 0.0, cfg.lam_G)
        losses.append(loss)
        if prev is not None:
            rel = abs(prev - loss) / (abs(prev) + 1e-12)
            stall = stall + 1 if rel < cfg.tol else 0
            if stall >= 3:
                converged = True
                break
        if loss < best:
            best, best_it = loss, it
        elif it - best_it >= cfg.lr_patience:
            for grp in opt_G.param_groups:
                grp["lr"] *= cfg.lr_decay
            best_it = it
        prev = loss
    _set_active([], all_params)
    return losses, converged
