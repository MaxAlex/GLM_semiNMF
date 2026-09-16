"""Chunked constrained composite optimization of the NB mean model.

All steps use physical G >= 0, unit loading columns, exact L1, and the same
objective. Dispersion estimation is an explicitly separate, bounded phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as tf

from ._dispersion import dispersion_from_moments
from ._likelihood import ETA_CLAMP, nb_deviance, nb_nll, softplus_inv

F_CLIP = 15.0  # separation safeguard, not an unreported projection
_CHUNK_ELEMENTS = 32_000_000
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
        device = torch.device(device)
        self.device, self.dtype = device, dtype
        auto = max(1, _CHUNK_ELEMENTS // max(self.p, 1))
        self.chunk = min(self.n, batch_size if batch_size else auto)
        self.Z = None if Z is None else torch.as_tensor(Z, device=device, dtype=dtype)
        self._sparse = sp.issparse(X)
        self._X_host = X
        self._resident = None
        if self.p * self.n <= _residency_limit(device, dtype):
            dense = X.toarray() if self._sparse else np.asarray(X)
            if not dense.flags.writeable:
                dense = dense.copy()
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
    g_param: str = "projected"
    F_trainable: bool = True

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
    lam_G: float = 0.0
    lam_G2: float = 0.0
    max_iter: int = 500
    tol: float = 1e-5  # stagnation trigger only
    stationarity_tol: float = 1e-3  # absolute physical KKT residual
    inner_steps: int = 5
    theta_every: int = 10
    theta_max_updates: int = 15
    dispersion_mode: str | float = "trend"
    update_theta: bool = True
    max_linesearch: int = 30
    initial_step: float = 1.0
    deadline: float | None = None
    verbose: bool = False


@dataclass
class FitResult:
    losses: list[float]
    n_iter: int
    converged: bool
    stop_reason: str
    initial_objective: float
    final_objective: float
    best_iteration: int
    stationarity: dict
    objective_components: dict
    dispersion_status: str
    theta_updates: int
    history: list[dict] = field(default_factory=list)
    initial_at_final_theta: float = np.nan
    elapsed_seconds: float = 0.0

    @property
    def timed_out(self):
        return self.stop_reason == "timeout"


def objective_components(state, data, lam=0.0, lam_G=0.0, lam_G2=0.0, *, full=True):
    """Global penalties are counted once, independently of sample chunking."""
    with torch.no_grad():
        nll = 0.0
        f, a, b, log_theta = (state.F.double(), state.a.double(), state.b.double(),
                               state.log_theta.double())
        gamma = None if state.gamma is None else state.gamma.double()
        for s, e, x, z in data:
            eta = a[:, None] + b[s:e] + f @ state.G(slice(s, e)).double().T
            if gamma is not None and z is not None:
                eta = eta + gamma @ z.double().T
            nll += nb_nll(x.double(), eta, log_theta[:, None], full=full).sum().item()
        g = state.G().double()
        result = dict(nll=nll, loading_l1=lam * state.F.double().abs().sum().item(),
                      usage_l1=lam_G * g.sum().item(),
                      usage_l2=0.5 * lam_G2 * g.square().sum().item())
        result["total"] = sum(result.values())
        return result


def full_objective(state, data, lam, lam_G=0.0, lam_G2=0.0):
    return objective_components(state, data, lam, lam_G, lam_G2)["total"]


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


def refresh_dispersion(state: FitState, data: DataSource, mode: str | float) -> float:
    """Method-of-moments update of log_theta from current residuals.

    Returns the max absolute change in log_theta, so the caller can freeze
    dispersion once it has stabilized: theta re-estimation feeds back into the
    mean model, and left running it can sustain a limit cycle in which the
    objective never meets the stall criterion.
    """
    if isinstance(mode, (int, float)):
        state.log_theta.fill_(float(np.log(mode)))
        return 0.0
    p = state.F.shape[0]
    kw = dict(dtype=state.F.dtype, device=state.F.device)
    ssr, s_mu, s_mu2 = torch.zeros(p, **kw), torch.zeros(p, **kw), torch.zeros(p, **kw)
    with torch.no_grad():
        for s, e, x, z in data:
            mu = state.mu(s, e, z)
            ssr += ((x - mu) ** 2).sum(dim=1)
            s_mu += mu.sum(dim=1)
            s_mu2 += (mu**2).sum(dim=1)
        new = dispersion_from_moments(ssr, s_mu, s_mu2, mode)
        delta = float((new - state.log_theta).abs().max().item())
        state.log_theta.copy_(new)
    return delta



def _parameters(state):
    return {name: getattr(state, name) for name in ("F", "G_raw", "a", "b", "gamma", "log_theta")
            if getattr(state, name) is not None}


def _snapshot(state):
    return {name: prm.detach().clone() for name, prm in _parameters(state).items()}


@torch.no_grad()
def _restore(state, saved):
    for name, value in saved.items():
        getattr(state, name).copy_(value)


@torch.no_grad()
def shift_baseline(state):
    """Feasible fit-only shift: preserves eta and cannot increase usage penalties.

    Caller must ensure the intercept is trainable. Transform never calls this.
    """
    if state.F.shape[1]:
        g = state.G()
        m = g.min(dim=0).values
        state.a.add_(state.F @ m)
        g = g - m
        state.G_raw.copy_(softplus_inv(g) if state.g_param == "softplus" else g)


@torch.no_grad()
def rescale_columns(state):
    """Initial canonicalization only; this is NOT penalty invariant.

    Never called after an optimization step. Fixed loadings retain their scale.
    Reject zero trainable loading columns instead of silently reinitializing.
    """
    if state.F.shape[1] and state.F_trainable:
        norms = state.F.norm(dim=0)
        if (norms <= 0).any():
            raise ValueError("trainable initial F columns must have nonzero norm")
        g = state.G() * norms
        state.F.div_(norms)
        state.G_raw.copy_(softplus_inv(g) if state.g_param == "softplus" else g)
    shift_baseline(state)


@torch.no_grad()
def physical_derivatives(state, data, cfg, *, diagnostics=False):
    """Smooth gradients and diagonal curvature in physical coordinates.

    F's exact L1 is handled by the constrained prox/KKT audit, not smoothed.
    The positive curvature is a preconditioner; backtracking handles coupling.
    """
    names = ("F", "G_raw", "a", "b", "gamma")
    grad = {n: torch.zeros_like(getattr(state, n)) for n in names
            if getattr(state, n) is not None}
    curv = {n: torch.zeros_like(v) for n, v in grad.items()}
    theta = state.log_theta.exp()[:, None]
    f2 = state.F.square()
    eta_min, eta_max, outside = np.inf, -np.inf, 0
    for s, e, x, z in data:
        x = x.to(state.F.dtype)
        z = None if z is None else z.to(state.F.dtype)
        eta = state.eta(s, e, z)
        if diagnostics:
            eta_min = min(eta_min, eta.min().item())
            eta_max = max(eta_max, eta.max().item())
            outside += int((eta.abs() > ETA_CLAMP).sum().item())
        prob = torch.sigmoid(eta - state.log_theta[:, None])
        complement = torch.sigmoid(state.log_theta[:, None] - eta)
        r = theta * prob - x * complement
        h = (theta + x) * prob * complement
        g = state.G(slice(s, e))
        grad["F"].add_(r @ g)
        curv["F"].add_(h @ g.square())
        grad["G_raw"][s:e] = r.T @ state.F + cfg.lam_G + cfg.lam_G2 * g
        curv["G_raw"][s:e] = h.T @ f2 + cfg.lam_G2
        grad["a"].add_(r.sum(dim=1))
        curv["a"].add_(h.sum(dim=1))
        grad["b"][s:e] = r.sum(dim=0)
        curv["b"][s:e] = h.sum(dim=0)
        if z is not None and state.gamma is not None:
            grad["gamma"].add_(r @ z)
            curv["gamma"].add_(h @ z.square())
    if diagnostics:
        curv["predictor_stats"] = dict(eta_min=eta_min, eta_max=eta_max,
                                       entries_outside_moment_range=outside)
    return grad, curv


@torch.no_grad()
def sphere_l1_prox(z, threshold):
    """Exact Euclidean prox of L1 plus the unit-sphere indicator, per column.

    Minimize ||f-z||²/2 + threshold*||f||_1 subject to ||f||_2=1.
    Positive soft-thresholds normalize. If all coefficients are nonpositive,
    a signed coordinate maximizing |z|-threshold is a global minimizer.
    Deterministic first-index tie breaking avoids artificial random restarts.
    """
    if not z.numel():
        return z.clone()
    shrunk = torch.sign(z) * (z.abs() - threshold).clamp(min=0)
    norms = shrunk.norm(dim=0)
    result = shrunk / norms.clamp(min=torch.finfo(z.dtype).tiny)
    dead = norms == 0
    if dead.any():
        columns = torch.where(dead)[0]
        rows = (z.abs() - threshold).argmax(dim=0)[dead]
        signs = torch.where(z[rows, columns] < 0, -torch.ones_like(z[rows, columns]), torch.ones_like(z[rows, columns]))
        result[:, columns] = 0
        result[rows, columns] = signs
    return result


def _maxabs(value):
    return value.abs().max().item() if value.numel() else 0.0


@torch.no_grad()
def stationarity(state, data, cfg, *, transform=False):
    """Audit exact L1 KKT conditions, independently of optimizer coordinates."""
    feasibility_tol = max(1e-10, 10 * torch.finfo(state.F.dtype).eps)
    # Audit the numerical values actually exported, with float64 arithmetic.
    # Only parameters and one data chunk are promoted, never the full dataset.
    if state.F.dtype != torch.float64:
        state = FitState(**{n: (v.double() if torch.is_tensor(v) else v)
                            for n, v in vars(state).items()})
    grad, curvature = physical_derivatives(state, data, cfg, diagnostics=True)
    g = state.G()
    rg = torch.where(g > 0, grad["G_raw"], grad["G_raw"].clamp(max=0))
    blocks = {"G": _maxabs(rg)}
    if not transform:
        if state.F_trainable and state.F.numel():
            f = state.F
            v = grad["F"] + cfg.lam * f.sign()
            multiplier = -(f * v).sum(dim=0)
            rf = v + f * multiplier
            rf = torch.where(f != 0, rf, grad["F"].sign() *
                             (grad["F"].abs() - cfg.lam).clamp(min=0))
            blocks["F"] = _maxabs(rf)
        blocks["a"] = _maxabs(grad["a"])
        if state.gamma is not None:
            blocks["gamma"] = _maxabs(grad["gamma"])
    if state.b_trainable:
        blocks["b"] = _maxabs(grad["b"])
    norm_error = (_maxabs(state.F.norm(dim=0) - 1)
                  if state.F_trainable and not transform else 0.0)
    feasibility = max(norm_error, _maxabs(g.clamp(max=0)))
    max_residual = max(blocks.values(), default=0.0)
    finite = all(np.isfinite(x) for x in [*blocks.values(), feasibility]) and all(
        bool(torch.isfinite(v).all()) for v in _parameters(state).values())
    contribution = (state.F.abs().max(dim=0).values * g.max(dim=0).values).max().item() if g.numel() else 0.0
    return dict(blocks=blocks, max_residual=max_residual, feasibility=feasibility,
                tolerance=cfg.stationarity_tol, finite=finite,
                max_contribution=contribution, predictor=curvature["predictor_stats"],
                # Dimensionless reporting only; success uses absolute residuals.
                scaled_blocks={k: v / (1 + {"F": cfg.lam, "G": cfg.lam_G}.get(k, 0))
                               for k, v in blocks.items()},
                passed=finite and max_residual <= cfg.stationarity_tol
                and feasibility <= feasibility_tol)


def _expired(cfg):
    return cfg.deadline is not None and time.monotonic() >= cfg.deadline


@torch.no_grad()
def objective_difference(state, data, cfg, old):
    """J(current)-J(old), without subtracting large likelihood totals.

    Dispersion must be identical. Compute predictor increments from parameter
    increments, and use log1p/expm1 for small softplus differences. This lets
    backtracking resolve improvements well below the NLL's absolute rounding.
    `old` may contain only the changed parameters.
    """
    previous = {n: old.get(n, v).double() for n, v in _parameters(state).items()}
    now = {n: v.double() for n, v in _parameters(state).items()}
    change = {n: now[n] - previous[n] for n in now}
    total = 0.0
    theta = now['log_theta'].exp()[:, None]
    for s, e, x, z in data:
        g0, dg = previous['G_raw'][s:e], change['G_raw'][s:e]
        eta0 = previous['a'][:, None] + previous['b'][s:e] + previous['F'] @ g0.T
        delta = change['a'][:, None] + change['b'][s:e]
        if 'F' in old:
            delta = delta + change['F'] @ g0.T
        if 'G_raw' in old:
            delta = delta + now['F'] @ dg.T
        if z is not None and state.gamma is not None:
            eta0 += previous['gamma'] @ z.double().T
            delta += change['gamma'] @ z.double().T
        d = eta0 - now['log_theta'][:, None]
        prob, complement = torch.sigmoid(d), torch.sigmoid(-d)
        small = delta.abs() < .5
        # Clamp only the argument used by the small-increment formula. Large
        # increments are evaluated by the exact stable NLL branch below.
        u = delta.clamp(-.5, .5)
        remainder = torch.log1p(prob * torch.expm1(u)) - prob * u
        r = theta * prob - x.double() * complement
        difference = r * u + (theta + x.double()) * remainder
        if not small.all():
            direct = nb_nll(x.double(), eta0 + delta, now['log_theta'][:, None]) - nb_nll(
                x.double(), eta0, now['log_theta'][:, None])
            difference = torch.where(small, difference, direct)
        total += difference.sum().item()
    total += cfg.lam * (now['F'].abs() - previous['F'].abs()).sum().item()
    total += cfg.lam_G * change['G_raw'].sum().item()
    total += .5 * cfg.lam_G2 * (change['G_raw'] * (now['G_raw'] + previous['G_raw'])).sum().item()
    return total


@torch.no_grad()
def _step(state, data, cfg, active):
    """One curvature-preconditioned composite step with Armijo backtracking."""
    if not active:
        return "accepted"
    grad, curv = physical_derivatives(state, data, cfg)
    old = {n: getattr(state, n).clone() for n in active}
    metric = {}
    for name in active:
        c = curv[name]
        if name == "F":
            c = c.max(dim=0, keepdim=True).values  # scalar metric per sphere
        metric[name] = c.clamp(min=1e-6)
    step = cfg.initial_step
    saw_finite, saw_safeguard = False, False
    if cfg.max_linesearch == 0:
        return "line_search_failed"
    for _ in range(cfg.max_linesearch):
        if _expired(cfg):
            _restore(state, old)
            return "timeout"
        for name in active:
            z = old[name] - step * grad[name] / metric[name]
            if name == "F":
                z = sphere_l1_prox(z, step * cfg.lam / metric[name])
            elif name == "G_raw":
                z = z.clamp(min=0)
            getattr(state, name).copy_(z)
        finite = all(torch.isfinite(getattr(state, n)).all() for n in active)
        saw_finite |= bool(finite)
        contribution = (state.F.abs().max(dim=0).values * state.G().max(dim=0).values).max().item() if state.F.shape[1] else 0.0
        if contribution > F_CLIP:
            saw_safeguard = True
        elif finite:
            change = objective_difference(state, data, cfg, old)
            slope = sum((grad[n].double() * (getattr(state, n) - old[n]).double()).sum().item() for n in active)
            if "F" in active:
                slope += cfg.lam * (state.F.double().abs().sum() - old["F"].double().abs().sum()).item()
            # No loss-dependent float32 allowance: that can hide large residuals.
            slack = 8 * np.finfo(float).eps
            if np.isfinite(change) and slope <= slack and change <= 1e-4 * slope + slack:
                return "accepted"
        step *= 0.5
    _restore(state, old)
    return "safeguard_hit" if saw_safeguard else ("line_search_failed" if saw_finite else "nonfinite")


@torch.no_grad()
def run_fit(state, data, cfg, *, transform=False):
    """Return the best complete checkpoint, certified at its own fixed theta.

    loss history includes initialization (index 0); restoration is an event,
    never an extra iteration. Each outer iteration is a completed block sweep.
    """
    started = time.monotonic()
    # Accept legacy internal states, but never optimize in saturated coordinates.
    if state.g_param == "softplus":
        state.G_raw.copy_(state.G())
        state.g_param = "projected"
    for prm in _parameters(state).values():
        prm.requires_grad_(False)
    theta_active = cfg.update_theta and not transform and cfg.theta_max_updates > 0
    if theta_active:
        refresh_dispersion(state, data, cfg.dispersion_mode)
    initial = _snapshot(state)
    components = objective_components(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
    initial_loss = best = components["total"]
    best_state, best_it = _snapshot(state), 0
    losses, history = [best], [{"event": "initial", "iteration": 0, "objective": best}]
    theta_updates, stall, n_iter = 0, 0, 0
    phase = "estimating" if theta_active else "fixed"
    reason = "max_iter"
    audit = stationarity(state, data, cfg, transform=transform)
    groups = [["G_raw"] if state.G_raw.numel() else []]
    if state.b_trainable:
        groups[0].append("b")
    if not transform:
        groups.append((["F"] if state.F_trainable and state.F.numel() else []) +
                      ["a"] + (["gamma"] if state.gamma is not None else []))
    if not np.isfinite(best) or not audit["finite"]:
        reason = "nonfinite"
    elif audit["max_contribution"] > F_CLIP:
        reason = "safeguard_hit"
    else:
        for it in range(cfg.max_iter):
            if _expired(cfg):
                reason = "timeout"
                break
            if audit["passed"] and not theta_active:
                reason = "stationary"
                break
            prev = losses[-1]
            sweep_start = _snapshot(state)
            status = "accepted"
            for group in groups:
                for _ in range(cfg.inner_steps):
                    status = _step(state, data, cfg, group)
                    if status != "accepted":
                        break
                if status != "accepted":
                    break
            if status != "accepted":
                # Earlier accepted blocks in this incomplete sweep are valid
                # checkpoints too; do not count the failed sweep as an iteration.
                partial = full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
                if np.isfinite(partial) and partial < best:
                    best, best_state, best_it = partial, _snapshot(state), n_iter
                    history.append(dict(event="partial_checkpoint", iteration=n_iter,
                                        attempted_iteration=it+1, objective=partial))
                reason = status
                break
            if not transform:
                # Unlike rescaling, this fit-only operation reduces usage penalties.
                shift_baseline(state)
            n_iter = it + 1
            loss = full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
            tied = abs(loss - best) <= 32 * np.finfo(float).eps * max(1, abs(best))
            same_theta = torch.equal(state.log_theta, best_state["log_theta"])
            if loss < best or (tied and same_theta and objective_difference(state, data, cfg, best_state) <= 0):
                best, best_state, best_it = loss, _snapshot(state), n_iter
            stall = stall + 1 if abs(prev - loss) <= cfg.tol * max(1, abs(prev)) else 0
            # Leave at least half the iteration budget for a fixed-theta finish.
            freeze = theta_active and (stall >= 3 or n_iter >= max(1, cfg.max_iter // 2))
            if theta_active and (freeze or n_iter % cfg.theta_every == 0):
                delta = refresh_dispersion(state, data, cfg.dispersion_mode)
                theta_updates += 1
                loss = full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
                history.append(dict(event="theta_refresh", iteration=n_iter, delta_log_theta=delta,
                                    objective=loss))
                if loss < best:
                    best, best_state, best_it = loss, _snapshot(state), n_iter
                freeze_reason = ("delta_tolerance" if delta < 0.05 else
                                 "refresh_cap" if theta_updates >= cfg.theta_max_updates else
                                 "mean_stall" if stall >= 3 else "polish_budget")
                freeze = freeze or delta < 0.05 or theta_updates >= cfg.theta_max_updates
                if freeze:
                    phase = "estimated_frozen"
                    theta_active = False
                    _restore(state, best_state)
                    loss = best
                    history.append(dict(event="freeze_restore", iteration=n_iter,
                                        best_iteration=best_it, reason=freeze_reason))
                stall = 0
            losses.append(loss)
            audit = stationarity(state, data, cfg, transform=transform)
            if cfg.verbose and n_iter % 10 == 0:
                print(f"[glm_seminmf] iter {n_iter} objective {loss:.8g} residual {audit['max_residual']:.4g}")
            if not audit["finite"]:
                reason = "nonfinite"
                break
            if stall >= 10 and not theta_active and not audit["passed"]:
                # Still audit every iteration: stagnation alone is never success.
                if abs(objective_difference(state, data, cfg, sweep_start)) <= 1e-16:
                    reason = "stalled_nonstationary"
                    break
    # The initial state remains a candidate even if every proposal failed.
    _restore(state, best_state)
    history.append(dict(event="return_best", iteration=n_iter, best_iteration=best_it))
    audit = stationarity(state, data, cfg, transform=transform)
    if _expired(cfg):
        reason = "timeout"
    elif audit["passed"] and not theta_active and reason not in ("nonfinite", "safeguard_hit"):
        reason = "stationary"
    components = objective_components(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
    final = _snapshot(state)
    _restore(state, initial)
    state.log_theta.copy_(final["log_theta"])
    initial_at_final = full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2)
    _restore(state, final)
    return FitResult(losses, n_iter, reason == "stationary", reason, initial_loss,
                     components["total"], best_it, audit, components,
                     phase if not theta_active else "estimating_budget_exit", theta_updates,
                     history, initial_at_final, time.monotonic() - started)


def run_transform(state, data, cfg):
    return run_fit(state, data, cfg, transform=True)
