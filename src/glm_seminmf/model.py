"""NBGLMSemiNMF: negative-binomial GLM semi-NMF with signed loadings.

Model (features x samples counts X):

    X_fs ~ NB(mu_fs, theta_f),   log mu = a + b + Z gamma^T + F G^T

with signed ``F`` (p x k), non-negative ``G`` (n x k), per-feature intercepts
``a``, per-sample log-exposure ``b``, optional covariate term, and per-feature
NB2 dispersion ``theta``. Fitted by constrained proximal block minimization of
``NLL + l1_F * ||F||_1 + l1_G * sum(G) + 0.5 * l2_G * sum(G**2)`` (see the implementation spec for the
identifiability rationale behind the constraints, and the ``l1_G`` parameter
docs for why the small usage penalty exists).
"""

from __future__ import annotations

import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tf

from ._fitting import (
    DataSource,
    FitConfig,
    FitState,
    full_objective,
    rescale_columns,
    run_fit,
    run_transform,
    total_deviance,
)
from ._init import initialize
from ._inputs import encode_Z, validate_X
from ._likelihood import ETA_CLAMP, softplus_inv

__all__ = ["NBGLMSemiNMF"]

# |corr| between two factors' loadings above which they are flagged duplicates.
_DUP_CORR = 0.95
# A factor whose non-zero-usage fraction falls below this is flagged degenerate.
_DEGEN_FRAC = 1e-3


class NBGLMSemiNMF:
    """Negative-binomial GLM factor model with signed feature loadings and
    non-negative sample usages.

    Parameters
    ----------
    n_components : int
        Number of latent factors ``k``.
    l1_F : float, default 0.0
        L1 penalty weight on the (column-normalized) loadings. The penalty is
        applied to the *summed* NLL, so scale it with the number of samples;
        ``0.001 * n`` is a good starting point (an order of magnitude more
        visibly degrades fit and factor recovery). Not optional in spirit: it
        controls separation divergence and strengthens identifiability;
        ``0.0`` is accepted but a separating feature can then drive a factor's
        contribution to diverge, which stops the fit with ``"safeguard_hit"``.
    l1_G : float or "auto", default "auto"
        Small L1 penalty on usages (sum of G). The linear predictor is
        invariant under ``G_k -> G_k + c`` with ``a -> a - c F_k``, so the
        likelihood alone leaves each usage column's baseline unidentified;
        this penalty smoothly anchors columns at the touch-zero
        representative, which keeps the ``G >= 0`` constraint active and
        measurably improves factor recovery and restart stability.
        ``"auto"`` uses ``0.005 * p``. Set 0.0 to disable (identifiability
        then rests on the orthant constraint alone).
    exposure : {"offset", "fit"} or ndarray of shape (n,), default "offset"
        ``"offset"``: fixed per-sample ``b_s = log(total_s / median total)``
        (the stable default). ``"fit"``: estimate ``b`` jointly — can absorb
        structure that belongs in ``G``. An array is used as fixed per-sample
        log-exposure offsets.
    dispersion : {"trend", "feature", "shared"} or float, default "trend"
        NB dispersion handling; ``"trend"`` shrinks per-feature moment
        estimates toward a mean-dispersion trend. A scalar or (p,) array fixes theta.
    init : {"svd", "nmf", "random"} or (F0, G0) tuple, default "svd"
        Warm start, computed on log1p of exposure-normalized counts.
    max_iter, tol : int, float
        Outer-iteration cap and relative-objective stagnation trigger.
        Success requires the independent physical stationarity test.
    batch_size : int or None, default None
        Samples per streamed chunk. ``None`` chooses a memory-safe size
        automatically. Semantics are full-batch either way; this only bounds
        memory (a dense p x n mean matrix is never materialized).
    device : {"auto", "cpu", "cuda", ...}, default "auto"
    random_state : int or None
        Seed. Same seed + same device => bit-identical results; CPU and GPU
        differ in the last digits.
    l2_G : float, default 0.0
        Optional squared usage penalty, ``0.5 * l2_G * sum(G**2)``.
    stationarity_tol : float, default 1e-3
        Maximum absolute physical KKT residual required in every active block.
    max_seconds : float or None
        Cooperative per-call deadline, including initialization. In-flight
        operations and a final audit can overrun it; benchmark process caps
        provide a hard external limit. Optional deviance scoring is skipped
        once the deadline expires.
    algorithm : str, default "proximal"
        Curvature-preconditioned exact-L1 constrained steps with backtracking.
        Legacy Adam/L-BFGS names are deprecated aliases, with a warning.
    learning_rate : float, default 1.0
        Initial multiplier of each curvature-scaled trial step, reduced by
        line search as needed. This is not an Adam learning rate.
    g_parametrization : str, default "projected"
        Optimize physical nonnegative G. Legacy "softplus" requests only
        inverse-softplus raw outputs; it no longer changes optimization.
    dtype : str, default "float64"
        Reference precision. float32 is supported but may stagnate above the
        requested absolute residual. Fixed loadings promote precision as needed.
    verbose : bool

    Attributes
    ----------
    F_ : ndarray (p, k)
        Signed loadings, unit-L2 columns, ordered by descending per-factor
        deviance explained. F_fixed preserves supplied values and order.
    G_ : ndarray (n, k)
        Physical nonnegative usages, with exact boundary zeros. Export does
        not snap or otherwise change the fitted parameters.
    G_raw_ : ndarray (n, k)
        Same as G_ in projected mode; inverse-softplus compatibility values
        in legacy softplus mode (zeros use the dtype tiny floor). Both can
        contain genuine boundary ties; no artificial rank distinctions.
    a_, b_, gamma_, theta_ : ndarrays
    loss_ : ndarray
        Full penalized objective at initialization and each completed outer
        iteration. len(loss_) == n_iter_ + 1; restoration is in history_.
        final_objective_ reports the returned checkpoint, which may be earlier.
    n_iter_, converged_ : int, bool
        Actual completed outer iterations and certification of returned state.
        ``converged_`` is True for ``stop_reason_`` of ``"stationary"`` (KKT
        residuals within ``stationarity_tol``) or ``"numerically_stationary"``
        (no representable step improves the objective, so the fit is optimal
        for the compute dtype even though its residual exceeds the tolerance;
        see ``stationarity_["numerical_floor"]``).
    stop_reason_, stationarity_, timed_out_
        Explicit outcome, physical KKT residuals/feasibility, and budget status.
        ``stationarity_["max_contribution"]`` reports the largest per-factor
        ``max|F[:,k]| * max(G[:,k])`` and ``safeguard_active`` whether it is
        above the divergence trigger. Neither constrains the optimizer nor
        enters ``passed``; only sustained growth stops a fit.
    initial_objective_, final_objective_, objective_components_, history_
        True initial/final objectives, final NLL/penalty components, phase events.
    improved_on_init_, mean_improved_on_init_ : bool
        Improvement of the complete state versus the initial state; separately,
        mean-model improvement evaluating both states at the returned theta.
    dispersion_status_, theta_updates_
        Dispersion estimating-phase status; no claim of joint theta optimality.
    transform_stop_reason_, transform_stationarity_, transform_converged_
        Corresponding diagnostics from the most recent transform call.
    deviance_explained_ : float
        1 - deviance(model)/deviance(intercepts + exposure + covariates null).
    component_stats_ : DataFrame
        Per factor: deviance_explained, usage_frac_nonzero, usage_mean,
        neg_loading_mass, n_features_above_threshold, degenerate,
        duplicate_of (-1 when none). Degenerate/duplicate factors are flagged,
        never dropped.
    """

    def __init__(
        self,
        n_components: int,
        l1_F: float = 0.0,
        l1_G: float | str = "auto",
        exposure: str | np.ndarray = "offset",
        dispersion: str | float = "trend",
        init: str | tuple = "svd",
        max_iter: int = 500,
        tol: float = 1e-5,
        batch_size: int | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
        *,
        algorithm: str = "proximal",
        g_parametrization: str = "projected",
        learning_rate: float = 1.0,
        inner_steps: int = 5,
        dtype: str = "float64",
        l2_G: float = 0.0,
        stationarity_tol: float = 1e-3,
        max_seconds: float | None = None,
    ):
        self.n_components = n_components
        self.l1_F = l1_F
        self.l1_G = l1_G
        self.exposure = exposure
        self.dispersion = dispersion
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.algorithm = algorithm
        self.g_parametrization = g_parametrization
        self.learning_rate = learning_rate
        self.inner_steps = inner_steps
        self.dtype = dtype
        self.l2_G = l2_G
        self.stationarity_tol = stationarity_tol
        self.max_seconds = max_seconds

    # ------------------------------------------------------------------ utils

    def _torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    def _resolved_l1_G(self, p: int) -> float:
        if self.l1_G == "auto":
            return 0.005 * p
        return float(self.l1_G)

    def _validate_options(self):
        if self.algorithm in ("adam", "adam_joint", "lbfgs"):
            warnings.warn(f"algorithm={self.algorithm!r} is deprecated; using the objective-consistent "
                          "proximal solver. learning_rate is now a curvature-scaled trial step.",
                          FutureWarning, stacklevel=3)
        elif self.algorithm != "proximal":
            raise ValueError("algorithm must be 'proximal' (legacy names are deprecated aliases)")
        if self.g_parametrization == "softplus":
            warnings.warn("softplus optimization is deprecated; physical projected usages are used. "
                          "G_raw_ is a compatibility inverse-softplus output with boundary ties.",
                          FutureWarning, stacklevel=3)
        elif self.g_parametrization != "projected":
            raise ValueError("g_parametrization must be 'projected' or legacy 'softplus'")
        for name in ("l1_F", "l2_G"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.l1_G != "auto" and (not np.isfinite(float(self.l1_G)) or float(self.l1_G) < 0):
            raise ValueError("l1_G must be 'auto' or finite and nonnegative")
        for name in ("stationarity_tol", "learning_rate", "tol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("max_iter", 0), ("inner_steps", 1), ("n_components", 0)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.batch_size is not None and (not isinstance(self.batch_size, int) or self.batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if self.max_seconds is not None and (not np.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    def _cfg(self, p: int) -> FitConfig:
        return FitConfig(
            lam=self.l1_F, lam_G=self._resolved_l1_G(p), lam_G2=self.l2_G,
            max_iter=self.max_iter, tol=self.tol, stationarity_tol=self.stationarity_tol,
            inner_steps=self.inner_steps, initial_step=self.learning_rate,
            dispersion_mode=self.dispersion, update_theta=isinstance(self.dispersion, str),
            verbose=self.verbose,
        )

    def _record_result(self, result, prefix=""):
        for name in ("n_iter", "converged", "stop_reason", "initial_objective", "final_objective",
                     "best_iteration", "stationarity", "objective_components", "dispersion_status",
                     "theta_updates", "timed_out", "elapsed_seconds"):
            setattr(self, prefix + name + "_", getattr(result, name))
        setattr(self, prefix + "loss_", np.asarray(result.losses))
        setattr(self, prefix + "history_", result.history)
        tolerance = 1e-10 * max(1, abs(result.initial_objective))
        setattr(self, prefix + "improved_on_init_", result.final_objective < result.initial_objective - tolerance)
        setattr(self, prefix + "initial_at_final_theta_", result.initial_at_final_theta)
        setattr(self, prefix + "mean_improved_on_init_", result.final_objective <
                result.initial_at_final_theta - 1e-10 * max(1, abs(result.initial_at_final_theta)))

    def _exposure_b(self, X, n: int) -> tuple[np.ndarray, bool]:
        """Return (initial b, b_trainable). Also records median_total_ at fit."""
        totals = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
        self.median_total_ = float(np.median(totals))
        if isinstance(self.exposure, np.ndarray):
            b = np.asarray(self.exposure, dtype=np.float64).ravel()
            if not np.isfinite(b).all():
                raise ValueError("exposure array must be finite")
            if b.shape[0] != n:
                raise ValueError(f"exposure array has length {b.shape[0]}, expected {n}")
            return b, False
        if self.exposure == "offset":
            if not hasattr(self, "median_total_"):
                self.median_total_ = float(np.median(totals))
            return np.log(totals / self.median_total_), False
        if self.exposure == "fit":
            if not hasattr(self, "median_total_"):
                self.median_total_ = float(np.median(totals))
            return np.log(totals / self.median_total_), True
        raise ValueError(f"exposure must be 'offset', 'fit', or an array, got {self.exposure!r}")

    # -------------------------------------------------------------------- fit

    def fit(self, X, Z=None, *, F_fixed=None, theta_fixed=None) -> "NBGLMSemiNMF":
        """Fit the model to counts ``X`` (p features x n samples).

        ``X``: dense ndarray or scipy CSR/CSC of raw integer counts.
        ``Z``: optional (n, q) covariate design (ndarray or DataFrame;
        DataFrame categoricals are one-hot encoded, first level dropped).
        ``F_fixed`` optionally supplies (p, k) frozen loadings, retaining their
        values, scale, and order. ``theta_fixed`` supplies a positive scalar
        or (p,) vector and overrides the constructor dispersion policy.
        """
        started = time.monotonic()
        self._validate_options()
        X, p, n = validate_X(X)
        if hasattr(self, "median_total_"):
            del self.median_total_
        Znp, self.z_names_ = encode_Z(Z, n)
        k = self.n_components
        device, dtype = self._torch_device(), self._torch_dtype()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        b0, b_trainable = self._exposure_b(X, n)
        F0, G0, a0, gamma0 = initialize(X, b0, k, self.init, self.random_state, Znp)
        if F_fixed is not None:
            F_fixed = np.asarray(F_fixed)
            if F_fixed.shape != (p, k) or not np.isfinite(F_fixed).all():
                raise ValueError("F_fixed must be finite with shape (p, n_components)")
            F0 = F_fixed.copy()
            # Preserve supplied values even when the ordinary compute default is lower precision.
            if not np.array_equal(F0.astype(np.float32).astype(F0.dtype), F0):
                dtype = torch.float64
        if not np.isfinite(F0).all() or not np.isfinite(G0).all():
            raise ValueError("initial factors must be finite")
        state = self._build_state(F0, G0, a0, b0, gamma0, Znp, p, n, k, b_trainable, device, dtype)
        state.F_trainable = F_fixed is None
        self.fixed_loadings_ = F_fixed is not None
        rescale_columns(state)
        data = DataSource(X, Znp, device, dtype, self.batch_size)
        cfg = self._cfg(p)
        fixed_theta = theta_fixed if theta_fixed is not None else (
            self.dispersion if not isinstance(self.dispersion, str) else None)
        if fixed_theta is not None:
            theta = np.asarray(fixed_theta, dtype=float)
            if theta.ndim > 1 or (theta.ndim == 1 and theta.shape != (p,)) or not np.isfinite(theta).all() or (theta <= 0).any():
                raise ValueError("fixed dispersion must be positive and finite, scalar or shape (p,)")
            state.log_theta.copy_(torch.as_tensor(np.broadcast_to(np.log(theta), (p,)).copy(), device=device, dtype=dtype))
            cfg.update_theta = False
        elif self.dispersion not in ("trend", "feature", "shared"):
            raise ValueError("unknown dispersion mode")
        cfg.deadline = None if self.max_seconds is None else started + self.max_seconds
        optimization_start = time.monotonic()
        result = run_fit(state, data, cfg)
        optimization_end = time.monotonic()
        self._record_result(result)
        if not result.converged:
            warnings.warn(f"did not converge: {result.stop_reason} after {result.n_iter} iterations "
                          f"(physical residual {result.stationarity['max_residual']:.3g})",
                          RuntimeWarning, stacklevel=2)
        if result.stop_reason == "safeguard_hit":
            warnings.warn(
                "factor contribution diverged (sustained growth above the separation trigger); "
                "this indicates a separating feature/factor, not a tight iteration budget. "
                "Returning the best checkpoint seen; raise l1_F or reduce n_components",
                RuntimeWarning, stacklevel=2)
        outside = result.stationarity["predictor"]["entries_outside_moment_range"]
        if outside:
            # The contribution level is not a separation signal: a genuinely
            # separating factor saturates below it, while ordinary fits at
            # moderate p*n exceed it. The predictor range is well defined --
            # beyond it, moment estimation and deviance are computed on a
            # clamped mean, so the reported fit quality is not exact.
            warnings.warn(
                f"{outside} fitted predictor entries lie outside +/-{ETA_CLAMP:g}, where moment "
                "estimation and deviance use a clamped mean; deviance_explained_ and theta_ are "
                "approximate there. Inspect stationarity_['predictor'] and component_stats_",
                RuntimeWarning, stacklevel=2)
        self.compute_dtype_ = str(dtype).removeprefix("torch.")
        self._fit_deadline = cfg.deadline
        self._finalize(state, data)
        self.total_seconds_ = time.monotonic() - started
        self.phase_seconds_ = dict(initialization=optimization_start-started,
                                   optimization_and_certification=optimization_end-optimization_start,
                                   finalization=time.monotonic()-optimization_end)
        return self

    def _build_state(self, F0, G0, a0, b0, gamma0, Znp, p, n, k, b_trainable, device, dtype) -> FitState:
        as_t = lambda arr: torch.as_tensor(np.asarray(arr, dtype=np.float64).copy(), device=device, dtype=dtype)
        G0t = as_t(G0)
        G_raw = G0t.clone()
        return FitState(
            F=as_t(F0).contiguous(),
            G_raw=G_raw.contiguous(),
            a=as_t(a0),
            b=as_t(b0),
            gamma=None
            if Znp is None
            else (torch.zeros(p, Znp.shape[1], device=device, dtype=dtype) if gamma0 is None else as_t(gamma0)),
            log_theta=torch.zeros(p, device=device, dtype=dtype),
            b_trainable=b_trainable,
            g_param="projected",
        )

    # -------------------------------------------------------- post-processing

    def _finalize(self, state: FitState, data: DataSource) -> None:
        k = state.F.shape[1]
        deadline = getattr(self, "_fit_deadline", None)
        expired = lambda: deadline is not None and time.monotonic() >= deadline
        d_model = np.nan if expired() else total_deviance(state, data)
        d_null = self._null_deviance(state, data)
        self.deviance_explained_ = float(1.0 - d_model / d_null) if d_null > 0 else (np.nan if np.isnan(d_null) else 0.0)
        dev_k = np.full(k, np.nan)
        for j in range(k):
            if expired():
                break
            dev_k[j] = ((total_deviance(state, data, drop_factor=j) - d_model) / d_null
                        if d_null > 0 else np.nan)
        self.finalization_timed_out_ = expired()
        order = np.arange(k) if self.fixed_loadings_ else np.argsort(-dev_k, kind="stable")
        with torch.no_grad():
            state.F.copy_(state.F[:, order.copy()])
            state.G_raw.copy_(state.G_raw[:, order.copy()])
        dev_k = dev_k[order]

        F = state.F.detach().cpu().numpy().astype(np.float64)
        G_raw = state.G_raw.detach().cpu().numpy().astype(np.float64)
        G = tf.softplus(state.G_raw).detach().cpu().numpy().astype(np.float64) if \
            state.g_param == "softplus" else np.maximum(G_raw, 0.0)

        # Direct constraints give exact zeros without changing the fitted mean.
        if self.g_parametrization == "softplus":
            G_raw = softplus_inv(state.G_raw).cpu().numpy().astype(np.float64)
        self.export_objective_change_ = 0.0
        self.export_max_predictor_change_ = 0.0

        frac_nz = (G > 0).mean(axis=0) if k else np.zeros(0)
        abs_mass = np.abs(F).sum(axis=0) + 1e-300
        neg_mass = np.abs(np.minimum(F, 0.0)).sum(axis=0) / abs_mass
        thr = 3.0 / np.sqrt(F.shape[0])
        n_above = (np.abs(F) > thr).sum(axis=0)

        dup = np.full(k, -1)
        if k > 1:
            centered = F - F.mean(axis=0)
            norms = np.linalg.norm(centered, axis=0)
            denom = norms[:, None] * norms[None, :]
            C = np.divide(centered.T @ centered, denom, out=np.zeros((k, k)), where=denom > 0)
            for j in range(k):
                partners = [i for i in range(k) if i != j and abs(C[j, i]) > _DUP_CORR]
                if partners:
                    dup[j] = partners[0]
        degenerate = frac_nz < _DEGEN_FRAC

        self.F_, self.G_, self.G_raw_ = F, G, G_raw
        self.G_internal_ = G.copy()
        self.a_ = state.a.detach().cpu().numpy().astype(np.float64)
        self.b_ = state.b.detach().cpu().numpy().astype(np.float64)
        self.gamma_ = None if state.gamma is None else state.gamma.detach().cpu().numpy().astype(np.float64)
        self.theta_ = np.exp(state.log_theta.detach().cpu().numpy().astype(np.float64))
        self.component_stats_ = pd.DataFrame(
            {
                "deviance_explained": dev_k,
                "usage_frac_nonzero": frac_nz,
                "usage_mean": G.mean(axis=0) if k else np.zeros(0),
                "neg_loading_mass": neg_mass,
                "n_features_above_threshold": n_above,
                "degenerate": degenerate,
                "duplicate_of": dup,
            }
        )
        if degenerate.any() or (dup >= 0).any():
            warnings.warn(
                "degenerate or duplicated factors detected (see component_stats_); "
                "this is evidence about n_components, factors were not dropped",
                RuntimeWarning,
                stacklevel=3,
            )

    def _null_deviance(self, state: FitState, data: DataSource) -> float:
        """Deviance of the intercepts + exposure + covariates model, fitted
        with the main model's dispersion held fixed."""
        device, dtype = state.F.device, state.F.dtype
        p, n = state.F.shape[0], state.G_raw.shape[0]
        null = FitState(
            F=torch.zeros(p, 0, device=device, dtype=dtype),
            G_raw=torch.zeros(n, 0, device=device, dtype=dtype),
            a=state.a.detach().clone(),
            b=state.b.detach().clone(),
            gamma=None if state.gamma is None else state.gamma.detach().clone(),
            log_theta=state.log_theta.detach().clone(),
            b_trainable=state.b_trainable,
            g_param=state.g_param,
        )
        cfg = self._cfg(p)
        cfg.max_iter = min(150, self.max_iter)
        cfg.update_theta = False
        cfg.lam = 0.0
        cfg.verbose = False
        cfg.deadline = getattr(self, "_fit_deadline", None)
        if cfg.deadline is not None and time.monotonic() >= cfg.deadline:
            self.null_stop_reason_ = "timeout"
            return np.nan
        result = run_fit(null, data, cfg)
        self.null_stop_reason_ = result.stop_reason
        return total_deviance(null, data)

    # -------------------------------------------------------------- transform

    def transform(self, X, Z=None, exposure: np.ndarray | None = None) -> np.ndarray:
        """Fit usages ``G`` for new samples with F, a, gamma, theta held fixed.

        ``X`` is (p, n_new) raw integer counts over the fit-time features;
        ``Z`` must be supplied iff the model was fitted with covariates.
        Returns the (n_new, k) physical nonnegative usage matrix without
        snapping. Raw compatibility values are stored as ``transform_raw_``;
        boundary ties are meaningful. ``exposure`` optionally supplies fixed per-sample
        log-exposure offsets; otherwise offsets come from the new samples'
        totals scaled by the fit-time median total (and are refined per
        sample when the model was fitted with ``exposure="fit"``).
        """
        started = time.monotonic()
        if not hasattr(self, "F_"):
            raise RuntimeError("transform called before fit")
        X, p, n = validate_X(X)
        if p != self.F_.shape[0]:
            raise ValueError(f"X has {p} features; model was fitted with {self.F_.shape[0]}")
        Znp, _ = encode_Z(Z, n, columns=self.z_names_ or None)
        device, dtype = self._torch_device(), self._torch_dtype()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if exposure is not None:
            b0, b_trainable = np.asarray(exposure, dtype=np.float64).ravel(), False
        else:
            totals = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
            b0 = np.log(totals / self.median_total_)
            b_trainable = isinstance(self.exposure, str) and self.exposure == "fit"

        if (Znp is None) != (self.gamma_ is None):
            raise ValueError("Z must be supplied iff the model was fitted with covariates")
        if Znp is not None and Znp.shape[1] != self.gamma_.shape[1]:
            raise ValueError("Z has the wrong number of covariates")
        if b0.shape != (n,) or not np.isfinite(b0).all():
            raise ValueError("exposure must be finite with shape (n_new,)")
        if not np.array_equal(self.F_.astype(np.float32).astype(float), self.F_):
            dtype = torch.float64
        k = self.F_.shape[1]
        as_t = lambda arr: torch.as_tensor(np.asarray(arr, dtype=np.float64).copy(), device=device, dtype=dtype)
        state = FitState(
            F=as_t(self.F_),
            G_raw=torch.zeros(n, k, device=device, dtype=dtype),
            a=as_t(self.a_),
            b=as_t(b0),
            gamma=None if self.gamma_ is None else as_t(self.gamma_),
            log_theta=as_t(np.log(self.theta_)),
            b_trainable=b_trainable,
            g_param="projected",
        )
        data = DataSource(X, Znp, device, dtype, self.batch_size)
        self._warm_start_G(state, data)
        cfg = self._cfg(p)
        cfg.update_theta = False
        cfg.deadline = None if self.max_seconds is None else started + self.max_seconds
        state.F_trainable = False
        result = run_transform(state, data, cfg)
        self._record_result(result, "transform_")
        if not result.converged:
            warnings.warn(f"transform did not converge: {result.stop_reason}", RuntimeWarning, stacklevel=2)

        G_raw = state.G_raw.detach().cpu().numpy().astype(np.float64)
        G = tf.softplus(state.G_raw).detach().cpu().numpy().astype(np.float64) if \
            state.g_param == "softplus" else np.maximum(G_raw, 0.0)
        self.transform_raw_ = (softplus_inv(state.G_raw).cpu().numpy().astype(np.float64)
                               if self.g_parametrization == "softplus" else G_raw)
        self.transform_b_ = state.b.cpu().numpy().astype(np.float64)
        self.transform_total_seconds_ = time.monotonic() - started
        return G

    def _warm_start_G(self, state: FitState, data: DataSource) -> None:
        """Initialize new-sample usages from the positive part of the projection
        of residual log counts onto the (unit-norm) loadings."""
        with torch.no_grad():
            for s, e, x, z in data:
                y = torch.log1p(x * torch.exp(-state.b[s:e]).unsqueeze(0))
                resid = y - state.a.unsqueeze(1)
                if z is not None and state.gamma is not None:
                    resid = resid - state.gamma @ z.T
                g0 = (resid.T @ state.F).clamp(min=1e-4)
                state.G_raw[s:e] = softplus_inv(g0) if state.g_param == "softplus" else g0

    def fit_transform(self, X, Z=None, *, F_fixed=None, theta_fixed=None) -> np.ndarray:
        """Fit the model and return the fitted usages ``G_`` (n, k)."""
        return self.fit(X, Z, F_fixed=F_fixed, theta_fixed=theta_fixed).G_
