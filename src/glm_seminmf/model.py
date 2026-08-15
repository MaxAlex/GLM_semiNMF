"""NBGLMSemiNMF: negative-binomial GLM semi-NMF with signed loadings.

Model (features x samples counts X):

    X_fs ~ NB(mu_fs, theta_f),   log mu = a + b + Z gamma^T + F G^T

with signed ``F`` (p x k), non-negative ``G`` (n x k), per-feature intercepts
``a``, per-sample log-exposure ``b``, optional covariate term, and per-feature
NB2 dispersion ``theta``. Fitted by block-alternating minimization of
``NLL + l1_F * ||F||_1 + l1_G * sum(G)`` (see the implementation spec for the
identifiability rationale behind the constraints, and the ``l1_G`` parameter
docs for why the small usage penalty exists).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tf

from ._fitting import (
    F_CLIP,
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
from ._likelihood import softplus_inv

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
        applied to the *summed* NLL, so scale it with the number of samples.
        Not optional in spirit: it controls separation divergence and
        strengthens identifiability; ``0.0`` is accepted but will warn if any
        loading hits the internal hard clip.
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
        estimates toward a mean-dispersion trend. A float fixes theta.
    init : {"svd", "nmf", "random"} or (F0, G0) tuple, default "svd"
        Warm start, computed on log1p of exposure-normalized counts.
    max_iter, tol : int, float
        Outer-iteration cap and relative-objective convergence tolerance
        (must hold for 3 consecutive outer iterations).
    batch_size : int or None, default None
        Samples per streamed chunk. ``None`` chooses a memory-safe size
        automatically. Semantics are full-batch either way; this only bounds
        memory (a dense p x n mean matrix is never materialized).
    device : {"auto", "cpu", "cuda", ...}, default "auto"
    random_state : int or None
        Seed. Same seed + same device => bit-identical results; CPU and GPU
        differ in the last digits.
    verbose : bool

    Attributes
    ----------
    F_ : ndarray (p, k)
        Signed loadings, unit-L2 columns, ordered by descending per-factor
        deviance explained.
    G_ : ndarray (n, k)
        Non-negative usages (scale absorbed from F normalization). Entries
        below a snap threshold are exact zeros.
    G_raw_ : ndarray (n, k)
        Pre-softplus usages: continuous and tie-free, for rank statistics.
    a_, b_, gamma_, theta_ : ndarrays
    loss_ : ndarray
        Full penalized objective per outer iteration.
    n_iter_, converged_ : int, bool
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
        algorithm: str = "adam",
        g_parametrization: str = "softplus",
        learning_rate: float = 0.05,
        inner_steps: int = 5,
        dtype: str = "float32",
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

    def _cfg(self, p: int) -> FitConfig:
        return FitConfig(
            lam=self.l1_F,
            lam_G=self._resolved_l1_G(p),
            max_iter=self.max_iter,
            tol=self.tol,
            algorithm=self.algorithm,
            lr_G=self.learning_rate,
            lr_F=self.learning_rate,
            inner_steps=self.inner_steps,
            dispersion_mode=self.dispersion,
            update_theta=not isinstance(self.dispersion, (int, float)),
            verbose=self.verbose,
        )

    def _exposure_b(self, X, n: int) -> tuple[np.ndarray, bool]:
        """Return (initial b, b_trainable). Also records median_total_ at fit."""
        totals = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
        if isinstance(self.exposure, np.ndarray):
            b = np.asarray(self.exposure, dtype=np.float64).ravel()
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

    def fit(self, X, Z=None) -> "NBGLMSemiNMF":
        """Fit the model to counts ``X`` (p features x n samples).

        ``X``: dense ndarray or scipy CSR/CSC of raw integer counts.
        ``Z``: optional (n, q) covariate design (ndarray or DataFrame;
        DataFrame categoricals are one-hot encoded, first level dropped).
        """
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

        state = self._build_state(F0, G0, a0, b0, gamma0, Znp, p, n, k, b_trainable, device, dtype)
        rescale_columns(state)
        data = DataSource(X, Znp, device, dtype, self.batch_size)

        cfg = self._cfg(p)
        losses, n_iter, converged, clip_hit = run_fit(state, data, cfg)

        self.loss_ = np.asarray(losses)
        self.n_iter_ = n_iter
        self.converged_ = converged
        if not converged:
            warnings.warn(
                f"did not converge in {self.max_iter} outer iterations "
                f"(tol={self.tol}); results may be unstable",
                RuntimeWarning,
                stacklevel=2,
            )
        if clip_hit:
            warnings.warn(
                "some loadings sit at the separation hard clip (factor "
                f"contribution to the log-mean bounded at {F_CLIP}); this "
                "indicates likelihood divergence under separation — increase "
                "l1_F rather than relying on the clip",
                RuntimeWarning,
                stacklevel=2,
            )

        self._finalize(state, data)
        return self

    def _build_state(self, F0, G0, a0, b0, gamma0, Znp, p, n, k, b_trainable, device, dtype) -> FitState:
        as_t = lambda arr: torch.as_tensor(np.asarray(arr, dtype=np.float64), device=device, dtype=dtype)
        G0t = as_t(G0)
        if self.g_parametrization == "softplus":
            G_raw = softplus_inv(G0t.clamp(min=1e-6))
        elif self.g_parametrization == "projected":
            G_raw = G0t.clone()
        else:
            raise ValueError(f"g_parametrization must be 'softplus' or 'projected'")
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
            g_param=self.g_parametrization,
        )

    # -------------------------------------------------------- post-processing

    def _finalize(self, state: FitState, data: DataSource) -> None:
        k = state.F.shape[1]
        d_model = total_deviance(state, data)
        d_null = self._null_deviance(state, data)
        self.deviance_explained_ = float(1.0 - d_model / d_null) if d_null > 0 else 0.0

        # Per-factor deviance explained by no-refit ablation, then order by it.
        dev_k = np.array(
            [(total_deviance(state, data, drop_factor=j) - d_model) / d_null for j in range(k)]
        )
        order = np.argsort(-dev_k)
        with torch.no_grad():
            state.F.copy_(state.F[:, order.copy()])
            state.G_raw.copy_(state.G_raw[:, order.copy()])
        dev_k = dev_k[order]

        F = state.F.detach().cpu().numpy().astype(np.float64)
        G_raw = state.G_raw.detach().cpu().numpy().astype(np.float64)
        G = tf.softplus(state.G_raw).detach().cpu().numpy().astype(np.float64) if \
            state.g_param == "softplus" else np.maximum(G_raw, 0.0)

        # Snap near-zero usages to exact zeros (softplus never reaches 0).
        col_max = G.max(axis=0, initial=0.0)
        G[G < np.maximum(1e-8, 1e-3 * col_max)[None, :]] = 0.0

        frac_nz = (G > 0).mean(axis=0) if k else np.zeros(0)
        abs_mass = np.abs(F).sum(axis=0) + 1e-300
        neg_mass = np.abs(np.minimum(F, 0.0)).sum(axis=0) / abs_mass
        thr = 3.0 / np.sqrt(F.shape[0])
        n_above = (np.abs(F) > thr).sum(axis=0)

        dup = np.full(k, -1)
        if k > 1:
            C = np.corrcoef(F.T)
            for j in range(k):
                partners = [i for i in range(k) if i != j and abs(C[j, i]) > _DUP_CORR]
                if partners:
                    dup[j] = partners[0]
        degenerate = frac_nz < _DEGEN_FRAC

        self.F_, self.G_, self.G_raw_ = F, G, G_raw
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
        cfg.max_iter = 150
        cfg.update_theta = False
        cfg.lam = 0.0
        cfg.verbose = False
        run_fit(null, data, cfg)
        return total_deviance(null, data)

    # -------------------------------------------------------------- transform

    def transform(self, X, Z=None, exposure: np.ndarray | None = None) -> np.ndarray:
        """Fit usages ``G`` for new samples with F, a, gamma, theta held fixed.

        ``X`` is (p, n_new) raw integer counts over the fit-time features;
        ``Z`` must be supplied iff the model was fitted with covariates.
        Returns the (n_new, k) non-negative usage matrix, snapped to exact
        zeros like ``G_``; the tie-free pre-softplus values are stored as
        ``transform_raw_``. ``exposure`` optionally supplies fixed per-sample
        log-exposure offsets; otherwise offsets come from the new samples'
        totals scaled by the fit-time median total (and are refined per
        sample when the model was fitted with ``exposure="fit"``).
        """
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
            b_trainable = self.exposure == "fit"

        k = self.F_.shape[1]
        as_t = lambda arr: torch.as_tensor(np.asarray(arr, dtype=np.float64), device=device, dtype=dtype)
        state = FitState(
            F=as_t(self.F_),
            G_raw=torch.zeros(n, k, device=device, dtype=dtype),
            a=as_t(self.a_),
            b=as_t(b0),
            gamma=None if self.gamma_ is None else as_t(self.gamma_),
            log_theta=as_t(np.log(self.theta_)),
            b_trainable=b_trainable,
            g_param=self.g_parametrization,
        )
        data = DataSource(X, Znp, device, dtype, self.batch_size)
        self._warm_start_G(state, data)
        cfg = self._cfg(p)
        cfg.update_theta = False
        # G-only optimization tolerates (and needs) a bolder step size than the
        # alternating fit: it converges slowly but stably at the fit-time lr.
        cfg.lr_G = 4.0 * cfg.lr_G
        losses, converged = run_transform(state, data, cfg)
        if not converged:
            warnings.warn("transform did not converge; increase max_iter", RuntimeWarning, stacklevel=2)

        G_raw = state.G_raw.detach().cpu().numpy().astype(np.float64)
        G = tf.softplus(state.G_raw).detach().cpu().numpy().astype(np.float64) if \
            state.g_param == "softplus" else np.maximum(G_raw, 0.0)
        col_max = G.max(axis=0, initial=0.0)
        G[G < np.maximum(1e-8, 1e-3 * col_max)[None, :]] = 0.0
        self.transform_raw_ = G_raw
        return G

    def _warm_start_G(self, state: FitState, data: DataSource) -> None:
        """Initialize new-sample usages from the positive part of the projection
        of residual log counts onto the (unit-norm) loadings."""
        with torch.no_grad():
            for s, e, x, z in data:
                y = torch.log1p(x * torch.exp(-state.b[s:e]).unsqueeze(0))
                resid = y - state.a.unsqueeze(1)
                g0 = (resid.T @ state.F).clamp(min=1e-4)
                state.G_raw[s:e] = softplus_inv(g0) if state.g_param == "softplus" else g0

    def fit_transform(self, X, Z=None) -> np.ndarray:
        """Fit the model and return the fitted usages ``G_`` (n, k)."""
        return self.fit(X, Z).G_
