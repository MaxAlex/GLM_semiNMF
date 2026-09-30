# glm-seminmf

Negative-binomial GLM semi-NMF: a generalized linear factor model for
overdispersed count matrices with **signed feature loadings** and
**non-negative sample usages**.

Applies to any count matrix where samples vary in total exposure and features
are overdispersed relative to Poisson — document–term matrices, survey counts,
event telemetry. Nothing in the model or implementation is domain-specific.

A full description of the estimation method — the math, the identifiability
analysis, the numerical stabilizations, and open questions — is in
[`docs/METHOD.md`](docs/METHOD.md).

## Model

Counts `X ∈ ℕ^{p×n}` (features × samples):

```
X_fs ~ NB(μ_fs, θ_f)          (NB2: Var = μ + μ²/θ)
log μ_fs = a_f + b_s + (Zγ)_fs + Σ_k F_fk · G_sk
```

| Term | Shape | Constraint | Role |
|---|---|---|---|
| `F` | p × k | signed | Loadings: log-fold change per unit factor activity |
| `G` | n × k | ≥ 0 | Per-sample factor usage |
| `a` | p | free | Feature intercepts |
| `b` | n | free or fixed | Per-sample log exposure |
| `Z`, `γ` | n × q, p × q | given / free | Measured nuisance covariates and their coefficients |
| `θ` | p | > 0 | Per-feature NB dispersion |

Objective: `NLL(F, G, a, b, γ; θ) + l1_F · ‖F‖₁ + l1_G · Σ G + 0.5 · l2_G · Σ G²`,
with G ≥ 0 and unit-L2 trainable loading columns. A constrained proximal solver
uses exact L1 and backtracking against this objective. Optional usage L2 defaults
to zero. Dispersion uses a separate bounded method-of-moments estimating phase.
See the [optimizer migration notes](docs/OPTIMIZER_CHANGELOG.md).

For single-cell UMI counts, see the [single-cell guide and measurements](docs/SINGLE_CELL_WORK.md).
The factor product decomposes **log expression relative to baseline**; usages
are activities, not transcript-count allocations or mixture proportions.

The `l1_G` term (default `0.005·p`) is an
addition beyond the original spec objective, adopted for an identifiability
reason found empirically: the predictor is invariant under
`G_k → G_k + c, a → a − c·F_k`, so the likelihood leaves each usage column's
baseline free. Unanchored, fitted `G` drifts dense, the orthant constraint
never binds, and rotational ambiguity partially returns (factor recovery
drops measurably). The penalty smoothly selects the touch-zero representative
of each usage column; a fit-only min-shift canonicalization enforces the same convention exactly. Set `l1_G=0.0` to
recover the bare spec objective.

**Check this penalty on sparse data.** `0.005·p` is calibrated on the count
scale of the package's generator, and its premise that the penalty is small
relative to the likelihood can fail badly. On 20 Newsgroups (2.45% nonzero,
mean count 0.044) it drives every usage to zero: all factors degenerate,
deviance explained 0.0005, and the fit certifies, correctly, at that degenerate
optimum. `component_stats_["degenerate"]` and the accompanying warning catch
it. See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the penalty window
measured on that data.

An opt-in `l1_G="information"` uses the initial NB score-information scale
instead of gene count. Its resolved coefficient is available as `l1_G_` and
is reused unchanged for transform. This is an experimental starting rule;
validate a numeric penalty path and optional `l2_G` on held-out observations.
The default remains `"auto"` for compatibility.

## Why signed F but non-negative G

- **`F` signed** is the point: real latent processes both elevate and
  suppress features. Forcing non-negative loadings splits one process into an
  elevated component plus a down-weighted "shadow" component, roughly doubling
  the factor count and creating artifactual anticorrelations between usages.
- **`G ≥ 0` restricts factor ambiguity** and makes inactive samples meaningful.
  It does not prove unique identification for arbitrary finite data. Correlated
  and dense-support factors may still admit alternative decompositions.
- The remaining scale ambiguity is fixed by constraining columns of `F` to
  unit L2 norm (initial scale absorbed into `G` before fitting), and factors are ordered by
  descending deviance explained, so factors are directly comparable across
  runs and datasets.
- The **L1 penalty on `F`** both controls separation divergence (features
  structurally zero where a factor is active would otherwise drive loadings to
  −∞) and strengthens identifiability, since sparsity is not
  rotation-equivariant. Treat `l1_F = 0` as a diagnostic mode, not a default:
  the fitter warns if any loading hits its hard clip.

## Install

```bash
pip install .            # torch, numpy, scipy, pandas
pip install '.[anndata]' # optional AnnData adapter
```

GPU is selected when CUDA is available (`device="auto"`). Float64 is the
correctness default; CPU/GPU comparisons use declared numerical tolerances.

## Worked example

```python
import numpy as np
from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf

# synthetic data from the model's own generative process
sim = simulate_nb_seminmf(p=2000, n=3000, k=8,
                          negative_loading_fraction=0.3, random_state=0)

model = NBGLMSemiNMF(
    n_components=8,
    l1_F=3.0,             # scaled to the summed NLL: grow it with n
    l1_G="auto",          # usage-baseline anchor, 0.005*p (see above)
    exposure="offset",    # b_s = log(total_s / median total), held fixed
    dispersion="trend",   # per-feature theta shrunk toward a mean trend
    init="svd",
    random_state=0,
)
model.fit(sim.X)                      # unnormalized counts, features x samples

model.F_                  # (p, k) signed loadings, unit-L2 columns
model.G_                  # (n, k) physical usages, >= 0, exact boundary zeros
model.G_raw_              # same as G_ by default; boundary ties are meaningful
model.deviance_explained_ # vs. intercepts + exposure + covariates null
model.component_stats_    # per-factor diagnostics (see below)

# usages for held-out samples, model frozen:
G_new = model.transform(sim.X)
```

With covariates (batch, run, instrument, …):

```python
import pandas as pd
Z = pd.DataFrame({"batch": batch_labels})   # categoricals one-hot, ref dropped
model.fit(X, Z=Z)
model.gamma_    # (p, q) per-feature covariate coefficients
```

### `component_stats_`

One row per factor, ordered by descending deviance explained:

| Column | Meaning |
|---|---|
| `deviance_explained` | Deviance increase when the factor is ablated (no refit), / null deviance |
| `usage_frac_nonzero` | Fraction of samples with non-zero usage |
| `usage_mean` | Mean usage |
| `neg_loading_mass` | Fraction of loading L1 mass that is negative |
| `n_features_above_threshold` | Features with `abs(loading) > 3/sqrt(p)` |
| `degenerate` | Every usage is exactly zero; check penalties as well as `n_components` |
| `rare` | Positive usage in fewer than 0.1% of samples; distinct from collapse |
| `duplicate_of` | Index of a factor with `abs(loading corr) > 0.95`, else −1 |

Degenerate or duplicated factors are **flagged, never dropped or
reinitialized**. Inspect usage penalties and `k` when interpreting them.

## Practical notes

- **Unnormalized nonnegative counts.** Fractional corrected counts are accepted
  without rounding. Fractional input with a maximum below 30 is rejected as
  potentially normalized; this heuristic can also reject low-depth corrected
  counts. The model normalizes via `b`.
- `exposure="offset"` (default) is the stable choice; `"fit"` estimates `b`
  jointly and can absorb structure that belongs in `G`. An array supplies
  fixed per-sample log-exposure offsets.
- For selected-gene matrices, compute exposures from the broader count matrix
  first, then use `fit(X, exposure=log_offsets)`. The AnnData adapter accepts
  `exposure_key` for a column of precomputed log offsets in `obs`. When fit
  used external exposures, `transform(..., exposure=...)` requires offsets
  with the same reference; it does not fall back to selected-gene totals.
- `init="pearson"` offers a clipped residual-SVD warm start from a nuisance-only
  NB mean, with working theta=100. `dispersion="trend_pooled"` pools signed
  variance excess before positivity and shrinks weak mean bins toward shared
  dispersion. Both are opt-in; the single-cell guide records their limits.
- `reuse_step_size=True` optionally reuses accepted block step sizes as
  line-search starts. `work_` records optimization derivative passes and
  line-search trials, excluding audits and finalization.
- `l1_F` applies to the summed NLL with `F` column-normalized, so its
  meaning is stable in `p` but should scale roughly with `n`; start around
  `0.001·n`. Benchmarks show `0.01·n` is already strong enough to crush real
  factors and stall convergence.
- `batch_size` bounds memory (sample-chunked streaming; a dense p × n mean
  matrix is never materialized). It does not change semantics — gradients are
  accumulated to full batch either way.
- **Reproducibility:** the same `random_state` on the same device gives
  bit-identical results. CPU and GPU differ in the last float digits.
  Sparse and dense inputs agree bitwise up to the internal densification
  threshold (~5·10⁷ entries), and to float tolerance above it.
- The factor contribution `max(abs(F[:,k])) * max(G[:,k])` is never clipped or
  bounded. Above 15 it arms divergence monitoring and is reported as
  `stationarity_["safeguard_active"]`; only sustained growth (doubling over
  `safeguard_patience` iterations) stops a fit, with `safeguard_hit`. Ordinary
  fits cross that level transiently on the way to a feasible optimum.
- `converged_` is set by either of two certificates, distinguished by
  `stop_reason_`. `"stationary"` means physical KKT residuals ≤
  `stationarity_tol` (default 1e-3) at the returned checkpoint with theta
  fixed. `"numerically_stationary"` means no representable step improves the
  objective, so the fit is optimal for the compute dtype even though its
  residual is larger; `stationarity_["numerical_floor"]` carries the evidence
  and the residual is still reported unchanged. `tol` only detects loss
  stagnation. Inspect `stop_reason_`, `stationarity_`, and `timed_out_`.
- `loss_[0]` is the true initial objective; `n_iter_` counts completed updates.
  `final_objective_` scores the returned best checkpoint. A flat trace or a
  timeout is not evidence of convergence.
- `fit(X, F_fixed=F, theta_fixed=theta)` freezes supplied loading values/order
  and scalar or per-feature dispersion. `F_fixed` determines `n_components`;
  `F_fixed_` and `fixed_loadings_` report whether the basis was supplied.
  Fixed-basis fits initialize usages by projection unless an explicit
  `init=(F0, G0)` supplies them. Transform also freezes intercepts and
  covariate coefficients. All frozen blocks remain unchanged.
- `simulate_nb_seminmf(..., F_fixed=F)` reuses a shared basis across draws;
  `draw_nb_counts(a, b, theta, F, G, rng=...)` also lets callers share nuisance
  parameters while varying cohort usages and exposures.
- `improved_on_init_` compares the returned objective with initialization;
  `mean_improved_on_init_` compares both means at the returned dispersion.
  An uncertified fit without improvement warns. Already stationary starts
  need no improvement, and the solver uses backtracking for both ordinary
  and fixed-basis fits.
- `max_seconds` supplies a cooperative deadline; in-flight work and the final
  audit can overrun it. Optional deviance scoring may be unavailable at timeout.
  Use the bounded benchmark harness when a hard process cap is needed.

## Explicitly out of scope

By design, left to the calling code: choice of `k`; consensus/multi-restart
aggregation; matching or clustering factors across datasets; data cleaning,
QC, stratification, feature selection; downstream statistics on `G`; any
annotation or interpretation of `F`. The library optimizes for the pattern of
many cheap independent fits whose factors are then compared: deterministic
seeding, unit-norm loadings, deviance-ordered factors.

## Development

```bash
uv sync --group dev
uv run pytest             # validation suite (spec section 6): factor recovery,
                          # signed recovery, identifiability, exposure
                          # invariance, covariate absorption, separation, ...
uv run python benchmarks/bench_optimizer.py --out benchmarks/runs/my_run
# bounded 500/2000-feature synthetic screen; never overwrites a run tag
```

Certified results, and what is still unresolved, are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md). Solver behavior and the
migration from the previous Adam/rescaling optimizer are documented in
[docs/METHOD.md](docs/METHOD.md) and
[docs/OPTIMIZER_CHANGELOG.md](docs/OPTIMIZER_CHANGELOG.md).
[docs/REMAINING_WORK.md](docs/REMAINING_WORK.md) is the current to-do list,
with the evidence behind each item's priority.
