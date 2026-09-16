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

The `l1_G` term (default `0.005·p`, small relative to the likelihood) is an
addition beyond the original spec objective, adopted for an identifiability
reason found empirically: the predictor is invariant under
`G_k → G_k + c, a → a − c·F_k`, so the likelihood leaves each usage column's
baseline free. Unanchored, fitted `G` drifts dense, the orthant constraint
never binds, and rotational ambiguity partially returns (factor recovery
drops measurably). The penalty smoothly selects the touch-zero representative
of each usage column; a fit-only min-shift canonicalization enforces the same convention exactly. Set `l1_G=0.0` to
recover the bare spec objective.

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
model.fit(sim.X)                      # raw integer counts, features x samples

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
| `degenerate` | Usage collapsed (~all-zero) — evidence about `n_components` |
| `duplicate_of` | Index of a factor with `abs(loading corr) > 0.95`, else −1 |

Degenerate or duplicated factors are **flagged, never dropped or
reinitialized** — a run that produces them is evidence about `k`.

## Practical notes

- **Raw integer counts only.** The model normalizes via `b`; normalized or
  log-transformed input raises.
- `exposure="offset"` (default) is the stable choice; `"fit"` estimates `b`
  jointly and can absorb structure that belongs in `G`. An array supplies
  fixed per-sample log-exposure offsets.
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
- Trial steps crossing the factor-contribution safeguard
  `max(abs(F[:,k])) * max(G[:,k]) > 15` are backtracked. An unresolved safeguard
  stops the run explicitly; accepted solutions are not silently clipped.
- `converged_` requires physical KKT residuals ≤ `stationarity_tol` (default
  1e-3) at the returned checkpoint, with theta fixed. `tol` only detects loss
  stagnation. Inspect `stop_reason_`, `stationarity_`, and `timed_out_`.
- `loss_[0]` is the true initial objective; `n_iter_` counts completed updates.
  `final_objective_` scores the returned best checkpoint. A flat trace or a
  timeout is not evidence of convergence.
- `fit(X, F_fixed=F, theta_fixed=theta)` freezes supplied loading values/order
  and scalar or per-feature dispersion. Transform also freezes intercepts and
  covariate coefficients. All frozen blocks remain unchanged.
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
