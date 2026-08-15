# glm-seminmf

Negative-binomial GLM semi-NMF: a generalized linear factor model for
overdispersed count matrices with **signed feature loadings** and
**non-negative sample usages**.

Applies to any count matrix where samples vary in total exposure and features
are overdispersed relative to Poisson — document–term matrices, survey counts,
event telemetry. Nothing in the model or implementation is domain-specific.

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

Objective: `NLL(F, G, a, b, γ; θ) + l1_F · ‖F‖₁ + l1_G · Σ G`, minimized by
block-alternating optimization (Adam per block, proximal soft-threshold for
the L1 term, method-of-moments dispersion with trend shrinkage).

The `l1_G` term (default `0.005·p`, small relative to the likelihood) is an
addition beyond the original spec objective, adopted for an identifiability
reason found empirically: the predictor is invariant under
`G_k → G_k + c, a → a − c·F_k`, so the likelihood leaves each usage column's
baseline free. Unanchored, fitted `G` drifts dense, the orthant constraint
never binds, and rotational ambiguity partially returns (factor recovery
drops measurably). The penalty smoothly selects the touch-zero representative
of each usage column; a per-iteration min-shift canonicalization (analogous
to the scale fixing) enforces the same convention exactly. Set `l1_G=0.0` to
recover the bare spec objective.

## Why signed F but non-negative G

- **`F` signed** is the point: real latent processes both elevate and
  suppress features. Forcing non-negative loadings splits one process into an
  elevated component plus a down-weighted "shadow" component, roughly doubling
  the factor count and creating artifactual anticorrelations between usages.
- **`G ≥ 0` is what makes the factorization identifiable.** With signed `F`
  alone, `FGᵀ = (FA)(GA⁻ᵀ)ᵀ` for any invertible `A`. Restricting `G` to the
  non-negative orthant reduces the ambiguity to permutation × positive
  diagonal (positive monomial matrices are the only invertible maps that
  preserve the orthant), which also eliminates the sign ambiguity in `F`. The
  constraint restricts *support*, not dependence: usages remain free to
  correlate, which downstream covariance-based statistics require.
- The remaining scale ambiguity is fixed by normalizing columns of `F` to
  unit L2 norm (scale absorbed into `G`), and factors are ordered by
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

GPU is used automatically when CUDA is available (`device="auto"`); the CPU
path is exact, not a stub — CPU and GPU agree to float precision.

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
model.G_                  # (n, k) usages, >= 0, exact zeros after snapping
model.G_raw_              # (n, k) pre-softplus usages (tie-free, for rank stats)
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
- Loadings are hard-clipped at `|F| ≤ 15` as a separation backstop; a warning
  at the clip means `l1_F` is too small, not that the clip is doing useful
  work.

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
uv run python benchmarks/bench.py   # timing, init/optimizer comparisons
```
