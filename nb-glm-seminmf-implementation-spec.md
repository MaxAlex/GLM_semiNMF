# Implementation Spec: NB-GLM Semi-NMF

A negative-binomial generalized linear factor model for overdispersed count matrices, with **signed feature loadings** and **non-negative sample usages**.

Applies to any count matrix where samples vary in total exposure and features are overdispersed relative to Poisson — document–term matrices, survey counts, event telemetry, and so on. Nothing in the model is domain-specific and nothing in the implementation should be.

This is a self-contained library task. It has no dependency on the analysis it feeds. The only thing worth knowing about downstream use: the fitted factors are consumed as *interpretable objects* — loadings are read as latent processes, usages are fed into covariance-based statistics. That makes **reproducibility of the factors across restarts** as important as reconstruction quality, and it is why several requirements below are about identifiability rather than fit.

---

## 1. Model

Counts `X ∈ ℕ^{p×n}` (features × samples):

```
X_fs  ~  NB(μ_fs, θ_f)
log μ_fs  =  a_f + b_s + (Zγ)_fs + Σ_k F_fk · G_sk
```

| Symbol | Shape | Constraint | Role |
|---|---|---|---|
| `F` | p × k | **signed** (free) | Feature loadings. `F_fk` = log-fold change in feature *f* per unit activity of factor *k*. |
| `G` | n × k | **≥ 0** | Per-sample factor usage. |
| `a` | p | free | Feature intercepts |
| `b` | n | free or fixed offset | Per-sample log exposure |
| `Z` | n × q | given | Covariate design matrix (batch, run, instrument, any measured nuisance variable) |
| `γ` | p × q | free | Per-feature covariate coefficients |
| `θ` | p | > 0 | Per-feature NB dispersion (NB2: `Var = μ + μ²/θ`) |

Negative log-likelihood, summed over all `(f,s)`:

```
NLL = −Σ [ log Γ(x+θ) − log Γ(θ) − log Γ(x+1)
           + θ log(θ/(θ+μ)) + x log(μ/(θ+μ)) ]
```

Objective:

```
minimize  NLL(F, G, a, b, γ; θ)  +  λ · ‖F‖₁
```

### Why these constraints (do not relax them without asking)

- **`F` signed** is the point of the model. Real latent processes both elevate and suppress features. A non-negative loading cannot represent suppression; forcing it splits one process into an elevated component plus a separately-down-weighted "shadow" component — roughly doubling the factor count and creating strong artifactual anticorrelations between usages. The purpose here is one factor per process.
- **`G ≥ 0`** is what makes the factorization identifiable. Under a signed `F` alone, `FGᵀ = (FA)(GA⁻ᵀ)ᵀ` for any invertible `A` — the factors are arbitrary. Restricting `G` to the non-negative orthant reduces the ambiguity to permutation × positive diagonal, because monomial matrices with positive entries are the only invertible maps preserving that orthant. It also eliminates the sign ambiguity in `F` (the diagonal must be positive), which matters for matching factors across runs. Crucially, it constrains *support*, not dependence — usages remain free to correlate arbitrarily, which downstream statistics require.
- **`log` link** gives exact additivity in log-mean and an automatic floor as suppression strengthens (`μ → 0`). Do not substitute a linear or ReLU link.
- **L1 on `F`** does double duty: it regularizes separation (§5.2) and, since sparsity is not rotation-equivariant, it strengthens identifiability where the orthant constraint alone is marginal. It is not optional.

---

## 2. API

```python
class NBGLMSemiNMF:
    def __init__(
        self,
        n_components: int,
        l1_F: float = 0.0,
        exposure: str | np.ndarray = "fit",   # "fit" | "offset" | array
        dispersion: str = "trend",            # "trend" | "feature" | "shared" | float
        init: str = "svd",                    # "svd" | "nmf" | "random" | (F0, G0)
        max_iter: int = 500,
        tol: float = 1e-5,
        batch_size: int | None = None,        # None = full batch
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ): ...

    def fit(self, X, Z=None) -> "NBGLMSemiNMF": ...
    def transform(self, X, Z=None) -> np.ndarray:
        """Fit G for new samples with F, a, gamma, theta held fixed."""
    def fit_transform(self, X, Z=None) -> np.ndarray: ...
```

**Inputs.** `X` accepts `scipy.sparse` CSR/CSC or a dense `ndarray`, oriented features × samples. **Raw integer counts only** — the model does its own normalization via `b`. Raise on non-integer input rather than silently rounding. `Z` is an `ndarray`, or a list of column names when a labeled container is passed; categorical columns get one-hot encoded with a dropped reference level.

Optional adapters for labeled containers (`pandas.DataFrame`, and `AnnData` if installed) may be provided as a thin convenience layer, but the core must not depend on them and must not assume any domain-specific field conventions.

**Fitted attributes.**

| Attribute | Shape | Notes |
|---|---|---|
| `F_` | p × k | Signed loadings, columns unit L2 norm |
| `G_` | n × k | Usages, ≥ 0, scale absorbed from `F` normalization |
| `G_raw_` | n × k | Pre-softplus `G̃`. **Must be exposed** — downstream rank-based statistics need a tie-free continuous variable, and `G_` has exact zeros. |
| `a_`, `b_`, `gamma_`, `theta_` | | |
| `loss_`, `n_iter_`, `converged_` | | Per-iteration loss trace |
| `deviance_explained_` | float | Vs. an intercept-plus-covariates null |
| `component_stats_` | DataFrame | Per factor: usage fraction non-zero, mean usage, deviance explained, fraction of loading mass negative, n features above a loading threshold |

`component_stats_` is not cosmetic — downstream steps filter and characterize factors using the non-zero-usage fraction and the negative-mass fraction. Include it.

**Ordering.** Sort factors by descending deviance explained so indices are comparable across runs before any explicit matching step.

---

## 3. Fitting

Block-alternating minimization. Each block is smooth, so either quasi-Newton or first-order methods work; the choice is yours to benchmark.

**Parametrization.** `G = softplus(G̃)` with `G̃` unconstrained. Preferred over projection/clamping because gradients don't die at the boundary and `G̃` is needed as an output anyway. Projected gradient with a clamp is an acceptable alternative *if* `G_raw_` is still produced — but benchmark both, since boundary behaviour differs materially.

**Scale fixing.** After each outer iteration, rescale columns of `F` to unit L2 norm and push the scale into the corresponding column of `G`. Do this *before* evaluating the convergence criterion, or the loss trace will show phantom movement. Note the L1 penalty interacts with this: `‖F‖₁` is not scale-invariant, so apply the penalty to the *normalized* `F` to keep λ meaningful.

**Dispersion.** Estimate `θ` by method of moments on current residuals, then hold fixed for a stretch of outer iterations before re-estimating (re-estimating every iteration is unstable). `"trend"` fits a mean–dispersion trend and shrinks per-feature estimates toward it — this should be the default; per-feature MLE is noisy for low-count features.

**Exposure.** `"offset"` computes `b_s = log(total counts_s / median total)` and holds it fixed. `"fit"` estimates `b` jointly. Offset is the safer default for stability; fitting can absorb structure that should land in `G`. Support both and document the difference.

**Initialization.** `"nmf"` runs a quick non-negative fit on `log1p` of exposure-normalized counts and uses it as a warm start — expected to be the most reliable. `"svd"` uses a truncated SVD of the same, with `G` set to the positive part. Warm-starting matters more than optimizer choice; benchmark this.

**Minibatching.** `G` is row-separable over samples, so the `G`-block minibatches cleanly. The `F`-block needs a full pass or accumulated gradients. Implement `batch_size` for `n > ~10⁵`.

**Convergence.** Relative change in penalized objective below `tol` for 3 consecutive outer iterations. Also stop at `max_iter` with `converged_ = False` and a warning.

---

## 4. Scale and performance

Sizing this for the intended usage pattern: **many independent fits over data strata, not one large fit.**

| Dimension | Typical | Upper |
|---|---|---|
| `p` (features, post-selection) | 2,000–5,000 | 10,000 |
| `n` (samples per fit) | 10³–10⁵ | 10⁶ |
| `k` | 5–15 | 50 |
| Number of independent fits | hundreds | tens of thousands |

Per-fit wall time is the binding cost, not peak scale. A single fit at p=3000, n=20000, k=10 should complete in well under a minute on GPU. Optimize for that regime.

GPU via PyTorch or JAX; CPU fallback required and must be correct, not merely present. Sparse input should stay sparse until the point it can't — materializing a dense p × n `μ` is the obvious memory trap.

---

## 5. Numerical requirements

These are the failure modes most likely to produce silently wrong results.

### 5.1 Overflow

`μ = exp(Θ)` overflows readily during early iterations. Clamp `Θ` to a safe range (e.g. `[-30, 30]`) and use `gammaln`, `log1p`, and `logsumexp`-style stabilized forms throughout the NLL. Use the stable softplus (`log1p(exp(-|x|)) + max(x,0)`).

### 5.2 Separation

For a feature that is zero in every sample where factor *k* is active, the likelihood is monotone in `F_fk → −∞`. Unpenalized, this diverges.

The L1 penalty is the primary control. Additionally: hard-clip `|F|` at a generous bound and **warn** when any entry hits it, since that indicates λ is too small rather than that the clip is doing useful work. Test for this explicitly (§6).

### 5.3 Degenerate factors

Factors can collapse to all-zero usage or duplicate another factor. Detect both (usage norm below threshold; loading correlation above threshold) and report in `component_stats_`. **Do not silently drop or reinitialize them** — the caller needs to know it happened, because a run that drops factors is evidence about `k`, not a nuisance to hide.

### 5.4 Reproducibility

Given `random_state`, results must be bit-reproducible on the same device. Document that CPU and GPU results will differ in the last digits.

---

## 6. Validation suite

This is the most important deliverable after the model itself, because the correctness criterion is not "loss goes down" — it is "the recovered factors are the ones that generated the data." Tests should run against synthetic data from the model's own generative process.

**Synthetic data generator.** A helper that draws `F` (signed, sparse), `G` (non-negative, sparse), `a`, `b`, `θ`, then samples `X ~ NB`. Parameterize by p, n, k, sparsity, dispersion level, and a `negative_loading_fraction` controlling how much of `F`'s mass is negative. Ship it — the tests need it, and so does anyone extending the code.

Required tests:

1. **Recovery.** Fit on synthetic data; match recovered factors to true factors by loading correlation (Hungarian assignment); assert mean correlation above threshold. Run across a grid of noise and sparsity levels.
2. **Signed recovery.** With `negative_loading_fraction > 0`, assert negative loadings are recovered with correct sign and approximately correct magnitude. This is the test that distinguishes this model from non-negative factorization — make it strict.
3. **Sign-ambiguity absence.** Across random restarts, assert matched factors never appear sign-flipped. The `G ≥ 0` constraint should guarantee this; failure means the constraint isn't being enforced correctly somewhere.
4. **Rotation stability.** Across restarts with different seeds on the same data, assert matched loading correlations exceed threshold. Report the distribution, not just pass/fail — this number is the practical identifiability of the implementation, and downstream work needs to know it.
5. **Null.** On data generated with `k_true = 0` (intercepts and covariates only), assert fitted factors explain negligible deviance.
6. **Exposure invariance.** Multiply all counts for a subset of samples by a constant and resample; assert `G` is approximately unchanged. Catches exposure/offset bugs, which are the highest-probability failure mode.
7. **Covariate absorption.** Generate data with a strong batch effect; assert that with `Z` supplied no factor correlates strongly with batch, and that without `Z` one does. Direct test that `Zγ` is doing its job.
8. **Separation.** Construct a feature that is structurally zero whenever a factor is active; assert `‖F‖∞` stays bounded and the clip warning fires when λ = 0.
9. **Rank recovery.** Fit at `k > k_true`; assert surplus factors show up as degenerate in `component_stats_` rather than fragmenting real ones.
10. **Gradient check.** Finite-difference the analytic gradients of each block on a small problem.
11. **CPU/GPU agreement** within tolerance; **sparse/dense agreement** exactly.

---

## 7. Benchmark script

Separate from tests. Should report, on both synthetic data and at least one real public count dataset:

- Wall time and peak memory vs. n, p, k
- Convergence trace comparison across `init` options
- Convergence trace comparison across optimizers tried
- Reconstruction deviance vs. NMF and vs. GLM-PCA at matched `k`
- Rotation stability (test 4) on real data, since this will be worse than on synthetic

The `init` and optimizer comparisons exist because the right choices are genuinely unknown here — treat that as an open question to resolve empirically and report, not a decision to make silently.

---

## 8. Deliverables

- Installable package, typed, `scikit-learn`-style API as in §2
- Test suite per §6, running in CI on CPU
- Benchmark script per §7 with results committed as a short markdown report
- README: model definition, the identifiability rationale from §1, worked example, and a documented statement of what is *not* handled (§9)
- Docstrings with shapes and constraints on every public method

---

## 9. Explicitly out of scope

Do not build these; they belong to the calling code:

- Choice of `k` — the model reports diagnostics, the caller decides
- Consensus or multi-restart aggregation across fits
- Matching or clustering factors across datasets
- Data cleaning, QC, stratification, or feature selection
- Any downstream statistics on `G`
- Any annotation, enrichment, or interpretation of `F`

The one thing worth designing *for* without implementing: the caller will run this hundreds to thousands of times over data strata and then match factors across the results. That argues for cheap per-fit cost, deterministic seeding, and factor outputs directly comparable across runs (hence the column normalization and consistent ordering in §2).

---

## 10. Open questions for the implementer

Resolve empirically and report; no need to ask first.

1. Optimizer per block — L-BFGS vs. Adam vs. IRLS/Fisher scoring. IRLS has the nicest theory but the `G ≥ 0` constraint complicates it.
2. Softplus reparametrization vs. projected gradient. Boundary behaviour differs and affects how many exact zeros land in `G_`.
3. Whether joint optimization of all blocks beats alternating at this scale.
4. Whether `θ` re-estimation should be on a fixed schedule or triggered by convergence of the mean model.
5. Whether `b` as fitted parameter vs. fixed offset changes recovered factors materially on real data — test 6 covers correctness, but this is about whether structure leaks into `b`.
