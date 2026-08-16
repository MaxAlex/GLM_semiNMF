# NB-GLM Semi-NMF: solution method

This document describes the estimation method implemented in `glm_seminmf`:
the model and objective, the identifiability analysis that drives several
non-obvious design choices, the numerically stabilized likelihood, the
optimization scheme (including the failure modes found during development and
their fixes), and open questions with possible improvements.

Companion documents: `README.md` (user-facing overview),
`benchmarks/RESULTS.md` (empirical comparisons), the implementation spec
(`nb-glm-seminmf-implementation-spec.md`).

---

## 1. Model

Counts `X ∈ ℕ^{p×n}` (features × samples) are modeled as negative binomial
(NB2) with a log-linear mean:

```
X_fs ~ NB(μ_fs, θ_f)                Var[X_fs] = μ_fs + μ_fs²/θ_f
η_fs = log μ_fs = a_f + b_s + (γ Zᵀ)_fs + Σ_k F_fk G_sk
```

- `F ∈ ℝ^{p×k}` — **signed** feature loadings; `F_fk` is the log-fold change
  of feature `f` per unit activity of factor `k`.
- `G ∈ ℝ₊^{n×k}` — **non-negative** per-sample usages.
- `a ∈ ℝ^p` — feature intercepts (baseline log abundance).
- `b ∈ ℝ^n` — per-sample log exposure; by default the fixed offset
  `b_s = log(total_s / median total)`, optionally estimated.
- `Z ∈ ℝ^{n×q}`, `γ ∈ ℝ^{p×q}` — known nuisance design and its per-feature
  coefficients (batch, instrument, ...).
- `θ ∈ ℝ₊^p` — per-feature NB dispersion, estimated by method of moments with
  a mean–dispersion trend (§6), not by gradient descent.

### Objective

```
minimize over (F, G, a, b, γ):
    NLL(F, G, a, b, γ; θ)  +  λ_F ‖F‖₁  +  λ_G Σ_sk G_sk
```

with `G = softplus(G̃)` parameterizing the constraint (§5.1). `λ_F` is the
spec's L1 on loadings (separation control + identifiability; §4.4). `λ_G` is
a small usage-anchoring term **added beyond the spec objective** for an
identifiability reason discovered empirically (§3.3); its default `0.005·p`
is small relative to the likelihood, and `λ_G = 0` recovers the bare spec
objective.

The NLL is the exact NB2 negative log-likelihood summed over all `p·n`
entries, so both penalties are on the summed-likelihood scale: `λ_F` should
grow roughly with `n` (≈ `0.001·n` is a good default; `0.01·n` demonstrably
over-penalizes), while `λ_G` scales with `p` because each usage entry
influences `p` likelihood terms.

---

## 2. Why these constraints (recap of the modeling rationale)

- **Signed `F`** is the point of the model: latent processes both elevate and
  suppress features. Non-negative loadings cannot express suppression;
  forcing them splits one process into an "up" component plus a shadow
  "down" component, inflating the factor count and inducing artifactual
  anticorrelations between usages.
- **`G ≥ 0`** is what makes the factorization identifiable at all (§3).
- **Log link** gives exact additivity of effects in log-mean and a natural
  floor as suppression strengthens (`μ → 0`).
- **NB2** absorbs feature-level overdispersion so the factors are not spent
  explaining variance that is really noise; with `θ → ∞` the model degrades
  gracefully to Poisson.

---

## 3. Identifiability

The likelihood depends on `(F, G)` only through `F Gᵀ`, so any invertible
`A ∈ ℝ^{k×k}` with `F' = F A⁻ᵀ`, `G' = G A` leaves the fit unchanged. The
implementation resolves this ambiguity in layers:

### 3.1 Orthant constraint → permutation × positive diagonal

Restricting `G` to the non-negative orthant admits only transformations `A`
with `G A ≥ 0` for the achieved `G` (and `A⁻¹` likewise for the reverse
direction). When the columns of `G` genuinely touch the boundary (each factor
is *off* in some samples), the only invertible maps preserving the orthant
are positive monomial matrices — permutation × positive diagonal. This kills
both rotations and sign flips: a sign flip `F_k → −F_k` would require
`G_k → −G_k < 0`. Crucially, the constraint restricts *support*, not
dependence: usages may correlate arbitrarily, which downstream
covariance-based statistics require.

### 3.2 Scale fixing

The remaining positive-diagonal freedom is removed by convention: after every
outer iteration, each column of `F` is rescaled to unit L2 norm and the scale
is pushed into the corresponding column of `G`
(`F_k ← F_k/c_k`, `G_k ← c_k G_k`, `c_k = ‖F_k‖₂`). Combined with ordering
factors by deviance explained, this makes factors directly comparable across
runs and datasets. The L1 penalty is evaluated on the (perpetually
re-normalized) `F`, keeping `λ_F` meaningful despite the penalty not being
scale-invariant.

### 3.3 Shift flatness — a gap in the orthant argument (found empirically)

The predictor is *exactly* invariant under a per-factor usage shift absorbed
by the intercepts:

```
G_k → G_k + c_k · 1,    a → a − c_k F_k        (any c_k ≥ −min_s G_sk)
```

since `Σ_k F_fk (G_sk + c_k) = (F Gᵀ)_fs + (F c)_f`. The likelihood is
therefore flat along `k` directions, and the flatness interacts badly with
§3.1: an optimizer following this manifold drifts `G` *dense* (all entries
interior), at which point the boundary-touching premise of the monomial-matrix
argument fails and rotational ambiguity partially returns. This is not
hypothetical — with nothing anchoring the shift, factor recovery on synthetic
data dropped to 0.55–0.85 (mean matched |loading correlation|) and became
chaotically sensitive to float rounding (different BLAS thread counts landed
in different near-equivalent optima).

Two complementary mechanisms anchor the shift at the *touch-zero
representative* of each column (the choice that maximizes boundary contact
and restores §3.1):

1. **Min-shift canonicalization**: each outer iteration, alongside scale
   fixing, `c_k = min_s G_sk` is subtracted from column `k` and `F c` is
   added to `a` — exactly likelihood-invariant, so it never fights the
   optimizer.
2. **A small L1 on `G`** (`λ_G Σ G`, default `0.005·p`): along the flat
   manifold, `Σ_s (G_sk + c)` is minimized exactly at the touch-zero point,
   so the penalty selects the same representative *smoothly*, giving the
   optimizer a consistent gradient signal rather than a once-per-iteration
   jump, and keeping usage supports genuinely sparse.

With both in place, recovery on the hard synthetic grid rose to 0.88–0.94,
fitted usage sparsity matched the generating sparsity, and results became
stable across devices and thread counts. This is the one deliberate extension
of the spec's objective, and it is user-controllable (`l1_G=0.0` disables).

### 3.4 L1 on `F`

Sparsity is not rotation-equivariant: among near-equivalent rotations of a
solution, `‖F‖₁` prefers the sparse representative. The penalty thus both
regularizes (§4.4, separation) and sharpens identifiability where the orthant
constraint alone is marginal.

### 3.5 What remains unidentified

If a factor's true usage support is dense (active in every sample), §3.1's
premise fails for that factor even with anchoring, and its recovery relies
on the L1 tie-break alone. Empirically the practical identifiability on
matched synthetic data is ≈0.9 mean loading correlation across restarts, not
1.0; `benchmarks/RESULTS.md` reports the distribution (spec test 4 reports it
per run).

---

## 4. Numerically stabilized likelihood

### 4.1 Stable NLL

From the NB2 pmf
`P(x) = Γ(x+θ)/(Γ(θ) x!) · (θ/(θ+μ))^θ (μ/(θ+μ))^x`, using
`log(θ+μ) = logaddexp(log θ, η)` and
`logaddexp(u,v) = u + softplus(v−u)`:

```
−log P = −lgamma(x+θ) + lgamma(θ) + lgamma(x+1)
         + θ · softplus(η − log θ)                  ← −θ log(θ/(θ+μ))
         − x · (η − logaddexp(log θ, η))            ← −x log(μ/(θ+μ))
```

No raw `exp(η)` appears anywhere in the training objective; the only
exponentials live inside `softplus`/`logaddexp`, which are overflow-safe.
The `lgamma` terms are constant in the mean parameters and are dropped inside
the F/G/a/b/γ blocks (they matter for the reported loss trace, which must be
comparable across θ updates, and are included there).

The gradient in the predictor is the standard GLM form

```
∂(−log P)/∂η = θ (μ − x) / (θ + μ),
```

obtained here by autograd through the softplus forms (verified against
central finite differences in float64, and the NLL itself against
`scipy.stats.nbinom.logpmf` to 1e-10).

### 4.2 Straight-through predictor clamp

`η` is clamped to `[−30, 30]` (spec 5.1) — but with a **straight-through
gradient**: the forward pass uses the clamped value, the backward pass treats
the clamp as identity. A hard clamp would zero the gradient of any entry
outside the box, stranding it there; with straight-through, the NLL gradient
at the boundary points back inside, so excursions self-correct.

### 4.3 Softplus parametrization of `G ≥ 0`

`G = softplus(G̃)` with `G̃` free. Gradients do not die at the boundary
(unlike a clamp), and `G̃` doubles as the tie-free continuous usage variable
(`G_raw_`) that downstream rank statistics need. The stable inverse is
`softplus⁻¹(y) = log(expm1 y) = y + log1p(−e^{−y})`, branch-selected by
magnitude. Because softplus never reaches 0 exactly, the reported `G_`
snaps entries below `max(1e-8, 1e-3 · column max)` to exact zeros.

A projected-gradient alternative (`g_parametrization="projected"`: free `G`,
clamped to the orthant after each step) is implemented; it benchmarks
slightly better on recovery but its raw values contain exact-zero ties, so
softplus remains the default (see RESULTS).

### 4.4 Separation control

For a feature that is zero in every sample where factor `k` is active, the
likelihood is monotone in `F_fk → −∞`. `λ_F` is the primary control. The
backstop is a hard clip — but **not on `|F|`**: with unit-norm columns,
`|F_fk| ≤ 1` always, so the spec's raw-value clip is unreachable, and the
divergence instead manifests as the entry swallowing its whole column while
the column scale migrates into `G`. The scale-invariant quantity that
actually diverges is the factor's contribution to the log-mean, so the clip
is

```
|F_fk| · max_s G_sk  ≤  15        (log-fold change at maximal activity)
```

applied per column after every F step. If any entry sits at this bound at
convergence, the fit warns: it means `λ_F` is too small, not that the clip is
doing useful work.

### 4.5 Deviance

Model quality is reported as NB deviance (the `lgamma` terms cancel between
the saturated and fitted models):

```
D = 2 Σ [ x log(x/μ) − (x+θ) log((x+θ)/(μ+θ)) ]        (x log x := 0 at x=0)
```

`deviance_explained_ = 1 − D_model / D_null`, where the null (intercepts +
exposure + covariates, `k = 0`) is refit with the model's θ held fixed so the
two deviances are computed under the same noise model. Per-factor deviance
explained is a no-refit ablation: factor `k`'s term is removed from `η` and
the deviance increase is reported relative to `D_null` (fast, and monotone
enough for ordering; it is not a partitioning — values need not sum to the
total).

---

## 5. Optimization

### 5.1 Block-alternating Adam with chunked full-batch gradients

Each outer iteration:

1. **G-block** — `inner_steps` (default 5) Adam steps on `(G̃ [, b])`, all
   other parameters frozen;
2. **F-block** — `inner_steps` Adam steps on `(F, a [, γ])`;
3. canonicalization (scale + shift, §3.2–3.3);
4. periodic dispersion refresh (§6);
5. objective evaluation and convergence/lr bookkeeping.

Both blocks are *row-separable* (G over samples, F/a/γ over features), which
is what makes the alternating scheme cheap; the code nevertheless accumulates
gradients over sample chunks before each step, so semantics are exactly
full-batch regardless of `batch_size` — chunking bounds memory (a dense
`p × n` μ is never materialized; peak working set is `p × chunk`), it does
not introduce stochasticity. `X` is kept resident on the device when it fits
in ~35% of free VRAM (queried at fit time), otherwise streamed per chunk from
a float32-staged host copy.

Two variants ship behind `algorithm=`: `"adam_joint"` (one Adam over all
blocks — benchmarks faster at scale with equal quality) and `"lbfgs"`
(strong-Wolfe L-BFGS per block — included for reference; it line-searches
into the clamped region of the objective and stalls under the
revert-on-non-finite guard, so it is not competitive as implemented).

### 5.2 Proximal handling of the L1 terms

`λ_G Σ softplus(G̃)` is smooth — it simply joins the G-block loss. `λ_F ‖F‖₁`
is handled by proximal soft-thresholding after each Adam step on F, with the
threshold matched to Adam's per-coordinate effective step size:

```
thr_i = lr · λ_F / (√v̂_i + ε),      F_i ← sign(F_i) · max(|F_i| − thr_i, 0)
```

**Stability caveat (found the hard way):** when a factor weakens, its
gradients vanish, `v̂ → 0`, and the raw threshold explodes — wiping the
column and destabilizing the whole fit (a bistable death spiral in which
factors were randomly lost). The denominator is therefore floored at
`0.1 × median(√v̂ + ε)` over F, capping any coordinate's threshold at 10× the
typical one.

### 5.3 Learning-rate schedule and convergence

Full-batch Adam at fixed lr oscillates near the optimum above any tight
relative tolerance, so the lr for all blocks is halved whenever the best
objective has not improved for 10 outer iterations (geometric decay lets the
criterion bind). Convergence is declared when the relative change of the full
penalized objective stays below `tol` for 3 consecutive outer iterations —
evaluated *after* canonicalization, so scale/shift bookkeeping cannot
masquerade as loss movement. Hitting `max_iter` sets `converged_ = False` and
warns.

**Best-iterate restore:** the fit snapshots parameters at every new best
objective and restores the snapshot if the final iterate is worse. This is
cheap insurance (one parameter copy) that turned out to matter: see §6.1.

### 5.4 Initialization

Warm-starting matters more than optimizer choice (confirmed empirically —
random init lands in materially worse optima). The default `"svd"` init:

1. `Y = log1p(X / exp(b))` — log1p of exposure-normalized counts (sparse-safe);
2. if covariates are present, `Y` is regressed on centered `Z` (OLS) and the
   coefficients both warm-start `γ` and are residualized out — otherwise the
   initial factors absorb the batch structure and fitted `γ` never fully
   displaces it (this fixed a covariate-confounding failure);
3. truncated SVD of the row-centered residual (implicit centering via a
   `LinearOperator`, so sparse `Y` is never densified);
4. each component is oriented so the usage side carries more positive than
   negative mass; `G₀` is the positive part of `V S`, `F₀ = U`.

`"nmf"` (internal multiplicative-update NMF on `Y`) and `"random"` are
available; `(F₀, G₀)` tuples are accepted. Initial usages receive a seeded
relative jitter of ~1e-4 — clamped-to-floor init entries would otherwise stay
bit-identical forever (dead entries take sign-saturated Adam steps of exactly
−lr in lockstep), producing ties in `G_raw_`.

### 5.5 `transform` (new samples)

`F, a, γ, θ` frozen; per-sample exposure from the stored median (or refit
when the model used `exposure="fit"`); `G̃` warm-started from the positive
part of `(log1p-normalized residual)ᵀ F` and optimized alone with the same
machinery at 4× the fit lr (the G-only problem is better conditioned and
crawls at the fit-time lr). With `a` fixed, the shift direction of §3.3 is
*not* flat, so transform needs no canonicalization and `λ_G` keeps fit and
transform objectives consistent.

---

## 6. Dispersion estimation

### 6.1 Method of moments, trend shrinkage — and a feedback instability

With `α = 1/θ`, NB2 gives `E(x−μ)² = μ + α μ²`, so per feature

```
α̂_f = ( Σ_s (x_fs − μ_fs)² − Σ_s μ_fs ) / Σ_s μ_fs²        (clamped to [1e-4, 1e3])
```

computed chunk-wise from the current fit. The default `"trend"` mode fits a
quadratic of `log α̂` on centered `log mean` (winsorized at the 1%/99%
quantiles so separation-level outliers don't steer it; solved by normal
equations — `torch.linalg.lstsq` is avoided because its first call in a
process differs bitwise from later calls, which silently broke seed
reproducibility) and shrinks per-feature estimates toward the trend with an
information weight `w_f = S_f/(S_f + 50)`, `S_f = Σ_s μ_fs` — features with
few expected counts, whose raw moment estimates are noise, get pulled
strongly to the trend.

**Instability (found empirically, the most consequential bug of the
project):** re-estimating θ on a fixed schedule forever creates a positive
feedback loop with the mean model. Any unexplained variance lowers θ; lower θ
flattens the likelihood, so the fixed penalties out-pull the data term and
the factors shrink; that raises the residuals, lowering θ further. On some
float trajectories this manifested as a perfectly regular limit cycle that
never met the stall criterion, with `‖F‖₁`, `Σ G`, and θ all decaying
smoothly together while the loss climbed ~6% above its own minimum. The
ablation was decisive: fixed θ → zero drift and clean convergence; θ updates
→ drift, with or without `λ_G`.

The schedule is therefore **bounded by construction**: θ refreshes every 10
outer iterations, and freezes permanently at the first of (a) `max |Δ log θ|
< 0.05` (stabilized), (b) 15 refreshes spent (budget), or (c) the objective
stalling (one final refresh, then convergence is judged at fixed θ). The
best-iterate restore (§5.3) additionally guarantees a fit can never return a
point worse than one it already visited, whatever the θ trajectory did.

### 6.2 Other modes

`"feature"` (raw per-feature MoM), `"shared"` (single pooled θ), or a fixed
float. All estimation happens outside autograd.

---

## 7. Reproducibility engineering

Bit-reproducibility given `(seed, device)` required several specific
mitigations, each of which silently broke it before being found:

| Source | Mitigation |
|---|---|
| `torch.linalg.lstsq` first-call vs later-call bitwise difference | normal equations + `torch.linalg.solve` in the trend fit |
| CSC `toarray()` returns F-order; BLAS accumulates differently over F- vs C-order memory | `ascontiguousarray` on densification |
| sparse vs dense matvec accumulation order (`X @ v`) | sparse inputs ≤ 5·10⁷ elements are densified up front — the fitting engine holds them dense on-device anyway, so one code path costs nothing; larger sparse inputs stay sparse and agree with dense to float tolerance only (documented) |
| clamped init usages starting bit-identical and moving in lockstep | seeded init jitter (§5.4) |
| float32 cancellation in canonicalization creating ties among dead usages | canonicalization arithmetic in float64 (`G` is only n×k) |

CPU and GPU agree to float tolerance (matched loading correlation > 0.98 in
the suite; identical to ~7 significant digits on the diagnostic problems),
not bitwise — documented.

---

## 8. Complexity and memory

Per gradient pass: one `p×k · k×chunk` matmul plus elementwise NLL over
`p × chunk`, i.e. `O(p·n·k)` flops per full pass and `O(p·chunk)` peak
working set. An outer iteration costs `2·inner_steps + 1` passes
(≈11 by default). Typical fits converge in 100–250 outer iterations.
Everything else (canonicalization, prox, dispersion) is `O(n·k + p·k + p)`
per iteration and negligible.

The p=3000, n=20000, k=10 reference fit runs in well under a minute on a
consumer GPU once the fitter converges normally (see RESULTS for measured
numbers); the pre-stabilization version burned its full 500-iteration budget
without converging, which is what the θ-schedule and lr-decay fixes bought.

---

## 9. Open questions and possible improvements

**Optimization**

- *Joint vs alternating default.* Joint Adam benchmarks ~1.4× faster with
  equal-or-better recovery at scale. Alternating is the shipped default
  mostly for its cleaner block semantics; flipping the default is a
  low-risk improvement once confirmed across the full grid.
- *A working second-order method.* The per-row subproblems are small GLMs;
  IRLS/Fisher scoring per feature row (F-block) is classical and the row
  Hessians are `k×k` — cheap. The `G ≥ 0` constraint complicates the G-block
  (each row becomes a tiny non-negative GLM, amenable to projected Newton or
  NNLS-style active sets). The current L-BFGS attempt fails on line search
  into the clamped region; a trust-region variant, or L-BFGS on the
  *softplus-reparameterized* smooth objective with a damped line search,
  might do better. Expected payoff: fewer outer iterations at large `n`.
- *True minibatch mode.* Gradients are currently exact full-batch. For
  `n ≳ 10⁶`, stochastic G-updates (exact, since rows are separable) with
  variance-reduced F-updates (SAGA-style or simply large minibatches) would
  cut per-iteration cost; the DataSource streaming layer already provides the
  chunk plumbing.
- *lr schedule.* The plateau-halving schedule works but was tuned by
  observation; cosine or Adam-with-warmup variants were not explored.

**Statistics**

- *Dispersion estimation.* MoM + trend is fast and adequate, but the
  θ-feedback analysis (§6.1) suggests the deeper fix is a *joint* criterion:
  either profile-likelihood θ updates (Cox–Reid adjusted, as in edgeR) which
  are less biased by fitted means, or explicit damping of the θ update. The
  refresh budget is a pragmatic bound, not a principled one; the open
  question is whether adjusted-profile-likelihood updates would converge as a
  bona fide block of the objective (they optimize a different function, so
  the alternation loses its single-Lyapunov-function structure — same root
  issue, better behaved in practice in the DGE literature).
- *Penalty selection.* `λ_F ≈ 0.001·n` and `λ_G = 0.005·p` are empirical
  defaults. Principled selection (stability selection across seeds, BIC-type
  criteria on the NB likelihood, or λ-paths with warm starts — cheap here
  since fits warm-start well) is unimplemented and would remove the main
  user-facing tuning burden.
- *Uncertainty.* No standard errors are provided. Per-row observed-information
  sandwich estimates for `F` rows (conditional on `G`, θ) would be cheap and
  useful for downstream thresholding of loadings; fully joint uncertainty is
  a research problem (the factorization's discrete ambiguities complicate
  bootstrap aggregation, though the canonicalization makes within-basin
  bootstraps meaningful).

**Identifiability**

- *Dense-support factors* (§3.5) remain the weak case: when a factor is
  active in every sample, the orthant constraint never binds for it. Possible
  strengthenings: anchor-feature conditions (separability-style assumptions,
  as in topic modeling), or a mild usage-sparsity prior beyond the current
  L1. The right choice depends on whether downstream analyses can tolerate
  the induced bias.
- *Shift anchoring alternatives.* The touch-zero convention is one canonical
  choice; a quantile-zero convention (e.g., 1st percentile at zero) would be
  more robust to single-sample outliers at the cost of exact likelihood
  invariance of the canonicalization step. The smooth `λ_G` component would
  need matching changes.
- *Relation to classical semi-NMF* (Ding–Li–Jordan): their multiplicative
  updates solve the Frobenius version of this problem; the GLM likelihood
  breaks the closed-form updates but the identifiability geometry (orthant +
  scale) is the same, and their analysis of when semi-NMF equals k-means
  suggests degenerate regimes (near-duplicate factors) that
  `component_stats_` flags but the optimizer does not avoid.

**Engineering**

- *Projected-G default.* Projected gradient slightly beats softplus on
  recovery in benchmarks; adopting it as default requires preserving a
  tie-free raw-usage output (e.g., storing the last pre-projection iterate),
  since its parameter values contain exact-zero ties.
- *Warm starts across strata.* The intended usage pattern is hundreds of
  independent fits over data strata followed by factor matching. Fits could
  warm-start from a reference stratum's `F` (via the `(F₀, G₀)` init path),
  which would likely both speed convergence and improve cross-stratum factor
  correspondence — deliberately left to the caller (spec §9), but the API
  supports it.
- *Multi-GPU / fit batching.* Many small independent fits would saturate a
  GPU better if batched (vmap-style over strata); unexplored.
