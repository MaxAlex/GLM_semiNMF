# Remaining work

Status as of 2026-09-18, after the optimizer repair described in
[OPTIMIZER_PLAN.md](OPTIMIZER_PLAN.md) and
[OPTIMIZER_CHANGELOG.md](OPTIMIZER_CHANGELOG.md). Every item below has evidence
behind its priority; the measurements are in
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md).

Nothing here blocks using the library. The solver optimizes one declared
objective, certifies the parameters it returns, and reports honestly when it
cannot. 94 tests pass.

## Finished

All five units of the plan, plus two things the measurements turned up:

- Objective/constraint contract, constrained exact-L1 steps, physical KKT
  certification, initialization and checkpoint protection, dispersion phases.
- Separation safeguard reworked from a contribution bound into a divergence
  detector, after it was found aborting ordinary fits on transient overshoot.
- Dispersion estimator fixed: shrinkage now keys on resolvable variance
  excess rather than count totals.
- Certified CPU and GPU penalty grids, initialization, exposure and
  20 Newsgroups comparisons.
- `numerically_stationary`, so fits at the arithmetic's limit stop being
  reported as failures.

## Open

### 1. A common dispersion for scoring baselines

**Why it matters:** the NMF and GLM-PCA comparison cannot currently be quoted.
Baselines are scored with *our* fitted theta, so the identical NMF fit scored
+0.0611 against one of our models and -0.0227 against another. The comparison
moves when our model moves.

**What to do:** score all methods with a single model-independent dispersion.
The null-model estimate (intercepts plus exposure, already computed inside
`_null_deviance`) is the natural neutral choice. Keep `deviance_explained_`
itself as documented for our own model, and report the common-theta score
separately as the comparable metric.

**Also in scope:** `_glmpca_eta` in `benchmarks/bench.py` reconstructs
GLM-PCA's predictor from an assumed convention that has never been verified.
Its -0.47 could be the reconstruction rather than the method.

### 2. Trend quality at the extremes of the mean range

**Why it matters:** one problem of sixteen (p=600, seed 12) fails to certify
because the feature with the largest mean takes the trend entirely (shrink
weight 0) and the trend is poor there — theta 1989 against a true 8.9.

**What to do:** weight the trend fit by information, or handle the ends
explicitly. Clamping the trend to its fitted response range was tried and
reverted: it did not bind, changed no aggregate, and flipped one problem each
way at the tolerance boundary.

### 3. The `l1_G="auto"` heuristic on sparse data — a decision, not a task

**Why it matters:** `0.005*p` is calibrated on the generator's count scale. On
20 Newsgroups (2.45% nonzero, mean count 0.044) it drives every usage to zero:
10/10 factors degenerate, deviance explained 0.0005, and the fit certifies —
correctly, since at `G = 0` the penalty gradient exceeds the likelihood's pull.
The diagnostics catch it, but the default's premise that the penalty is "small
relative to the likelihood" does not hold at that count scale.

**Options:** leave it and document (done in the README), make the heuristic
count-aware, or warn when every factor is flagged degenerate. Changing it is a
regularization default, which this work deliberately did not touch, so it needs
an explicit decision.

### 4. The p=3000, n=20000, k=10 target

**Why it matters:** missed by roughly an order of magnitude once stationarity
is required — 5.57 s per outer iteration on GPU, 54 iterations in 300 s, 3.9 GB
peak. Compute and kernel-launch bound, not memory bound.

**What to do:** two separable pieces. Restate the target with a stationarity
standard attached, since as written it never specified one and so is not
testable; and reduce per-iteration cost. The float32 row of that run ended
`line_search_failed` before the numerical-floor certificate existed and has not
been re-run, so its status is unknown.

### 5. Second-order acceleration

Twice demoted, and worth stating why so it is not re-prioritized by habit. It
will not rescue fits that stop at the numerical floor — there is nothing left to
step toward, confirmed by a continuation test that accepted zero further steps.
And the well-conditioned settings in the grid already certify in 2-7 s on GPU,
so the payoff is confined to weak-usage-penalty configurations needing ~1200 to
1740 iterations. The cheaper lever is passes per outer iteration: about 24, of
which roughly 12 are backtracking trials.

### 6. Lowering the numerical floor — diagnostics only

The composite slope is formed as the difference of two separately accumulated
aggregates that nearly cancel near the optimum (a ~2000:1 cancellation, which
*is* the KKT condition), and `sphere_l1_prox` quantizes its output at
renormalization. Accumulating the slope elementwise —
`(grad + lam * sign(F)) . dF` per entry, sign flips handled exactly — would put
the cancellation where it is small.

This would lower the floor and let more fits meet the strict KKT criterion. It
would not improve any fit: a float32 run stopping at a residual 10,000x larger
than its float64 counterpart matched it to four decimals in recovery and to
5.8e-7 relative in objective. Low priority now that such fits certify.

A curvature-aware criterion belongs with it. An absolute `stationarity_tol`
does not account for curvature, so two fits the same distance from their optima
(1.18e-9 and 1.39e-9) get opposite verdicts. Gradient over curvature has units
of parameter distance, so this is not the forbidden move of dividing
stationarity by the NLL.

## Deferred by the plan, unchanged

Stochastic minibatching at n around 10^6, which waits on per-iteration
efficiency; penalty selection, regularization paths and uncertainty estimates;
dense-support identification and alternate usage priors; warm starts across
strata; multi-GPU and batching of independent fits.

## Known stale artifacts

`benchmarks/runs/optimizer_repair_cpu_v1/` predates the repair and is retained
only as a record. The `scaling_target_v2` float32 row predates the
numerical-floor certificate. Historical `bench.py` outputs
(`results_*.json` at the top of `benchmarks/`) were produced by the superseded
solver; `RESULTS.md` keeps their tables with their invalidated conclusions
named, including one claim annotated as contradicting its own artifact.
