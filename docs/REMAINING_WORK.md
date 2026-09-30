# Remaining work

Status as of 2026-09-22, after the single-cell adaptation in
[SINGLE_CELL_WORK.md](SINGLE_CELL_WORK.md) and the optimizer repair described in
[OPTIMIZER_PLAN.md](OPTIMIZER_PLAN.md) and
[OPTIMIZER_CHANGELOG.md](OPTIMIZER_CHANGELOG.md). Every item below has evidence
behind its priority; the measurements are in
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md).

Nothing here blocks using the library. The solver optimizes one declared
objective, certifies the parameters it returns, and reports honestly when it
cannot. 111 tests pass. Sparse single-cell fits remain statistically and
numerically demanding; the latest bounded screen is not a convergence claim.

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
- Active-block derivative computation: identical measured objectives with
  about 14% lower GPU block-update time on the declared microbenchmark.
- Opt-in `l1_G="information"`, `init="pearson"`, and
  `dispersion="trend_pooled"`, with low-count calibration and held-out-gene
  benchmarks. Statistical defaults remain unchanged.
- External exposure overrides and AnnData `exposure_key`; transform requires
  compatible offsets after externally normalized fits. Rare support is now
  distinguished from entirely collapsed usages.
- Independent common-null dispersion for benchmark scoring. GLM-PCA predictor
  reconstruction now uses fitted intercepts and the correct size-factor
  convention, verified against upstream glmpca-py 0.1.0 before postprocessing.

## Open

### 1. Complete certified single-cell comparisons

**Why it matters:** the new 42-run synthetic screen spans depth, measured batch
effects, rare activities, correlated loadings, seven configurations, and two
perturbed starts. All fits exhausted 45 seconds, so their scores describe
budgeted iterates. Their shared calibration nulls also reached their iteration
cap. The harness records both limitations, and freezes one common null across
all candidates within a case.

The seven-configuration real-data pilot has a certified common null and
certified transforms. The automatic-penalty fit certifies with 4/5 programs
collapsed; all six alternatives exhaust their CPU budgets with residuals
96–331. Their predictive improvements are preliminary, not numerical-floor
cases or converged comparisons.

**What to do:** identify the blocks limiting certification, then compare at
matched stationarity and independently validated regularization settings.
Separate genes unobserved in training from supported genes when interpreting
predictive performance. The new `work_` counters support pass-cost analysis.

**Baseline follow-up:** rerun the historical 20 Newsgroups NMF/GLM-PCA
comparison with the corrected scoring and predictor reconstruction before
quoting method rankings. The earlier numbers remain invalidated.
`_null_deviance()` intentionally holds the fitted model's theta fixed;
independent dispersion estimation now lives in `benchmarks/scoring.py`.

### 2. Trend quality at the extremes of the mean range

**Why it matters:** one problem of sixteen (p=600, seed 12) fails to certify
because the feature with the largest mean takes the trend entirely (shrink
weight 0) and the trend is poor there — theta 1989 against a true 8.9.

**What changed:** information weighting of the positive-excess-only trend was
tested and rejected: it did not resolve low-mean selection bias. The opt-in
pooled trend includes signed excesses, shrinks weak mean bins toward a shared
estimate, and uses flat endpoint extrapolation. It substantially improves
constant-dispersion low-count calibration, but this does not settle the old
endpoint case or residual-dispersion bias after fitting latent programs.

**What remains:** validate the old endpoint reproduction and improve fitted-mean
calibration. In the rare/correlated synthetic case, pooling can approach the
Poisson ceiling and worsen recovery. Do not promote this option to the default
on the strength of the simpler calibration experiment.

### 3. Selecting sparse-data usage penalties

**Why it matters:** `0.005*p` is calibrated on the generator's count scale. On
20 Newsgroups (2.45% nonzero, mean count 0.044) it drives every usage to zero:
10/10 factors degenerate, deviance explained 0.0005, and the fit certifies —
correctly, since at `G = 0` the penalty gradient exceeds the likelihood's pull.
The diagnostics catch it, but the default's premise that the penalty is "small
relative to the likelihood" does not hold at that count scale.

**Decision:** retain `"auto"` for compatibility and provide the opt-in
information-scaled rule. The real single-cell pilot also reproduces severe
collapse with `"auto"`; weaker penalties retain more factors but generally
need longer optimization. Their apparent predictive gains are provisional.
Select penalties on observations excluded from usage inference, with independent
donor validation before a default change. The current real-data panel was
preselected using discovery donors, so it is an engineering pilot only.

### 4. The p=3000, n=20000, k=10 target

**Why it matters:** missed by roughly an order of magnitude once stationarity
is required — 5.57 s per outer iteration on GPU, 54 iterations in 300 s, 3.9 GB
peak. Compute and kernel-launch bound, not memory bound.

**What to do:** two separable pieces. Restate the target with a stationarity
standard attached, since as written it never specified one and so is not
testable; and reduce per-iteration cost. The float32 row of that run ended
`line_search_failed` before the numerical-floor certificate existed and has not
been re-run, so its status is unknown.

The active-block optimization reduces one measured component of runtime; the
large target is still open. Step-size reuse showed no benefit in that timing
case and defaults to False. Dense zero contributions and float64 line-search
evaluation remain substantial work even for sparse inputs.

### 5. Second-order acceleration

Twice demoted, and worth stating why so it is not re-prioritized by habit. It
will not rescue fits that stop at the numerical floor — there is nothing left to
step toward, confirmed by a continuation test that accepted zero further steps.
And the well-conditioned settings in the grid already certify in 2-7 s on GPU,
so the payoff is confined to weak-usage-penalty configurations needing ~1200 to
1740 iterations. The cheaper lever is passes per outer iteration: about 24, of
which roughly 12 are backtracking trials.

The single-cell screen strengthens the case for profiling weak-penalty fits:
many remain well above tolerance after roughly 1,600–2,100 updates. Numerical-
floor arguments do not explain those residuals. Diagnose active blocks before
choosing between better curvature, fewer passes, or stronger regularization;
stronger regularization is not a free computational improvement.

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
