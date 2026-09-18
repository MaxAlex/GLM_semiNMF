# Benchmark results

Two eras of results live in this file. **Current results** come from the
objective-consistent solver and report physical stationarity of the parameters
actually returned. **Historical results** at the bottom were produced by the
superseded Adam/rescaling solver; they are kept for provenance and are not
corrected in place, because the solver they measured no longer exists.

A run that exhausted its budget is reported as unresolved. It is never counted
as a success, and never as evidence that a configuration is bad.

## Current: bounded synthetic optimizer screen

Produced by `bench_optimizer.py`. Raw artifacts, with source hashes, git
revision and full per-run diagnostics, are in
`runs/optimizer_repair_cpu_v2/` and `runs/optimizer_repair_gpu_v2/`.

```
bench_optimizer.py --out runs/optimizer_repair_<dev>_v2 --full-grid \
    --seeds 0 1 --seconds 300 --max-iter 2000 --threads 4 --device <dev>
```

Data: `simulate_nb_seminmf(p, n=200 train + 100 held out, k=5,
baseline_log_mean=-1.5, dispersion=5)`, float64, dispersion fixed at the
generating theta, exposure fixed at the generating `b`, two SVD-based starts
perturbed by N(0, 0.001). Penalty grid from the handoff: `l1_F = cF * n` for
`cF` in {1e-4, 1e-3, 1e-2} crossed with usage L2 in {0.01, 0.1, 1}, plus two
controls at `cF=1e-3` — usage L1 = 10, and no usage penalty at all.
Certification is a maximum absolute physical KKT residual of 1e-3 in every
active block. Machine: 12-core CPU, RTX 3060 12GB, torch 2.13, CUDA 13.0.

| | CPU | GPU |
|---|---|---|
| fits certified | 33/44 | 38/44 |
| transforms certified | 44/44 | 44/44 |
| certified fit wall: min / median / max | 3.8 / 34.4 / 151.6 s | 1.8 / 7.3 / 67.3 s |
| unresolved | 8 timeout, 3 max_iter | 6 max_iter |
| peak host RSS | 648 MB | 1098 MB |
| peak GPU memory | n/a | 55 MB |

### The usage penalty, not the loading penalty, sets the conditioning

Median iterations and wall time over certified runs, both devices:

| usage penalty | p | iters | CPU s | GPU s |
|---|---|---|---|---|
| L1 = 10 | 500 | 58 | 3.9 | 1.8 |
| L2 = 1 | 500 | 87 | 5.7 | 2.5 |
| L2 = 0.1 | 500 | 258 | 17.0 | 7.3 |
| L2 = 0.01 | 500 | 1742 | 128.6 | 49.7 |
| none | 500 | — | never certified | never certified |
| L1 = 10 | 2000 | 73 | 22.2 | 4.1 |
| L2 = 1 | 2000 | 132 | 36.4 | 7.0 |
| L2 = 0.1 | 2000 | 195 | 55.2 | 10.6 |
| L2 = 0.01 | 2000 | ~1190 | timeout at 300 s | 61.5 |
| none | 2000 | — | never certified | never certified |

Iteration count spans a factor of 30 across the usage penalty and is almost
flat in `l1_F`: at p=500 with usage L2 = 0.01, `cF` of 1e-4 / 1e-3 / 1e-2 took
2000 / 1843 / 1641 iterations. Every unresolved run in the grid is a weak or
absent usage penalty.

The no-usage-penalty control never certifies on either device (residuals 13 to
52) and is the only configuration whose contribution crosses the separation
trigger (18.6 at p=500; 21.4-30.9 at p=2000, depending on how far the
unresolved run got, versus 7.2-12.0 everywhere else). This
is the identifiability argument in the README showing up as conditioning: the
predictor is invariant under `G_k -> G_k + c` with `a -> a - c F_k`, so with
nothing anchoring the usage baseline there is a flat direction, usages stay
dense (1% exact zeros, against 18-93% elsewhere), and no stationary point is
reached. It is also the only configuration whose fitted predictor left
`+/-30`: on GPU, where both p=2000 runs got the full 2000 iterations, 3 entries
fell outside it in each. That is what the predictor-range warning reports, and
it is the reason `deviance_explained_` is approximate for those two runs.

Fitting `l1_G="auto"` (`0.005*p`, i.e. 10 at p=2000) is the fastest certified
configuration in the grid. That is a statement about conditioning, not a
recommendation to raise the penalty: stronger usage penalties also drive far
more usages to exactly zero (93% at L1=10 versus 18% at L2=0.01), which is a
modelling choice, not a free speedup.

### Transform

Transform certified on every run of both grids, in 2-8 iterations and
0.04-0.27 s for 100 held-out samples. With loadings, intercepts, covariates and
dispersion fixed and a positive usage penalty the G-only problem is strictly
convex, and it behaves that way. Transform cost is negligible next to fitting
and is reported separately in every artifact.

### CPU versus GPU

Restricted to the 32 configurations certified on both devices:

- Wall-time ratio CPU/GPU: min 2.02x, median 2.49x, max 6.02x. The speedup
  grows with problem size — around 5.2x at p=2000 against 2.2-2.6x at p=500 —
  so at these sizes the GPU is still substantially launch-latency bound.
- 31 of 32 took the **identical** number of outer iterations on both devices,
  and final objectives agree to a maximum relative difference of 6.1e-16.
  Device choice changes the timing, not the trajectory.

### The p=3000, n=20000, k=10 target is not met

Artifacts in `runs/scaling_target_v2/`. Bounded at 300 s and 2000 iterations,
with dispersion and exposure fixed and `l1_G="auto"` (the grid's
fastest-converging setting):

| device | dtype | iters reached | s/iteration | residual | outcome |
|---|---|---|---|---|---|
| GPU | float32 | 54 | 5.57 | 1.6e3 | `line_search_failed` |
| GPU | float64 | 54 | 5.58 | 2.6e3 | timeout |
| CPU | float32 | 4 | 76.9 | 1.6e6 | timeout |

All three are budget exits; none is a fit. At 5.57 s per outer iteration and
the 73 iterations this penalty setting needed at p=2000, certifying this size
on GPU would take roughly 7 minutes — about an order of magnitude past the
spec's 60-second goal, measured against a certification standard the original
goal never had. Peak GPU memory was 3.9 GB, comfortably inside 12 GB, so this
is compute and kernel-launch bound rather than memory bound.

The float32 GPU row is also informative on its own: it ends
`line_search_failed`, meaning backtracking could no longer resolve a
sufficient decrease at float32 precision. Tolerances are not relaxed to hide
that, which is why float64 is the default.

### Dispersion estimation dominates time-to-certification

The grid above holds dispersion fixed, which isolates the mean-model solver.
Measured separately on `simulate_nb_seminmf(p=300, n=500, k=3)` at
`max_iter=1000`, comparing `dispersion="trend"` against the generating theta:

| seed | trend | fixed |
|---|---|---|
| 6 | `max_iter`, residual 115 | `stationary`, 334 iters, 51 s |
| 7 | `stationary`, 538 iters, 89 s | `stationary`, 202 iters, 29 s |
| 9 | `stationary`, 429 iters, 73 s | `stationary`, 280 iters, 34 s |
| 11 | `max_iter`, residual 104 | `stationary`, 675 iters, 100 s |

Fixed dispersion certified 4 of 4; estimated dispersion failed 2 of 4 and
needed 1.5-2.6x more iterations where it did succeed. Factor recovery was
comparable either way (0.91-0.96), so this is a cost in certification, not in
fit quality.

Traced on seed 6, the cause is the estimated theta values themselves, not the
phase machinery. The phase behaves as designed: it froze cleanly at iteration
77 with `best_iteration` also 77, so nothing was rewound, and the remaining 923
iterations were a fixed-theta polish. It froze on `mean_stall`, never on theta
stability -- `delta_log_theta` across the eight refreshes ran
8.01, 5.44, 1.63, 0.81, 1.70, 1.15, 2.30, 0.81, oscillating rather than
approaching the 0.05 freeze tolerance.

The frozen estimate is median-unbiased (median log ratio to the generating
theta -0.02, 74% of features within 2x) but has a heavy upper tail: its maximum
is 9707 against a generating maximum of 47.8. Refitting from a clean start at
that frozen theta reaches only residual 190.9 in 1000 iterations, where the
generating theta certifies in 334. Since curvature carries theta directly
(`h = (theta + X) * sigmoid(d) * sigmoid(-d)`), a few grossly overestimated
features are enough to condition the mean-model problem badly.

The cause is specific: 17 of 300 features had `ssr - s_mu <= 0`, and those
same 17 were the ones estimated above theta = 1000 against a true theta near
13. Their moment estimate is outside the parameter space, so it clamps to the
alpha floor; the count-based shrink weight does not rescue them because they
have plenty of counts (median `s_mu` 290, weight 0.85). Underdispersion by
chance, not near-Poisson behavior.

### Fix: shrink on resolvable excess, not on counts

`_dispersion.py` now sets the shrink weight from how large a feature's variance
excess is relative to its own sampling scale, `z = (ssr - s_mu) / sqrt(2*s_mu2)`
clamped at zero, combined multiplicatively with the existing count weight. A
feature with no resolvable excess takes the trend rather than the floor, and
features outside the parameter space no longer enter the trend fit. `"feature"`
mode stays raw by contract.

Measured end to end over 16 problems (p=300/n=500 and p=600/n=300, 8 seeds
each, `dispersion="trend"`, `max_iter=1000`), old estimator against new:

| | old | new |
|---|---|---|
| certified | 10/16 | **12/16** |
| median RMSE of log theta vs truth | 1.564 | **0.643** |
| largest theta over all runs | 9775 | 1989 |
| median factor recovery | 0.937 | 0.938 |
| total wall time | 1262 s | 976 s |

Theta accuracy improved on 16 of 16 problems, and recovery moved between
-0.001 and +0.026, so this buys certification and accuracy without trading
away fit quality. It also restored `test_exposure_invariance` to the default
estimated-dispersion path, which had needed a fixed-theta workaround.

Four problems still do not certify, all at p=600. Three are near misses that
end `line_search_failed` or `max_iter` at residuals of 1.1e-3 to 1.3e-2
against a 1e-3 tolerance -- step resolution, not dispersion. The fourth
(seed 12, residual 7.26) still carries one theta of 1989 against a true 8.9:
that feature has the largest mean in the dataset, takes the trend entirely
(weight 0), and the trend is simply poor at the top of the mean range. A trend
value bounded to the fitted response range was tried and reverted: it changed
no aggregate and flipped one problem each way at the tolerance boundary, so
the mechanism is trend quality at the extremes of the mean range, which is a
separate piece of work.

## Current: initialization, exposure, and public counts

Produced by `bench.py` at `--quick` sizes on GPU, float64, with an explicit
budget (`--max-iter`, `--max-seconds`) recorded in each artifact. Raw outputs
in `runs/public_v2/`. These replace the historical `bench.py` sections; the
optimizer-comparison sweep they used to contain is gone, because `algorithm`
and `g_parametrization` are now aliases of one solver.

### Initialization (p=600, n=2000, k=6, budget 8000 iterations / 900 s)

| init | wall | iters | stop | residual | initial objective | final objective | recovery |
|---|---|---|---|---|---|---|---|
| svd | 88.7 s | 709 | `line_search_failed` | 7.2e-3 | 2,692,209 | 2,428,488 | **0.939** |
| nmf | 57.3 s | 467 | `line_search_failed` | 2.7e-3 | 3,546,182 | **2,417,762** | 0.921 |
| random | 25.1 s | 221 | `line_search_failed` | 2.7e-3 | 2,801,482 | 2,577,771 | 0.152 |

None certified, and the budget is not what stopped them: raising it 16x from
the 500-iteration default moved only svd (500 to 709 iterations, residual
0.204 to 0.0072) before its line search gave out too. All three converge to a
residual floor of 2.7e-3 to 7.2e-3 at this problem size, against a 1e-3
tolerance, which is the step-resolution limit rather than three different
optima.

Read at roughly matched residuals, the historical "svd wins on every axis"
does not survive: **nmf reaches the lowest objective** while svd has the best
factor recovery, and nmf gets there from the worst starting point of the three
(initial objective 3.55e6 against svd's 2.69e6) -- a comparison the old code
could not make, because it recorded the first iterate rather than the true
initialization. Random init remains far worse on recovery.

### Exposure, offset versus fitted b (p=600, n=2000, k=6, exposure_sd=0.8)

| mode | wall | stop | residual | recovery | deviance explained |
|---|---|---|---|---|---|
| `"offset"` | 163.5 s | `line_search_failed` | 1.2e-2 | **0.939** | **0.572** |
| `"fit"` | 62.1 s | `line_search_failed` | 1.3e-3 | 0.892 | 0.387 |

Cross-agreement between the two fits' loadings 0.921; agreement between their
exposures 0.799. Offset stays the right default, now on stronger grounds than
the historical run gave: it wins on recovery **and** on deviance explained,
consistent with fitted `b` absorbing structure that belongs in `G`. The
`b` agreement of 0.799 also confirms the audited artifact values (0.76 CPU,
0.91 GPU) rather than the "r > 0.99" the historical table claimed.

The wall times are not comparable: the two runs stopped at residuals 8.6x
apart, so the historical "fitting b doubles wall time" is not reproduced and
not contradicted either -- it is unmeasured at equal stationarity.

### 20 Newsgroups (p=1500, n=2328, 2.45% nonzero, mean count 0.044, k=10)

**The default usage penalty collapses this fit.** `l1_G="auto"` is `0.005*p`,
here 7.5, and at that strength every usage goes to zero: all 10 factors
degenerate, `G == 0` on 100% of entries, deviance explained 0.0005. The fit
certifies `stationary` in 15 iterations, correctly -- at `G = 0` the usage
penalty's gradient exceeds the likelihood's pull on counts this sparse, so
zero satisfies the KKT conditions for every entry. The diagnostics catch it
(10/10 degenerate flags and a warning), and the `0.005*p` heuristic is simply
calibrated on the synthetic generator's count scale, which is roughly 20x
denser than this.

The same conditioning window as the synthetic grid appears on real data:

| `l1_G` | stop | iters | residual | deviance explained | `G == 0` | degenerate |
|---|---|---|---|---|---|---|
| 7.5 (`"auto"`) | stationary | 15 | 5.3e-4 | 0.0005 | 1.000 | 10/10 |
| 1.0 | stationary | 1223 | 9.5e-4 | **0.1492** | 0.969 | 0 |
| 0.25 | timeout | 1099 | 190 | unresolved | 0.831 | 0 |
| 0.05 | timeout | 898 | 1.1e3 | unresolved | 0.381 | 0 |

Too strong certifies an empty model; too weak cannot certify at all. No
default is changed on this evidence -- it is one dataset -- but the heuristic's
documented premise that the penalty is "small relative to the likelihood" does
not hold at this count scale.

At `l1_G = 1.0`, where the model is alive and certified:

| method | budget | wall | deviance explained |
|---|---|---|---|
| ours | 1223 iters, `stationary`, residual 9.4e-4 | 336 s | **0.1492** |
| NMF (sklearn, `nndsvda`) | 400 iters | 0.2 s | -0.0227 |
| GLM-PCA (`fam="nb"`) | 300 iters | 441 s | -0.4709 |

Both baselines score below the intercepts-plus-exposure null. **Treat that
comparison as provisional**, for a reason visible in the artifacts: the
baselines' deviance is computed with *our* fitted `theta`, so the identical
NMF fit scored +0.0611 against the degenerate model's theta and -0.0227
against this one. A sound baseline comparison needs a common, model-independent
dispersion, which is a scoring definition this harness does not yet have.
GLM-PCA's predictor is also reconstructed from its returned factors and
loadings by an assumed convention (`_glmpca_eta`), which is unverified.

**Restart stability, measured properly, is 0.91 and not 1.0.** Varying
`random_state` alone reports a perfect 1.0, but that is determinism, not
identification: with `init="svd"` the seed only sets the starting vector for
`svds`, which converges to the same subspace, so the starts differ by ~1e-14
relative and match at |corr| = 1.000000. Perturbing the warm start instead --
enough to move it to 0.9998 agreement with the unperturbed init, a 0.02%
nudge -- gives pairwise loading agreement of 0.866, 1.000, 0.867 (mean 0.911)
with all three replicates certified. So this dataset has at least two distinct
certified stationary points whose loadings differ materially, which a
deterministic-restart test cannot see.

## What `line_search_failed` actually means

Instrumenting the final failing line search (p=600, n=2000, k=6, nmf init,
group `['F', 'a']`) shows it is not a bad direction, a dispersion problem, or
an exhausted budget. Across the 30 halvings the displacement shrinks
monotonically from 3.2e-8 to 1.7e-16, a factor of 5e7, while the slope stays
pinned at ~1e-11 and changes sign at the third trial. A real directional
derivative would shrink in proportion to the step. It does not, because at
that point the step is being formed out of last-bit changes to parameters of
order one.

The floor is in the F-block step, not in the objective evaluator. Measuring
`objective_difference` directly -- displacing the smooth `a` block and
comparing against `g.d + 0.5 d'Hd` -- it resolves changes of 8.9e-21 to
1.2e-16 across four problems, which is 1e-10 to 1e-5 of one ulp of the
objective. The remaining gap here is 4.9e-12, comfortably representable. An
earlier version of this section attributed the failure to objective resolution;
that was wrong.

A per-block breakdown of the failing trials shows what is actually happening:

| trial | step | slope_F | slope_F_l1 | \|dF\| | \|da\| | cos(dF) | cos(da) |
|---|---|---|---|---|---|---|---|
| 0 | 1.0 | +3.4928e-8 | -3.4946e-8 | 5.7e-9 | 3.7e-8 | - | - |
| 6 | 1.6e-2 | +5.8065e-10 | -5.5027e-10 | 9.0e-11 | 5.7e-10 | 1.0000 | 1.0000 |
| 18 | 3.8e-6 | +4.8093e-11 | -2.8422e-13 | 2.2e-14 | 1.4e-13 | 0.9985 | 0.9999 |
| 24 | 6.0e-8 | +3.2230e-11 | -8.5265e-14 | 9.6e-16 | 2.3e-15 | 0.8235 | 0.9651 |
| 29 | 1.9e-9 | +1.6156e-11 | -8.5265e-14 | 9.9e-16 | 0 | 0.2665 | - |

Two things combine. First, **the composite slope is the residue of a ~2000:1
cancellation**: at trial 0 the smooth term is +3.4928e-8 and the loading-L1
term is -3.4946e-8, leaving a net of -1.8e-11. That cancellation is intrinsic
near the optimum -- it *is* the KKT condition, the smooth gradient balancing
the L1 subgradient -- but the code forms it as the difference of two separately
accumulated aggregates, so the digits lost to cancellation are lost for good.

Second, **the F displacement floors at ~1e-15 and randomizes**. `sphere_l1_prox`
renormalizes each column to unit norm, so its output is quantized at about eps
relative to entries of order 1/sqrt(p); below that the displacement stops
tracking the step (9.6e-16 at trial 24, still 9.9e-16 at trial 29) and its
direction decorrelates (cosine 1.0000 -> 0.2665). The `a` block has neither
problem: `|da|` halves cleanly to zero and `slope_a` with it.

So the noise floor of the slope, ~1e-11, is the same size as the true remaining
descent, and the Armijo test cannot tell them apart. `line_search_failed` is
the honest report of that. The fit is at the optimum its step construction can
reach, which is not the same as the optimum its objective evaluator could
resolve.

That distinction matters for any fix: calibrating a tolerance against the
objective increment's precision would be calibrating against the wrong
quantity, since that precision is five to nine orders finer than the binding
one. A more promising, and untested, direction is to accumulate the composite
slope elementwise -- `(grad + lam * sign(F)) . dF` per entry, with sign flips
handled exactly -- so the cancellation happens per entry, where it is small,
instead of between two large sums.

The reason a 7e-3 gradient corresponds to so tiny a gap is curvature. Loadings
are unit-norm, so all scale lives in the usages and the F-block Hessian
diagonal accumulates over samples; it reaches 5e6 here. Dividing the residual
by it gives the distance to the optimum in parameter space:

| p | n | stop | residual | F curvature | residual / curvature |
|---|---|---|---|---|---|
| 300 | 500 | `stationary` | 9.8e-4 | 8.4e5 | 1.18e-9 |
| 600 | 2000 | `line_search_failed` | 7.0e-3 | 5.05e6 | 1.39e-9 |
| 600 | 2000 | timeout | 0.276 | 8.8e6 | 3.1e-8 |
| 600 | 300 | max_iter | 6.32 | 4.9e7 | 1.3e-7 |

The first two rows sit essentially the same distance from their optima, 1.2e-9
and 1.4e-9. One is labelled `stationary` and the other is not, purely because
`stationarity_tol` is an absolute gradient threshold and the gradient-to-
distance conversion factor differs between the problems. Curvature spans 8.4e5
to 4.9e7 across these four fits, so a fixed 1e-3 silently demands a 58x
stricter parameter-space accuracy on one problem than another.

Two consequences for the remaining work. Second-order steps will not rescue
these cases as they stand -- the step construction, not the step direction, is
what runs out. And a certification criterion that accounts for curvature (the
implied objective gap, or the Newton-step norm, which is a parameter distance)
would be dimensionally correct where an absolute gradient is not. That is not
the forbidden move of dividing stationarity by the NLL or its constants:
gradient over curvature has units of parameter distance.

### It costs nothing in fit quality

Before treating any of that as solver work, the question is whether pushing
through the floor would produce a better fit. Measured on p=600, n=2000, k=6:

| | stop | iters | residual | objective | recovery | dev. explained |
|---|---|---|---|---|---|---|
| float64, to failure | `line_search_failed` | 663 | 1.48e-2 | 2428487.6 | 0.9395 | 0.6027 |
| continuation, fresh curvature | `line_search_failed` | **0** | 1.48e-2 | 2428487.6 | 0.9395 | - |
| float32, to failure | `line_search_failed` | 93 | **153.8** | 2428489.0 | 0.9399 | 0.6027 |

Rebuilding the state from the returned parameters and running again with
freshly computed curvature and a 4000-iteration budget accepts **zero** steps
and moves the objective by exactly 0. The floor is hard, and nothing is left
on the table.

The float32 row is the more informative one. Its line search quits after 93
iterations instead of 663, at a residual 10,000x larger -- and the fit is
indistinguishable: the objective is 1.4 higher out of 2.43e6 (5.8e-7
relative), recovery is 0.9399 against 0.9395, and deviance explained agrees to
four decimals. Stopping early cost nothing and saved 7x the iterations.

So the residual at which the line search gives out is a poor proxy for how good
the fit is, and `line_search_failed` near an optimum is closer to a feature
than a defect. **What is wrong here is the label, not the answer**: `converged_`
reads False on fits that are numerically final. Fixing the slope accumulation
and the criterion would correct the reporting and make benchmark tables
readable; it would not produce better fits, and it is not a reason to pursue
second-order steps. Scope it as a diagnostics fix.

Evidence is one synthetic problem with a matched float64/float32 pair plus the
continuation test. The 20 Newsgroups fit certified outright, so it never
reached this regime, and the p=3000 float32 row in the scaling section
(`line_search_failed` at residual 1561) has no converged counterpart to
compare against, so it remains unverified.

## Open, with no results yet

- Trend quality at the extremes of the mean range, the one dispersion
  mechanism the estimator fix did not address (p=600 seed 12 above).
- A model-independent dispersion for scoring baselines, without which the
  NMF and GLM-PCA comparison above cannot be read as a baseline comparison.
  Verifying the GLM-PCA predictor reconstruction belongs with it.
- Diagnostics only, not fit quality (see "It costs nothing in fit quality"):
  cancellation-free accumulation of the composite slope, then a curvature-aware
  certification criterion calibrated against the floor that remains, so
  `converged_` stops reading False on numerically final fits.
- Second-order acceleration (projected Newton for G, sphere-respecting
  second-order F). Each outer iteration currently costs about 24 data passes:
  ~11 gradient evaluations, ~12 objective-difference evaluations for
  backtracking, and one full objective. Reducing passes and reducing iteration
  count are separate levers; the well-conditioned settings in the grid already
  certify in 2-7 s on GPU, so the payoff is concentrated in the weak-usage-
  penalty settings that need ~1200-1740 iterations.
- Public-data comparison (20 Newsgroups deviance against NMF and GLM-PCA,
  restart stability) at matched data, noise and scoring definitions. The
  historical numbers below predate certification and are not comparable.
- Real-data application screens, which belong downstream of these gates.

---

## Historical (superseded solver, retained for provenance)

Everything below was produced by the Adam/rescaling solver that this work
replaced, on `bench.py` at `--quick` sizes (p=600, n=2000, k=6). **Its
`converged` column reflects loss stagnation, not stationarity**, and its
optimizer comparison is no longer reproducible: `algorithm` and
`g_parametrization` are now deprecated aliases of a single solver, and
`bench.py` no longer sweeps them. Several conclusions it drew have since been
contradicted or invalidated:

- "**Adam wins**" compared three implementations of an objective none of them
  optimized consistently, and scored them by a loss whose penalties moved
  between steps.
- Softplus was kept as the default because `G_raw_` had to be "tie-free for
  downstream rank statistics". Those distinctions were manufactured; usages
  are now optimized directly and their boundary ties are reported as real.
- The GPU sections were recorded as pending after the container lost GPU
  access. GPU access was restored on 2026-09-16 and the current sections above
  supersede them.
- "svd wins on every axis (speed, loss, recovery)" is not reproduced: read at
  matched residuals, nmf reaches the lower objective (see the current
  initialization section). Its restart-stability figures also varied only
  `random_state`, which with `init="svd"` does not produce distinct starts.
- A saved artifact reported 235.2 s for p=3000, n=20000, k=10, which does not
  support the claim elsewhere that the target already ran well under a minute;
  the measurement above shows the target is missed by a wider margin once
  stationarity is required.
- Real-data artifacts reporting 501 iterations against a 500-iteration budget
  reflect restoration being appended to the loss trace, which is now a
  `history_` event rather than an iteration.

<details>
<summary>Historical tables as originally recorded</summary>

Machine: 12-core CPU, RTX 3060 12GB (CUDA), torch 2.13.

### 1. Optimizer per block: Adam vs L-BFGS vs IRLS

| config (init=svd, softplus G) | wall s (CPU) | outer iters | final loss | recovery* |
|---|---|---|---|---|
| Adam, alternating (default)   | 38.1 | 151 | 2,433,181 | 0.930 |
| Adam, joint                   | 26.3 |  99 | 2,436,064 | 0.941 |
| L-BFGS per block              |  4.8 |  11 | 2,758,600 | 0.000 |

*recovery = mean matched |loading correlation| against the generating factors
(Hungarian assignment).

IRLS was not implemented: the `G >= 0` constraint breaks its per-block
least-squares structure, as the spec anticipated.

### 2. Softplus reparametrization vs projected gradient

| G parametrization | wall s | final loss | recovery | exact-zero frac in `G_` |
|---|---|---|---|---|
| softplus (default) | 38.1 | 2,433,181 | 0.930 | 0.69 |
| projected clamp    | 30.0 | 2,429,121 | 0.955 | 0.65 |

### 3. Joint vs alternating optimization

Joint Adam matched alternating on recovery (0.941 vs 0.930) and was ~1.4x
faster on CPU at these sizes.

### 4. Theta re-estimation schedule

Resolved during development rather than benchmarked: re-estimating every
iteration was visibly unstable (loss spikes at each refresh); freezing after a
stall removed the spikes without changing the fitted factors materially.

### 5. Exposure: fixed offset vs fitted b

| mode | wall s | recovery | notes |
|---|---|---|---|
| `"offset"` (default) | 41.0 | 0.943 | |
| `"fit"`              | 80.9 | 0.939 | b agrees with offset b at r > 0.99 |

> **Audited against the artifact and found wrong.** `results_exposure_cpu.json`
> records `b_agreement: 0.76` and `cross_agreement: 0.967` for this `--quick`
> CPU run, not the "r > 0.99" claimed above; `results_exposure_gpu.json` gives
> 0.91 and 0.994. The recovery figures do match. The claim is left in place as
> originally written, with this correction alongside it.

### Initialization comparison

| init | wall s | outer iters | final loss | recovery |
|---|---|---|---|---|
| svd (default) | 38.1 | 151 | 2,433,181 | **0.930** |
| nmf           | 81.6 | 334 | 2,463,888 | 0.887 |
| random        | 53.0 | 220 | 2,517,002 | 0.411 |

### Findings that changed the implementation

- **Usage-baseline flatness.** The predictor is invariant under
  `G_k -> G_k + c, a -> a - c F_k`. Unanchored, fitted G drifts dense and
  recovery drops. A small L1 on G (`l1_G = 0.005 p`) anchors the touch-zero
  representative. *(Independently confirmed by the current grid, where the
  no-usage-penalty control is the only configuration that never certifies.)*
- **Prox threshold flooring.** Adam-scaled proximal thresholds exploded when a
  weak factor's gradients vanished. *(No longer applicable: there is no Adam
  proximal threshold.)*
- **Dispersion trend fitting** must avoid `torch.linalg.lstsq` (first call in
  a process differs bitwise from later calls, breaking seed reproducibility);
  normal equations are used instead. *(Still applicable.)*

</details>
