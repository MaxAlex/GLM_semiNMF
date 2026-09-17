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

## Open, with no results yet

- Trend quality at the extremes of the mean range, the one dispersion
  mechanism the estimator fix did not address (p=600 seed 12 above).
- Public-data and exposure comparisons, which have no known theta. The
  estimated-dispersion path now certifies 12/16 synthetic problems rather than
  10/16, so these are no longer blocked, but they should report stop reasons
  and residuals per fit rather than assuming certification.
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
