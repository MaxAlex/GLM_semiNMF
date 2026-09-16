# Optimizer correctness and development plan

Prepared 2026-09-16 against checkout `aa23358`, using
[`optimizer_work_handoff.md`](../optimizer_work_handoff.md), the implementation,
tests, [`METHOD.md`](METHOD.md), the
[implementation spec](../nb-glm-seminmf-implementation-spec.md), and saved
benchmark JSONs. This is an implementation plan; no solver changes or new
benchmark runs are included.

The immediate deliverable is a solver that optimizes a declared constrained
objective and reports stationarity of the parameters it returns. Performance
and optimizer comparisons follow that correctness gate. Keep the biological
model and regularization defaults unchanged; optional usage L2 should default
to zero. A stationary fit is not a guarantee of factor identification,
global optimality, or useful biological prediction.

## Findings and scope reconciliation

The handoff describes application revision `618e3ac` plus uncommitted usage-L2
changes. That revision is not in this checkout's local Git objects. This tree
has no `l2_G`, `F_fixed`, fixed per-feature dispersion input, or
`improved_on_init_`, and no `test_l2_usage.py`. Obtain and compare the
application changes before implementing compatibility work; do not assume
they are already upstream. The referenced pilot artifact directory is
accessible at `/data/agepath/pilot/loading_l1_usage_l2_2026-09-16/`, but the
parent-repository scripts are not part of this repository.

Code inspection confirms the normalization/penalty inconsistency, repeated
loading penalty inside the L-BFGS chunk loop, smooth-L1-plus-prox mixture,
stale Adam proximal learning rate, loss-only stopping, missing initial
checkpoint, and best-restoration iteration-count problem. `run_transform`
also tracks a best loss without saving/restoring its corresponding state.

Additional findings belong in the same correction:

- **Fixed scalar dispersion is not initialized correctly.** `_build_state`
  sets `log_theta=0`, while `_cfg` disables dispersion updates for a numeric
  value. Consequently the fixed-theta path appears to use theta=1 regardless
  of the requested scalar. Test this before using fixed-theta reproductions.
- **The likelihood's straight-through clamp is not an objective gradient.**
  Outside `[-30, 30]`, finite differences of the clamped forward objective
  disagree with its backward derivative. Existing finite-difference tests
  exercise ordinary predictors; extreme-predictor tests check only finiteness.
- **Bounds are applied outside acceptance.** The contribution clip
  `abs(F[g,k]) * max(G[:,k]) <= 15` couples F and G. A later G step can change
  feasibility. It must become an explicit constraint or a reported safeguard
  that prevents an unconstrained stationarity claim when active.
- **An active block can be skipped.** Alternating Adam/L-BFGS skip the entire
  G/b block when `k=0`, even if b is trainable. This affects the null model
  used for deviance as well as explicit zero-factor fits.
- **Export changes the fitted mean.** Deviance and loss are computed before
  usage snapping, while public `G_` contains snapped values. Certification,
  predictions, and reported diagnostics must identify which state they use.

The old specification itself prescribes post-step rescaling and loss-only
convergence. Its affected sections need revision along with the code; simply
implementing them more faithfully would retain the reported defects.

## 1. Establish a reproducible baseline and executable contract

Work primarily in `_fitting.py`, `_likelihood.py`, `model.py`, and focused
optimizer tests. First reconcile application API additions and record code
hashes, including any local patch. Preserve existing pilot results and use a
new run tag for subsequent experiments.

Define the fixed-dispersion objective once:

```
J_theta = NB_NLL(a + b + gamma @ Z.T + F @ G.T; theta)
          + lambda_F * sum(abs(F))
          + lambda_G1 * sum(G)
          + 0.5 * lambda_G2 * sum(G**2)
G >= 0
||F[:, k]||_2 = 1 for each active, trainable loading column
```

Specify zero-column/degenerate-factor semantics explicitly. Preserve component
count and report degeneracy without silently dropping or restarting factors.
For supplied fixed loadings, preserve their values, scale, and order exactly;
apply unit-column constraints only to trainable loadings. Validate rather than
silently normalize a fixed input when a particular mode requires unit columns.
Validate nonnegative finite penalty weights and positive finite scalar/vector
theta inputs.

Separate initial canonicalization, baseline shifting, and optimization updates.
Record the actual starting state after allowed canonicalization and initial
theta setup. A baseline shift is allowed only when its compensating intercept
change is feasible; never shift a frozen transform intercept. Evaluate every
allowed operation against the declared objective and constraints.

Create small deterministic CPU/float64 failures for each confirmed issue before
rewriting the optimizer. Include scalar theta values other than 1, vector theta
once integrated, covariates, fixed loadings, trainable exposure, and `k=0`.
Restore a working development environment: the current `.venv` symlink is
broken and system Python lacks torch. No tests were executed for this review.

**Exit gate:** a shared objective evaluator with separate NLL and penalty
components, documented constraints/frozen blocks, and regressions that expose
the old behavior. All later optimizers and diagnostics must use this contract.

## 2. Implement one correct constrained reference path

Start with alternating, direct-G, constrained proximal updates in float64.
This is the recommended correctness baseline; choose the production default
only after benchmarking. Do not make repairs to several legacy algorithms the
prerequisite for obtaining one trustworthy path.

- **G:** optimize physical nonnegative usages with projected/proximal steps
  and backtracking. Include usage L1 and L2 exactly. An entry at zero must be
  able to activate when its physical gradient is negative. First validate
  G-only inference against an independently implemented bounded optimizer.
- **F:** derive and implement a constrained composite step on unit-column
  spheres, with exact loading L1 included in the subproblem. Prototype a
  scalar-metric proximal subproblem, including zero/tie cases, before adding
  an adaptive metric. Accept against the actual objective at feasible F/G.
  Post-hoc normalization of an unconstrained step with its norm pushed into
  G is not this update. A loading retraction with G fixed is acceptable only
  as part of a derived, objective-tested constrained step.
- **a, b, gamma:** update every active block and include it in acceptance and
  residual checks. Frozen tensors must remain unchanged. Handle intercept/
  exposure gauge freedom consistently; do not let it obstruct residual tests.
- **Acceptance:** use a documented sufficient-decrease/backtracking rule for
  the baseline. Restore the full affected state on rejection or nonfinite
  values; bound line-search work and report failure. A future nonmonotone
  method needs its own explicit acceptance and convergence safeguards.
- **Likelihood:** use the stable, differentiable NB expression as the training
  objective, without a straight-through clamp. Test numerical behavior at
  extreme predictors. Keep exponentiation needed for moments/prediction
  separately guarded and report any clipping. If a predictor bound is
  retained for training, implement its actual constrained objective instead.
- **Separation:** initially treat contribution-limit activation as a reported
  safeguard/budgeted failure, without silently clipping a successful iterate.
  Preserve and return the best valid checkpoint. If the hard contribution
  bound is to define a supported constrained optimum, implement its coupled
  F/G feasibility and KKT conditions before certifying bound-active fits.

Unify objective assembly so global loading penalties appear exactly once per
full pass, independently of chunk count. Use only one documented loading-L1
treatment per algorithm. An explicitly smoothed-L1 diagnostic is a different
objective, with its epsilon and residuals reported; it must not be scored as
an exact-L1 solution or followed by an unexplained extra L1 prox.

Legacy Adam/joint Adam and L-BFGS must either be adapted to the same contract
and pass the gates, or be clearly deprecated/restricted with a migration path.
For adaptive proximal updates, use the actual optimizer learning rate and the
same metric in the proposal and prox, including any denominator floor.
Expose algorithm-specific step controls; do not imply Adam plateau decay
controls the current `lr=1.0` L-BFGS line search. Do not retain an unsupported
algorithm behind an apparently equivalent public option.

**Exit gate:** fixed-theta fits have feasible, objective-consistent updates;
G-only objective and physical KKT residuals agree with the independent oracle.
Positive usage L2 gives strict convexity for G-only inference with F,
intercepts, exposure, covariates, and theta fixed under the exact NB objective.
This does not establish convexity or global recovery for joint F/G fitting.

## 3. Certify convergence and protect initialization/checkpoints

Implement an optimizer-independent physical-parameter audit alongside step 2,
so it can validate the new path throughout development.

- For G, report the projected/KKT gradient: ordinary gradient on positive
  entries and `min(gradient, 0)` at zero. Specify the numerical active-set
  tolerance. Never infer physical stationarity from `dJ/dG_raw` alone.
- For smooth unit-norm F, report the tangent gradient. For exact L1, use a
  constrained proximal residual or minimize the subgradient/KKT residual over
  loading subgradients and sphere multipliers. Include the admissible
  `[-1,1]` subgradient at zero loadings; a smooth tangent formula is insufficient.
- Audit a, trainable b, gamma, feasibility, and any actual bounds. Report both
  maximum absolute residuals and explicitly defined scaled residuals per
  block. Calibrate absolute/relative tolerances with float64 oracles and dtype
  checks, without dividing stationarity by the full NLL or its constants.
- Use loss stagnation as a trigger for an audit or step adjustment. Declare
  success only when the declared residual/feasibility tolerances pass at
  fixed theta. A tiny step, dead softplus derivative, or exhausted line search
  must not become convergence.

Capture a true initial objective/checkpoint before the first step, and include
it among best-state candidates. Check for an already-stationary start before
doing an update. Compare improvement using an absolute-plus-relative numerical
tolerance. A stationary initial state that needs zero iterations is a success.

Return structured fitting information shared by fit and transform. Proposed
fields include `initial_objective_`, `final_objective_`, `stop_reason_`,
`stationarity_`, `n_iter_`, `best_iteration_`, `timed_out_`, objective components,
and dispersion status. Keep transform diagnostics separately available.
Distinguish `stationary`, `max_iter`, `timeout`, `stalled_nonstationary`,
`line_search_failed`, `nonfinite`, and `safeguard_hit` outcomes. Count completed
outer iterations explicitly; initialization and restoration are history events,
not iterations. Update the existing test that equates trace length with
iteration count if history semantics change.

Restore and re-audit the selected checkpoint, including theta; never inherit
`converged_` from an iterate that was discarded. Apply the same initial/best
protection to transform. Keep the saved best checkpoint immutable through
failed proposals and dispersion changes.

Prefer exact constrained usages as the authoritative fitted parameters.
If compatibility retains export snapping, expose the unsnapped state and
separate internal/exported residuals, objective changes, and a maximum predictor
change (measured or bounded). Recompute diagnostics on the public parameters;
public `converged_` must not certify a materially different, nonstationary export.
Make any change to `G_raw_`/`transform_raw_` semantics explicit. Exact boundary
solutions have real ties: a pre-projection value or seeded jitter is neither
guaranteed tie-free nor automatically a meaningful rank statistic. Preserve
legacy outputs where feasible, but do not fabricate rank distinctions to
justify an optimizer choice.

**Exit gate:** tests cover stationary initialization, failed first update,
near-zero activation, genuine boundary optima, best restoration, snapped
exports, max-iteration/time exits, and bit-identical frozen parameters.

## 4. Make dispersion phases and final-state selection explicit

Keep the current bounded MoM/trend policy initially; changing the statistical
estimator is a separate project. Fix scalar/vector initialization first.
Describe theta as fixed by the caller, frozen after an estimating-rule update,
or actually optimized. A refresh cap is not evidence of dispersion stability.

Partition history into fixed-theta mean-optimization phases. Reset stopping
comparisons whenever theta changes and record refresh events, log-theta change,
and the reason for freezing. An estimating-rule refresh need not decrease NLL.
Full changing-theta NLL is comparable as a value, but its decrease is not the
same assertion as fixed-theta mean improvement or joint likelihood stationarity.

Recommended selection policy: preserve the initial and best complete snapshots
under the full objective, including their theta; select a complete candidate,
freeze its theta, and finish with a bounded mean-model polish and final audit.
Reserve time for that phase. If the budget prevents certification, return the
best valid state with a nonconverged reason. Record improvement of the returned
full state against the actual initial full state separately from improvement
of mean parameters when both are evaluated at the returned theta.

**Exit gate:** periodic refresh, freeze-on-stall, refresh-cap, early-checkpoint
restoration, and timeout during a phase transition cannot produce a false
convergence claim or an ambiguous improvement flag.

## 5. Validate the repair, then address performance

Run validation in increasing order of cost:

1. Float64 finite differences for the smooth likelihood and L1-only,
   L2-only, and combined objectives away from L1 kinks; directional/subgradient
   tests at zero loadings. Cover normalization chain rules if normalized
   coordinates are used, covariates, extreme eta, and all trainable blocks.
2. Chunk-invariant objective components and gradients, including the global F
   penalty. Test float32/float64 and dense/sparse paths with declared tolerances;
   do not require identical nonconvex trajectories across accumulation orders.
3. Independent G-only bounded solves and small known-signal joint problems.
   Compare objective and residuals, plus existing signed recovery, exposure,
   null, covariate, separation, and reproducibility behavior. Keep optimizer
   correctness distinct from statistical recovery.
4. CPU synthetic screens at 500 and 2,000 genes, small fixed sample counts and
   K=5, two reproducible starts, fixed theta, and the handoff penalty grid.
   Start with a reduced smoke subset, then expand. Set explicit thread limits
   and wall-clock caps; report unresolved exits without relabeling them.
5. GPU/CPU performance and precision checks, then the existing public-data
   and scaling benchmarks. Run the frozen private-data screen only as an
   application integration check once small synthetic gates pass.

Put wall-clock checks inside iterative/line-search work, and use an outer
benchmark process cap for operations that cannot be interrupted promptly.
Report initialization, optimization, certification, and finalization/null-fit
time separately, plus total public-call time. Avoid an unbounded null fit or
postprocessing phase after a nominal solver timeout.

Every benchmark should retain source/patch hash, seed, data/config identity,
dtype, device, thread count, penalties, theta policy, dimensions/density,
objective components, residuals, feasibility/boundary/clipping status, actual
iterations, stop reason, elapsed time, and CPU/GPU peak memory. Record fit and
transform separately. Stop suppressing all runtime warnings without retaining
their diagnostic content. Compare time to the same stationarity standard,
including the number of unsuccessful runs.

Only then profile and introduce diagonal scaling, projected Newton for G,
second-order F updates that respect the column constraint, or a corrected
joint solver. Unit-column constraints couple feature rows, so a collection of
independent rowwise IRLS fits does not by itself solve the F problem. Retain
sample chunking and bounded working memory in every production path.

## Outstanding objectives and their placement

| Existing objective | Relationship to the handoff | Placement |
|---|---|---|
| Working second-order solver; Adam vs L-BFGS/IRLS | Direct: current comparison includes inconsistent objectives and unreliable success labels | Correct reference first; projected Newton/constrained second-order acceleration in step 5 |
| Softplus vs projected G; raw usages for rank statistics | Direct: boundary stagnation and exported-state semantics | Steps 2–3; resolve compatibility before changing defaults |
| Joint vs alternating default | Direct: both currently normalize outside the step | Rebenchmark corrected variants at equal stationarity after step 4 |
| Learning-rate schedule | Direct: stale prox threshold and misleading L-BFGS controls | Correct effective steps in step 2; tuning only after correctness |
| Theta update schedule and feedback | Direct: convergence and checkpoint comparisons cross objectives | Step 4; keep bounded estimating-rule policy initially |
| SVD/NMF/random initialization comparison | Initial-state protection changes what improvement means | Rebenchmark in step 5 with the true initial objective |
| Exposure fit vs offset | Requires auditing b and a valid null fit; does not justify a new default | Steps 1–3 correctness, step 5 public-data comparison |
| GPU target, scaling/memory, fit/transform timing | Handoff requires bounded, auditable performance evidence | Step 5; preserve the p=3000, n=20000, k=10 under-60-second target as a goal |
| Public counts: NMF/GLM-PCA deviance and restart stability | Outstanding report work; old scores lack a stationarity audit | Step 5, matched data/noise/scoring definitions and explicit budget exits |
| True stochastic minibatching at n around 10^6 | Separate scalability extension | Defer until the full-batch solver and audits are established |
| Profile-likelihood/damped dispersion estimation | Separate statistical-estimation objective | Follow-up study with its own criterion and validation |
| Penalty selection/paths, uncertainty estimates | New statistical capabilities, not correctness repairs | Deferred roadmap; no new regularization defaults in this work |
| Dense-support identification, alternate usage priors/quantile anchoring | May change the model or predictor and cannot fix optimizer inconsistency | Deferred research; qualify existing identification claims |
| Warm starts across strata; multi-GPU/fit batching | Warm starts already supported via initialization; orchestration is separate | Retain compatibility; caller-owned warm starts and later throughput work |

## Benchmark and documentation corrections

`benchmarks/RESULTS.md` says GPU results are pending, but GPU and real-data JSONs
are present. The saved p=3000, n=20000, k=10 run reports **235.2 s**, so
`METHOD.md`'s claim that this target already runs well under a minute is not
supported by that artifact. The GPU comparison also does not establish joint
Adam as faster: it reports 59.6 s versus 52.2 s for alternating, with different
recovery and convergence behavior. These are historical observations, not
certified comparisons of corrected algorithms.

The real-data artifacts report 501 iterations for a nominal 500-iteration
budget, consistent with restoration being appended to the loss trace. Some
report perfect restart matching despite weak deviance; deterministic starts
and repeated stationary failures must be distinguished from evidence of
identification. The CPU exposure table's correlation claims also disagree
with its saved JSON. Audit report values against artifacts, record provenance,
and preserve historical files rather than rewriting them as corrected results.

Update README, METHOD, API docstrings, and affected spec sections together:
objective/constraints, exact versus smoothed L1, fixed-F semantics, dispersion
phases, returned-state certification, raw/sparse usages, step controls, and
timeouts. Explain that nonnegativity and boundary contact alone do not prove
unique identification of an arbitrary finite factorization. Replace old
conclusions such as “Adam wins” with results from certified comparisons.

Suggested review units are: (1) contract/API reconciliation and reproductions;
(2) objective-consistent constrained engine plus independent residual audit;
(3) initialization, restoration, public diagnostics, and dispersion phases;
(4) bounded acceptance suite and performance report; (5) validated acceleration
and any justified default changes. Each unit should have explicit passing
gates; none should claim completion based only on a flatter loss trace.
