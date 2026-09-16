# GLM-semiNMF optimizer: developer handoff

Date: 2026-09-16. Scope: optimizer correctness and diagnostics, not a request
to change the biological model or adopt new regularization defaults.

Inspected submodule base commit: `618e3ac2b44acc7028737f7d7b67ea7aeed76e73`,
**plus uncommitted local changes** adding optional squared usage L2. These
changes are in `src/glm_seminmf/{model.py,_fitting.py}` and
`tests/test_l2_usage.py`. Do not assume the base commit alone contains L2.

## Summary / requested outcome

Please make fitting optimize one explicitly defined constrained objective,
and certify convergence on the parameters actually returned. Current loading
normalization preserves the likelihood but changes the penalties outside the
optimization step. Loss stagnation can consequently be reported as convergence
without physical-parameter stationarity. There are also initial-objective and
L1 implementation consistency issues detailed below.

These issues affect confidence in fitted programs. They do **not** explain all
poor biological prediction: an independent constrained-objective diagnostic
can reach stationary fits that still predict poorly on real data. Nor do our
CPU timeouts establish a fundamental limitation of GLM-semiNMF.

## 1. Priority: loading normalization must be part of the optimization problem

Intended mean model, omitting optional covariates for clarity:

```
eta[g,c] = a[g] + b[c] + (F @ G.T)[g,c]
J = NB_NLL + lambda_F * sum(abs(F))
           + lambda_G1 * sum(G) + 0.5 * lambda_G2 * sum(G**2)
G >= 0; each nondegenerate F column has L2 norm 1.
```

Relevant locations: `_fitting.py:rescale_columns`, `_lbfgs_block`,
`_prox_and_clip_F`, `run_fit`.

The current sequence updates unrestricted F, then normalizes each column
and pushes its norm into G. For one column, writing `c = ||F_k||_2`:

```
F_k <- F_k / c
G_k <- c * G_k
```

The predictor is unchanged, but that column's penalty changes from

```
lambda_F * ||F_k||_1 + lambda_G1 * sum(G_k)
                         + 0.5 * lambda_G2 * ||G_k||_2**2
```

to

```
lambda_F * ||F_k||_1 / c + lambda_G1 * c * sum(G_k)
                         + 0.5 * lambda_G2 * c**2 * ||G_k||_2**2.
```

The F-block line search does not account for the resulting usage-penalty
change. The same statistical inconsistency matters for post-step Adam
normalization; it is not specific to L2 or L-BFGS. **Normalization itself is
not the bug; treating this operation as penalty-invariant is the problem.**

Requested fix: implement a mathematically consistent constrained/composite
update. Possibilities include constrained/proximal optimization on unit-norm
loading columns or normalized coordinates with the correct chain rule and
a documented treatment of loading L1. Evaluate step acceptance and stopping
against that same objective. This is not a recommendation to remove the norm
constraint, blindly normalize without rescaling G, or tune the learning rate
until the trace looks flat. If a nonmonotone method is used, document its
acceptance/convergence safeguards rather than requiring every step to decrease.

The baseline shift `G_k <- G_k - min(G_k)` with the compensating intercept
shift is a different operation: for nonnegative usages it decreases either
usage penalty while preserving eta, provided the intercept shift is feasible.
Keep scale and baseline handling conceptually separate. `F_fixed` must remain
bit-identical; `transform` must not move frozen intercepts to perform a shift.

Existing regression `test_loading_rescale_is_not_ridge_objective_invariant`
demonstrates the issue: start with zero-minimum G and F columns of norm 2;
normalizing preserves eta while multiplying the usage ridge contribution by 4.
This test documents the algebra; passing it does not mean the optimizer is fixed.

## 2. Priority: convergence must measure physical-parameter stationarity

Currently `run_fit`/`run_transform` stop after three sufficiently small relative
objective changes. This can detect stagnation rather than an optimum. Including
large likelihood constants in the relative-loss denominator further makes this
test unsuitable as the sole convergence criterion.

Requested diagnostics, with theta fixed while testing mean-model convergence:

- For directly constrained G, use the projected/KKT gradient: ordinary gradient
  for positive entries, `min(gradient, 0)` at the lower bound. A tiny derivative
  with respect to `G_raw` is not sufficient when `G=softplus(G_raw)`; it may
  simply reflect a nearly zero softplus derivative.
- For smooth unit-norm F, use the tangent gradient
  `grad_F - F * colsum(F * grad_F)`. With exact L1, use a valid constrained
  proximal/subgradient residual, including zero loadings, rather than pretending
  the smooth formula alone certifies a nonsmooth optimum.
- Check all other active blocks and bounds. Report whether dispersion was held
  fixed, stabilized by an estimating rule, or actually optimized; do not conflate
  mean-model stationarity with a joint likelihood optimum over dispersion.
- Recheck the returned best checkpoint after restoration, not only the last
  iterate. If export snaps usages to zero, distinguish internal from exported
  parameter diagnostics and bound/report the change it introduces.
- Return an explicit stop reason, residuals, actual iteration count, and timeout
  status. Appending a restored loss value must not count as another iteration.

Observed example: a legacy synthetic fit marked converged/passed by the old
screen had approximate maximum loading tangent / G projected / intercept
gradients of **58 / 1.38 / 13** on its exported parameters. Export snaps small
G values, so these are not a substitute for an internal-state audit, but clearly
warrant one. The independent screen checked the physical parameters directly.

Softplus-zero initialization deserves a regression: a usage initialized very
near zero must be able to activate when its physical reduced gradient calls for
it. A positive jitter start helped diagnose the boundary issue but is not an
adequate general convergence fix.

## 3. Priority: record and protect the actual initialization

`run_fit` records the first loss only **after** the first update.
`model.py:fit` defines `improved_on_init_` by comparison with that first loss,
then warns that the returned fit is the initialization if it did not improve.
That interpretation is incorrect: the first recorded iterate is not the input
initialization, and the initial state is not in the best-checkpoint candidates.

Requested fix:

- Record a true initial objective and checkpoint after initial canonicalization
  and any specified initial dispersion setup, before the first optimizer step.
- Define improvement with a numerical tolerance and clear dispersion semantics.
  If the aim is a fixed-dispersion mean comparison, evaluate both states at the
  same theta. Distinguish that from improvement in the full changing-theta NLL.
- Preserve that initial checkpoint as a legitimate best-state candidate.
- Do not label an already stationary initialization a numerical failure merely
  because it did not improve. Do not prescribe learning-rate reduction as the
  universal remedy for any non-improvement.
- Base convergence status on the state ultimately returned.

## 4. Additional inconsistencies found by code inspection

These have not been separately quantified as causes of the pilot failures.

1. **Chunk-dependent loading penalty in L-BFGS.** `_lbfgs_block` adds the full
   smooth loading-L1 penalty *inside* the sample-chunk loop. Thus M chunks
   apply it M times, while `full_objective` includes it once. Accumulate this
   global penalty once. The grid fits used one chunk, so this does not explain
   their failures, but it matters for scalable fitting.
2. **Two different loading-L1 treatments in the same L-BFGS update.** The closure
   uses `sqrt(F**2 + 1e-8)` and is followed by a separate soft-thresholding step.
   Establish which composite objective/update this implements; do not mix a
   smooth penalty and an additional proximal penalty without a derivation.
   It should not be described simply as exactly "double L1": the two step
   mechanisms have different scales.
3. **Adam proximal threshold uses stale nominal learning rate.** The proximal
   threshold uses `cfg.lr_F`, while plateau decay modifies optimizer parameter
   group learning rates. Reconcile the threshold with the actual effective
   step/metric, including the existing adaptive-denominator floor.
4. **L-BFGS learning-rate diagnostics are misleading.** `_lbfgs_block` constructs
   an optimizer with `lr=1.0`; the generic configured learning rate affects the
   external proximal step, and the plateau code decays the Adam optimizers.
   Expose/document the actual step controls for each algorithm.

## 5. Evidence and a bounded reproduction path

Experiment detail: `docs/worklog/2026-09-15_loading_l1_usage_l2.md` in the parent
agepath repository. Data were MuSC-masked naive-CD4 counts, fixed exposures,
training-null dispersion held fixed, K=5. Real screen: four donors, 200 training
and 100 separate test cells each, 500 genes, two nearby SVD-based starts.

Main grid: `lambda_F = cF * n_train`, cF in {0.0001, 0.001, 0.01}; usage L1=0;
usage L2 in {0.01, 0.1, 1}. Controls: usage L1=10 and no usage penalty, both at
cF=0.001. L1=10 is the old 2,000-gene setting, not auto-L1 for 500 genes.

Artifacts (local; these are not bundled in the note):

```
/data/agepath/pilot/loading_l1_usage_l2_2026-09-16/
  data/                 frozen counts, exposures, null parameters, split manifest
  feasibility/          partial legacy-optimizer sweep, retained as diagnostic
  normalized_v1/        complete independent constrained-objective screen
  full_confirmation/   eight bounded 2,000-gene attempts; all timed out
```

Parent-repository scripts:

- `scripts/pilot/screen_loading_l1_usage_l2.py`: legacy controlled screen and
  model-matched synthetic cases; synthetic cases need no private biological data.
- `scripts/pilot/normalized_loading_solver.py`: independent float64 joint
  L-BFGS-B diagnostic; normalized F inside the objective, direct bounded G,
  smooth L1 epsilon=1e-6, analytic gradients with finite-difference tests.
- `scripts/pilot/screen_normalized_loading_l1_usage_l2.py`: same grid/data using
  that diagnostic, with physical-gradient and separation checks.
- `scripts/pilot/summarize_loading_l1_usage_l2.py`: tables from saved results.

The independent solver passed 45/88 real and 38/66 synthetic attempts under
strict gates; it is **not a proposed drop-in production replacement**. Its
strong-L1 cases are often slow, it uses dense small matrices and smoothed L1,
and all eight full-panel attempts hit their 45s cap. It is a cross-check that
some stationary solutions are obtainable, not a proof of global optimality.

On the real center point (A26, cF=0.001, L2=0.1), numerical preconditioning and
tight loss tolerance gave a 959-iteration fit in ~14s with maximum physical
gradient 0.00549. Stronger ridge real fits still yielded only ~0.12% median
omitted-gene prediction. Do not promise biological recovery from this fix alone.

No experiment was run on GPU. Reproduction should start on small synthetic
CPU data, with explicit thread limits and wall-clock caps, not by rerunning
the full cohort. Saved protocols contain code hashes; use a new run tag for
modified code rather than overwriting prior outputs.

## 6. Acceptance tests for an upstream fix

1. Finite-difference/objective-gradient agreement for L1-only, L2-only and
   combined penalties, including normalization and optional covariates.
2. Objective and gradients invariant to sample chunk size, especially global
   loading-L1 accounting in the L-BFGS closure.
3. Objective-consistent constrained updates and physical stationarity at returned
   checkpoints on small known-signal problems; no full-data global-optimum claim.
4. Fixed-F, fixed-intercept, fixed-dispersion G inference agrees in objective
   and KKT residual with an independent bounded optimizer. For positive L2 this
   G-only objective is strictly convex; do not extend that guarantee to joint F/G.
5. Near-zero usages can activate; genuine boundary optima remain nonnegative;
   norm/baseline transformations have their advertised predictor/penalty effects.
6. Initial-state protection, already-optimal initialization, failed first step,
   best-checkpoint restore, dispersion-phase transitions, and correct stop reasons.
7. `F_fixed` and transform-frozen parameters unchanged; consistent predictions
   across CPU precision/chunk choices within declared tolerances.
8. A bounded sparse-count benchmark at 500 and 2,000 genes with time, iterations,
   objective, stationarity, clipping/boundary status and memory reported. Treat
   budget exits as unresolved, not as biological negatives or successful fits.

Suggested order: resolve the objective/constraint contract and L1 consistency,
add stationarity/initialization diagnostics, then optimize scalability. Increasing
budgets or relaxing convergence criteria alone would not resolve this report.
