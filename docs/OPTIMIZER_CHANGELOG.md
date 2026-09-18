# Optimizer repair: migration and validation notes

This records what changed when the solver was rebuilt to optimize one declared
constrained objective and certify the parameters it returns, why each change
was made, and what a caller has to do about it.

Motivating review: [`optimizer_work_handoff.md`](../optimizer_work_handoff.md).
Plan: [`OPTIMIZER_PLAN.md`](OPTIMIZER_PLAN.md). Resulting contract:
[`METHOD.md`](METHOD.md). Sections of
[the implementation spec](../nb-glm-seminmf-implementation-spec.md) that
prescribe post-step rescaling, softplus boundary behavior, loss-only stopping,
or tie-free raw usages are historical and are superseded by `METHOD.md`.

The biological model and the regularization defaults are unchanged. Optional
usage L2 defaults to zero. A stationary fit is not a claim of global
optimality, unique factor identification, or biological predictive value.

## Why the solver was rebuilt

The previous solver did not optimize the objective it reported.

1. **Loading normalization sat outside the optimization step.** Each iteration
   normalized `F` columns and pushed the norm into `G`. That preserves `eta`
   but multiplies the loading L1 by `1/c` and the usage L2 by `c**2`, so the
   line search accepted steps against a different objective than the one being
   reported. Normalization is not itself wrong; treating it as penalty
   invariant is.
2. **Convergence measured loss stagnation, not stationarity.** Three small
   relative objective changes ended a fit, with large likelihood constants in
   the denominator. A tiny derivative with respect to `G_raw` under
   `G = softplus(G_raw)` can also just mean a dead softplus derivative.
3. **The recorded "initial" state was the first iterate, not the input.** So
   `improved_on_init_` compared against something the caller never supplied,
   and the real initialization was not a candidate for best-state restoration.
4. Smaller inconsistencies: the loading L1 was added once per sample chunk
   inside the L-BFGS closure (so `M` chunks applied it `M` times); that closure
   mixed a smoothed `sqrt(F**2 + 1e-8)` penalty with a separate soft-threshold
   step; the Adam proximal threshold used the nominal `lr_F` while plateau
   decay changed the actual group learning rates; fixed scalar dispersion was
   ignored (`log_theta` stayed 0, i.e. theta=1); the `G`/`b` block was skipped
   entirely when `k=0` even with `b` trainable; and export snapped usages after
   the loss and deviance had already been computed.

## What changed

### Objective and constraints

One objective, evaluated the same way everywhere:

```
J_theta = NLL(a + b + gamma @ Z.T + F @ G.T; theta)
          + l1_F * sum(abs(F)) + l1_G * sum(G) + 0.5 * l2_G * sum(G**2)
G >= 0;  ||F[:, k]||_2 = 1 for trainable loading columns
```

`objective_components()` returns `nll`, `loading_l1`, `usage_l1`, `usage_l2`
and `total` separately, and global penalties are counted once per pass
regardless of chunking. Column normalization now happens **once**, during
initial canonicalization, before the initial objective is recorded; there is no
post-step rescaling. The baseline shift `G_k -> G_k - min(G_k)` with
`a -> a + F @ m` is kept separate: it preserves `eta` and cannot increase
either usage penalty, and transform never performs it.

### Steps

`algorithm="proximal"` alternates `G`/`b` and `F`/`a`/`gamma` blocks with
curvature-preconditioned composite steps and Armijo backtracking whose slope
includes the exact change in loading L1. `F` columns are updated by the exact
prox of L1 plus the unit-sphere indicator (`sphere_l1_prox`), including the
case where every coefficient thresholds to zero. `G` is optimized directly in
the nonnegative orthant, so an entry at zero activates when its physical
gradient is negative. There is no smoothed-L1 surrogate and no separate
proximal step layered on top of one.

### Certification

`stationarity()` is an independent float64 audit of the values actually
returned: ordinary gradients at positive `G`, `min(gradient, 0)` at zero; for
`F`, the sphere multiplier plus admissible `[-1, 1]` subgradients at zero
loadings; plus `a`, trainable `b`, `gamma`, feasibility, and finiteness.
`converged_` requires every active block's absolute residual to meet
`stationarity_tol` at the returned checkpoint with theta fixed. `tol` now only
detects stagnation, which triggers an audit rather than declaring success.

### Separation safeguard (revised again after the rebuild)

The `max|F[:, k]| * max(G[:, k]) <= 15` contribution limit used to be enforced
by rejecting any trial step that crossed it. That was wrong twice over:

- Legitimate fits cross the level **transiently**. On a routine synthetic
  problem (`simulate_nb_seminmf(p=300, n=500, k=3, random_state=6)`) the
  contribution rises from 3.5 at initialization to a peak near 15.5 around
  iteration 20-30, then decays to 12.5 and settles below the generating value
  of 11.4. The destination was always feasible; only the path was not.
- Rejection cannot recover from its own boundary. Once an iterate sits on the
  limit, the admissible step length collapses toward zero while backtracking
  only reaches `2^-29` of the initial step, so every trial fails. That fit
  aborted at iteration 18 with a physical residual of 1488 and factor recovery
  0.812, where continuing reached residual 57 and recovery 0.936.

The level is now a **divergence trigger**, not a bound. Crossing it is
reported (`stationarity_["safeguard_active"]`, with `max_contribution`) but
never rejects a step. Fitting stops with `safeguard_hit`
only when the contribution at least doubles over `safeguard_patience` (10)
consecutive completed iterations while staying above the trigger — sustained
growth, which a transient peak and a plateau both fail to produce.

One consequence is worth knowing. NB improvement from `eta -> -inf` saturates
exponentially, so a genuinely separating factor's gradient falls below any
fixed absolute `stationarity_tol` at a **finite** usage: such a fit reports
`stationary` at a large contribution rather than diverging, and tightening the
tolerance moves the stopping point further out.

Measurement also shows the contribution level carries little information about
separation at realistic sizes: a constructed separating factor saturates near
11, while ordinary converged fits at p=300, n=500 sit at 15-17. It is an
extreme-order statistic over `p*n` pairs and, with unit-norm loadings, tracks
problem size. So `safeguard_active` is reported but not warned about. The
warning attached to a returned fit is the predictor range instead: entries
outside `+/-30` are where moment estimation and deviance use a clamped mean,
making `deviance_explained_` and `theta_` approximate. Divergence proper is
caught by the growth test, which is scale free.

### Reported state

Usages are no longer snapped on export: `G_`, `G_raw_` and `G_internal_` are
the same physical constrained values, `export_objective_change_` is 0, and
boundary ties are real rather than disguised by jitter.

## API changes

| Before | Now |
|---|---|
| `algorithm="adam"`, `"adam_joint"`, `"lbfgs"` | deprecated aliases of `"proximal"`; they warn and run the same solver |
| `g_parametrization="softplus"` | deprecated; usages are optimized directly. The option now only selects inverse-softplus *output* values |
| `learning_rate` as an Adam step size | initial multiplier of a curvature-scaled trial step, halved by line search |
| `len(loss_) == n_iter_` | `len(loss_) == n_iter_ + 1`; `loss_[0]` is the true initial objective and restoration is an event in `history_`, not an iteration |
| `converged_` from loss stagnation | physical KKT residuals at the returned checkpoint |
| `G_raw_` tie-free pre-softplus values | equal to `G_`; ties are genuine |
| dispersion scalar silently ignored | honored; also available per call as `fit(..., theta_fixed=...)` |

New attributes: `stop_reason_`, `stationarity_`, `timed_out_`,
`initial_objective_`, `final_objective_`, `objective_components_`,
`best_iteration_`, `history_`, `dispersion_status_`, `theta_updates_`,
`mean_improved_on_init_`, `initial_at_final_theta_`, `phase_seconds_`,
`total_seconds_`, `finalization_timed_out_`, `compute_dtype_`,
`export_objective_change_`, `export_max_predictor_change_`, and the
corresponding `transform_*` diagnostics. New constructor options:
`l2_G`, `stationarity_tol`, `max_seconds`, `inner_steps`, `dtype`. New fit
arguments: `F_fixed`, `theta_fixed`.

Stop reasons: `stationary`, `numerically_stationary`, `max_iter`, `timeout`,
`stalled_nonstationary`, `line_search_failed`, `nonfinite`, `safeguard_hit`.

`numerically_stationary` exists because an absolute gradient tolerance is not
always reachable: on larger problems the step construction runs out of
precision while the fit is already final. Measured on p=600, n=2000, k=6, a run
that stopped `line_search_failed` at residual 1.5e-2 accepted **zero** further
steps when restarted with fresh curvature, and its objective, recovery and
deviance matched a float32 run that stopped at a residual 10,000x larger. The
label was wrong, not the fit. The certificate uses the objective changes the
line search has already computed, so it costs no extra passes, and it is
refused when any block outside the exhausted group is still above tolerance.

### Migrating

- Replace `converged_` checks that assumed loss stagnation. A fit can now
  legitimately end `max_iter` with a small residual; read `stationarity_`.
- Anything indexing `loss_` by iteration must account for `loss_[0]` being the
  initialization.
- Drop `algorithm=` and `g_parametrization=` arguments; they no longer select
  anything and will warn.
- Code relying on `G_raw_` being tie-free for rank statistics needs a
  different tie-break: the ties are real and were previously manufactured.
- `float64` is the default dtype. Pass `dtype="float32"` for the old
  precision, and expect fits that cannot resolve `stationarity_tol` to stop
  nonstationary rather than silently relaxing it.

## Validation

`tests/test_optimizer.py` holds the acceptance suite requested by the handoff:

1. Float64 finite differences against the assembled objective for L1-only,
   L2-only and combined penalties, with covariates, plus the normalized-
   coordinate chain rule checked against autograd.
2. Objective and gradients invariant to chunk count, including global loading
   L1 accounting.
3. `sphere_l1_prox` checked against a brute-force grid search over the circle,
   including the all-thresholded case.
4. Fixed-`F`/intercept/dispersion `G` inference agreeing in objective and KKT
   residual with an independent SciPy `L-BFGS-B` solve on an independently
   written NumPy objective.
5. Near-zero usages activating from exactly 0 and 1e-100; genuine boundary
   optima staying nonnegative; normalization and baseline shift having their
   advertised, distinct penalty effects.
6. Stationary initialization returning in zero iterations; failed first step,
   `max_iter`, timeout, nonfinite and best-restore paths all preserving the
   initial checkpoint; dispersion freeze/restore transitions.
7. `F_fixed`, `theta_fixed` and transform-frozen parameters bit-identical
   before and after; float32/float64 predictions agreeing within declared
   tolerances; the float32 audit not relaxing its tolerance.
8. Separation: self-limiting behavior reported honestly, sustained growth
   stopping the fit, transient overshoot above the trigger not being fatal,
   and the level test not entering the line search.

Bounded synthetic performance evidence is produced by
`benchmarks/bench_optimizer.py` and written up in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). Budget exits are
reported as unresolved, never as successes or as negative results.
