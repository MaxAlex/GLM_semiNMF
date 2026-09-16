# NB-GLM semi-NMF: objective and solution method

The optimizer uses physical nonnegative usages, unit-norm trainable loading
columns, exact loading L1, and backtracking against one fixed-dispersion
objective. This document supersedes the previous Adam/rescaling method.
[OPTIMIZER_PLAN.md](OPTIMIZER_PLAN.md) records the motivating review;
[OPTIMIZER_CHANGELOG.md](OPTIMIZER_CHANGELOG.md) describes migration and validation.

## Objective and constraints

For counts X with p features and n samples, k components, and optional
sample covariates Z:

```
eta = a[:, None] + b[None, :] + gamma @ Z.T + F @ G.T
X[g,c] ~ NB2(exp(eta[g,c]), theta[g])
J_theta = NLL + lambda_F * sum(abs(F))
              + lambda_G1 * sum(G) + 0.5 * lambda_G2 * sum(G**2)
G >= 0; ||F[:,k]||_2 = 1 for trainable loading columns
```

NB2 has variance mu + mu²/theta. Penalties are on the summed-likelihood scale.
Existing loading/usage-L1 defaults are unchanged; usage L2 defaults to zero.
There is no smooth loading-L1 surrogate. `F_fixed` retains its supplied scale,
values, and order; the sphere constraint applies only to trainable columns.
Zero initial trainable loading columns are rejected with an explicit error,
rather than silently reinitialized. Usages may collapse to zero; such factors
remain present and are flagged in component statistics.

Signed F models both increases and decreases. Nonnegative G restricts allowable
factor transformations but does not prove unique identification of a finite
factorization. Dense supports and correlated or duplicated factors remain
statistical difficulties. Stationarity does not establish global optimality,
unique identification, or biological predictive usefulness.

## Normalization and baseline shifts

Initial trainable F columns are normalized and their scales absorbed into G
before the initial checkpoint is recorded. This preserves the predictor but
changes the penalties; it defines the starting point, not an optimizer step.
There is no post-step F/G rescaling.

During fit, a column baseline can be shifted by

```
m[k] = min(G[:,k])
G <- G - m; a <- a + F @ m
```

This preserves eta and does not increase either usage penalty for nonnegative
weights. It is separate from normalization. Transform freezes a and never
performs this shift. Fixed F is never normalized, updated, or reordered.

## Likelihood and derivatives

Writing d = eta - log(theta), the stable mean-dependent negative log-likelihood
is `theta * softplus(d) + X * softplus(-d)`. The full objective additionally
contains `-lgamma(X+theta) + lgamma(theta) + lgamma(X+1)`. These constants are
omitted only within fixed-theta line searches; full reported objectives retain
them. No straight-through clamp is used by the training likelihood.

With s = sigmoid(d), derivatives in eta are

```
r = theta * sigmoid(d) - X * sigmoid(-d)
h = (theta + X) * sigmoid(d) * sigmoid(-d)
```

Analytic physical gradients and diagonal curvature are accumulated across
sample chunks. The F penalty is global and counted once. G penalties are
linear/quadratic in physical G. Objective/gradient semantics are independent
of chunk count, within floating-point accumulation tolerance.

Moment estimation and deviance calculation still guard exponentiation at
eta in [-30,30]; that safeguard is not the training objective. Large predictors
are therefore separately reported in the diagnostics. The conditional
stationarity claim concerns the exact training likelihood, not a clamped
surrogate or the MoM estimating rule.

## Constrained composite steps

`algorithm="proximal"` alternates G/b and F/a/gamma blocks. Every active block
is updated, including fitted exposure when k=0. Each block takes `inner_steps`
curvature-preconditioned steps, default 5. The diagonal curvature is floored
at 1e-6; F uses one curvature scalar per column, the maximum over feature rows.
Curvature is a preconditioner, not a claim that the coupled Hessian is diagonal.

For ordinary parameters, propose x - t grad/metric. Project G onto the
nonnegative orthant. For each F column, solve exactly

```
minimize ||f-z||²/2 + tau*||f||_1, subject to ||f||_2=1
z = f_old - t*grad_NLL/metric; tau = t*lambda_F/metric
```

If soft-thresholding z by tau leaves a nonzero vector, normalize that vector.
If all entries threshold to zero, choose the signed coordinate maximizing
`abs(z)-tau`, with deterministic first-index tie breaking. This solves the
sphere-constrained subproblem, including its nonsmooth exceptional case;
it is not an unconstrained update followed by scale transfer into G.

Backtracking starts at `learning_rate` (default 1), halves the multiplier, and
accepts a finite feasible proposal satisfying composite Armijo decrease with
coefficient 1e-4. The slope includes the smooth directional derivative and
the exact change in loading L1. At most 30 trials are made. Failed steps are
restored; there is no unrelated proximal threshold or hidden Adam schedule.

A factor contribution larger than 15 in absolute value triggers a separation
safeguard during trial evaluation. Backtracking may find a safe smaller step;
otherwise fitting returns its best checkpoint with `safeguard_hit`. It does
not clip an accepted iterate or claim KKT stationarity for a coupled hard-bound
problem. An initially unsafe state is returned with the same explicit failure;
it is not silently repaired. The fit reports the maximum contribution.

## Physical stationarity

An independent audit recomputes gradients on parameter values in float64,
chunk by chunk, including when optimization uses float32.

For G, residuals are ordinary gradients at strictly positive entries and
`min(gradient,0)` at exact zeros. No softplus chain factor enters the audit.
For unit-norm F, with g the smooth gradient and lambda its L1 weight:

```
nu[k] = -sum(F[:,k] * (g[:,k] + lambda*sign(F[:,k])))
residual = g + lambda*sign(F) + F*nu   where F != 0
residual = sign(g)*max(abs(g)-lambda,0) where F == 0
```

This selects the sphere multiplier from the unit constraint and includes valid
L1 subgradients at zeros. With lambda=0 it reduces to the tangent gradient.
The audit also checks a, trainable b, gamma, nonnegativity, loading norms,
finiteness, predictor range, and factor contribution. Frozen blocks are excluded
from stationarity conditions and must remain unchanged.

`stationarity_tol` is the maximum absolute residual in each active block,
default 1e-3. Feasibility tolerance is max(1e-10, 10*compute-dtype epsilon).
Scaled residuals are provided as descriptive values: each block's absolute
residual divided by 1 plus its L1 weight (zero weight for intercept/covariate
blocks). Success uses absolute residuals, never full-NLL-relative scaling.
`tol` only detects stagnation; it cannot establish convergence.

## Initialization, checkpoints, and dispersion

SVD, NMF, random, and supplied F/G starts remain available. Initial
canonicalization and the initial dispersion setup precede the first recorded
objective. Scalar/vector fixed theta is installed before optimization. There
is no random jitter added to distinguish truly zero usages.

Every completed outer iteration produces a history value. The true initial
state is a candidate for best-state restoration. The returned best complete
checkpoint includes theta and is re-audited. `n_iter_` counts completed outer
iterations; `loss_[0]` is initialization, so its length is n_iter_+1.
Restoration is an event in `history_`, not a fake iteration. `final_objective_`
refers to the returned state and can differ from the last trace entry.

For estimated dispersion, retain MoM residual estimation and existing feature,
shared, or trend shrinkage. An initial estimate is followed by refreshes every
10 iterations, at most 15 refreshes. Freeze on a small log-theta change (<0.05),
three small loss changes, the refresh cap, or halfway through the iteration
budget to leave a fixed-theta finish. On freezing, restore the best complete
checkpoint visited and polish the mean at its theta. Record refresh/freeze
reasons and restored iteration. A refresh is an estimating-rule event and need
not decrease J. Loss-stall bookkeeping resets across theta changes.

`converged_` certifies the returned mean parameters conditional on their theta.
It is never a claim of a joint likelihood optimum over theta, nor does a refresh
cap imply statistical stability. `improved_on_init_` compares full initial and
returned states. `mean_improved_on_init_` evaluates both mean states at the
returned theta, with a numerical tolerance. An already-stationary start is a
successful zero-iteration fit even without improvement.

## Returned values, transform, and budgets

G_ is the physical constrained solution, without export snapping. G_internal_
is a copy of the same values; export objective/predictor changes are zero.
`G_raw_` equals G_ by default. Legacy softplus mode supplies inverse-softplus
compatibility values, using a finite dtype-tiny floor at zero. Both modes can
have ties. These are not invented continuous ranks for inactive samples.

Transform freezes F, a, gamma, theta and optionally b. It uses the same G
objective, physical residuals, checkpoint protection, and deadline handling.
Results are in `transform_*` diagnostic attributes; fitted attributes are not
modified. With positive usage L2 and all other parameters fixed, the G-only
objective is strictly convex. This does not extend to joint F/G fitting.

Outcomes include stationary, max_iter, timeout, stalled_nonstationary,
line_search_failed, nonfinite, and safeguard_hit. `max_seconds` is cooperative:
initialization, an in-flight tensor/data pass, and final certification can
exceed it. Optional null/deviance scoring stops when the deadline expires and
reports unavailable values as NaN, with `finalization_timed_out_`. The benchmark
harness adds an outer process cap. Report total call time as well as solver time.

## Precision, complexity, and remaining work

Float64 is the correctness default. Float32 may stop nonstationary when its
precision cannot resolve a requested residual; tolerances are never silently
relaxed. Fixed F can promote precision to preserve supplied values. Existing
sparse staging and sample chunking remain; no full p-by-n mean is stored.
Each derivative/likelihood pass costs O(p*n*k) work and O(p*chunk) workspace,
plus parameter arrays. Backtracking adds measured extra passes.

The legacy algorithm names issue a migration warning and run this same solver;
they are not separate benchmark candidates. Working second-order and joint
methods, GPU acceleration, stochastic minibatching, and batching independent
fits remain subsequent optimizations subject to these correctness gates.
Profile-likelihood dispersion, penalty selection, uncertainty, and stronger
identification assumptions remain separate statistical projects. Historical
GPU results do not establish that the under-60-second target has been met.
