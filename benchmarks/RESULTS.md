# Benchmark results

Produced by `bench.py`. Machine: 12-core CPU, RTX 3060 12GB (CUDA), torch 2.13.

> **Status: GPU numbers pending.** The container lost GPU access mid-run
> (NVML error); scaling wall times and the real-data section will be filled
> in once CUDA is restored. Everything below ran on CPU (`--quick` sizes:
> p=600, n=2000, k=6, synthetic data from the package generator), which is
> valid for the *comparative* questions since recovery quality is
> device-independent.

## Open questions from the spec (section 10), resolved empirically

### 1. Optimizer per block: Adam vs L-BFGS vs IRLS

| config (init=svd, softplus G) | wall s (CPU) | outer iters | final loss | recovery* |
|---|---|---|---|---|
| Adam, alternating (default)   | 38.1 | 151 | 2,433,181 | 0.930 |
| Adam, joint                   | 26.3 |  99 | 2,436,064 | 0.941 |
| L-BFGS per block              |  4.8 |  11 | 2,758,600 | 0.000 |

*recovery = mean matched |loading correlation| against the generating factors
(Hungarian assignment).

**Adam wins.** L-BFGS with strong-Wolfe line search steps into the
overflow-clamped region of the NB objective and diverges; with a
revert-on-non-finite guard it simply stalls at the warm start. It is kept in
the code behind `algorithm="lbfgs"` for reference but is not competitive as
implemented. IRLS was not implemented: the `G >= 0` constraint breaks its
per-block least-squares structure (as the spec anticipated), and Adam's
performance left little reason to pursue it.

### 2. Softplus reparametrization vs projected gradient

| G parametrization | wall s | final loss | recovery | exact-zero frac in `G_` |
|---|---|---|---|---|
| softplus (default) | 38.1 | 2,433,181 | 0.930 | 0.69 |
| projected clamp    | 30.0 | 2,429,121 | 0.955 | 0.65 |

Projected gradient is slightly better on this config (lower loss, higher
recovery, comparable speed). Softplus remains the default because `G_raw_`
must be tie-free for downstream rank statistics, and the projected
parametrization's raw values contain exact-zero ties; revisit if the gap
persists across the full grid (GPU run pending).

### 3. Joint vs alternating optimization

Joint Adam over all blocks matched alternating on recovery (0.941 vs 0.930)
and was ~1.4x faster on CPU. Alternating remains the default pending the
full-size GPU comparison; `algorithm="adam_joint"` is supported and safe.

### 4. Theta re-estimation schedule

Resolved during development rather than benchmarked: theta is re-estimated
every 10 outer iterations while the mean model is still moving, then frozen
after the first convergence stall (with one final refresh), so the
convergence criterion is judged at fixed dispersion. Re-estimating every
iteration was visibly unstable (loss spikes at each refresh); the
freeze-on-stall variant removed the spikes without changing the fitted
factors materially.

### 5. Exposure: fixed offset vs fitted b

| mode | wall s | recovery | notes |
|---|---|---|---|
| `"offset"` (default) | 41.0 | 0.943 | |
| `"fit"`              | 80.9 | 0.939 | b agrees with offset b at r > 0.99 |

On synthetic data with exposure_sd = 0.8 the two agree on the factors
(cross-fit matched |corr| ~= 0.99) and on b itself; fitting b doubles wall
time for no recovery gain. Offset confirmed as the right default. Real-data
comparison pending GPU.

## Initialization comparison (spec section 3: "warm-starting matters")

| init | wall s | outer iters | final loss | recovery |
|---|---|---|---|---|
| svd (default) | 38.1 | 151 | 2,433,181 | **0.930** |
| nmf           | 81.6 | 334 | 2,463,888 | 0.887 |
| random        | 53.0 | 220 | 2,517,002 | 0.411 |

The spec expected nmf to be the most reliable; empirically the truncated SVD
of row-centered log1p-normalized counts wins on every axis (speed, loss,
recovery). Random init lands in materially worse optima — warm-starting
matters more than any optimizer choice tested, confirming the spec's prior.

## Findings that changed the implementation (beyond the spec)

- **Usage-baseline flatness.** The predictor is invariant under
  `G_k -> G_k + c, a -> a - c F_k`. Unanchored, fitted G drifts dense, the
  orthant constraint never binds, and recovery drops (0.55-0.85, chaotic
  across float-rounding differences). A small L1 on G (`l1_G = 0.005 p`) plus
  per-iteration min-shift canonicalization anchors the touch-zero
  representative: recovery rose to 0.88-0.94 on the hard synthetic grid and
  became stable across thread counts. See README for discussion.
- **Prox threshold flooring.** Adam-scaled proximal thresholds explode when a
  weak factor's gradients vanish (v_hat -> 0), wiping columns and
  destabilizing fits; flooring the denominator at 0.1x its median fixed a
  bistable failure mode.
- **Dispersion trend fitting** must avoid `torch.linalg.lstsq` (first call in
  a process differs bitwise from later calls, breaking seed reproducibility);
  normal equations are used instead.

## Pending (requires GPU restored)

- Wall time / peak memory vs (n, p, k), including the p=3000, n=20000, k=10
  target (< 60 s spec budget).
- 20 Newsgroups real-data section: deviance vs NMF and GLM-PCA at matched k,
  rotation stability across restarts on real data.
- CPU-vs-GPU wall-time ratio for the fit and transform paths.
