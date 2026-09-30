# Merge validation — 2026-09-30

No numerical regression was observed in the rerun cases. The merged code
preserves the local constrained optimizer and integrates the incoming
consensus-basis API, simulation helpers, fractional counts, sparse SVD fix,
and initialization-progress warning.

Parents: local `62b024c`, incoming `11084b7`. Both parent packages were extracted
unchanged with `git show` and selected using `PYTHONPATH`. All runs used the
same `uv.lock` environment: Python 3.12.3, torch 2.13.0+cu130, NumPy 2.5.2,
and an RTX 3060. Each protocol records the imported package and source hashes;
the protocol Git revision is the local parent because the merge was not yet
committed. See [environment.json](environment.json).

## Resolution decisions

- Keep the local proximal solver, exact penalties, physical stationarity
  checks, best-checkpoint restoration, divergence detection, deadlines, and
  single-cell options. Its existing L2 usage penalty covers the incoming L2
  implementation. The optimizer module is unchanged from the local parent.
- Fixed loadings infer `n_components`, expose both `F_fixed_` and
  `fixed_loadings_`, and initialize usages by projection without an unrelated
  SVD. Explicit `(F0, G0)` starts still supply usages. Supplied values, scale,
  and order are preserved, promoting precision when required. Local fixed
  dispersion and external-exposure support remain available.
- The old Adam-specific fourfold step increase is superseded by the new
  solver's curvature scaling and backtracking, also used by transform.
- Warn when an uncertified fit has not improved within numerical tolerance;
  already stationary starts need no warning. Retain the separate comparison
  at final dispersion. Tests now exercise these semantics instead of requiring
  the old Adam divergence failure to recur.
- Preserve fractional-count acceptance and its existing normalization
  heuristic. Explicitly reject nonfinite values, which could otherwise bypass
  that heuristic. README documents that fractional maxima below 30 are rejected.
- Preserve shared-basis simulation and `draw_nb_counts`; validate malformed
  fixed bases. Retain vector-shape handling for sparse and implicit SVD.

## Measurements

The assertions in [compare.py](compare.py) regenerate
[comparison.json](comparison.json) from the saved results.

| Comparison against local parent | Outcome |
|---|---|
| Fixed-theta optimizer, 500 and 2,000 genes, 200 training / 100 test samples | Exact objectives, residuals, iterations, and transform objectives |
| Sparse/moderate single-cell synthetic data, legacy/combined options, four fits | Exact objectives, scores, loading arrays, resolved penalties, and work counts |
| Real 2,000-gene / 800-cell fixture, legacy/ridge/combined options, three fits | Exact objectives, scores, loading arrays, and convergence status |
| Dispersion calibration, two seeds, 108 reported rows | Exact match |
| 50 GPU block updates, 1,000 genes / 3,000 cells, three repeats | Median 0.959 s before, 0.949 s after; identical objective and work |

Both optimizer fits and their transforms certified stationary. The objectives
were 72,975.11316938941 and 290,524.24749082106, after 261 and 199 iterations.
The final rerun (`optimizer_merged_final`) includes the finite-input validation
fix. The timing comparison uses `timing_*_final`, run after other work ended;
earlier timing and optimizer artifacts are retained. The roughly 1% timing
difference is not evidence of a speedup, and this measures block updates,
not complete-fit time.

Synthetic single-cell fits reached their 300-iteration caps; all four
transforms certified, but their common nulls did not. On real data, the legacy
fit certified after 13 iterations; ridge and combined reached the 30-iteration
cap. The ridge transform also reached its cap. These are comparisons of
identical budgeted results, not new convergence or biological claims. The real
fixture is the existing preselected discovery panel, not independent validation.

The fixed-basis benchmark uses two seeded 150-gene / 300-cell datasets with
covariates, 220 training cells, fixed theta=10, and identical penalties. It
scores the objective independently with NumPy/SciPy. All supplied bases are
bit-identical after fitting.

| Seed | Incoming objective | Local objective | Merged objective | Incoming / merged held-out NLL |
|---|---:|---:|---:|---:|
| 0 | 18,343.091066 | 17,831.695731 | 17,831.695731 | 7,136.012726 / 6,967.904307 |
| 1 | 16,758.250755 | 16,196.469077 | 16,196.469077 | 6,414.919032 / 6,223.673166 |

Merged versus local objectives differ by at most 2.84e-9. Both new-solver fits
reach the 1,000-iteration cap, with residuals 0.00112 and 0.00230 against a
0.001 target. The incoming solver's convergence flags use its older criterion
and are not comparable certificates. Fixed-basis held-out NLL observes all
genes of new cells during projection; it is not withheld-gene prediction.
One incoming usage column collapsed; its undefined correlation is stored as
JSON null. The benchmark harness now serializes that case directly as null.

## Tests and reproduction

All **154 unique tests passed**: the initial full suite (135), followed by
the fixed-basis/simulation/sparse-SVD files (30, including 13 new cases), and
input validation (17, including six new cases). Commands and durations are in
[tests.json](tests.json). Compilation, conflict-marker review, and Git
whitespace checks also passed.

The workspace's `.venv` link was broken; validation used an isolated environment
at `/tmp/glm-seminmf-merge-venv`, installed with
`uv sync --frozen --group dev --group benchmark` and `UV_PROJECT_ENVIRONMENT`
set to that directory. No dependency versions were changed.

To rerun, extract each parent's `src` directory into a separate directory,
select it with `PYTHONPATH`, and use the same interpreter for all revisions.
Use fresh output directories; every run's `protocol.json` records its arguments.
The principal commands, with output-directory arguments omitted, are:

```sh
python benchmarks/bench_optimizer.py --genes 500 2000 --seeds 0 --device cuda --threads 1 --max-iter 2000 --seconds 180
python benchmarks/bench_single_cell.py --cases sparse moderate --profiles legacy combined --seeds 0 --max-iter 300 --seconds 180 --device cuda --threads 1
python benchmarks/bench_single_cell.py --real-npz benchmarks/data/naive_cd4_discovery_800.npz --profiles legacy ridge combined --seeds 0 --max-iter 30 --seconds 180 --device cuda --threads 1
python benchmarks/bench_single_cell.py --mode dispersion --genes 1000 --cells 500 --seeds 0 1 --device cuda --threads 1
python benchmarks/bench_single_cell.py --mode timing --genes 1000 --cells 3000 --device cuda --threads 1
python benchmarks/bench_fixed_basis.py --seeds 0 1 --max-iter 1000 --device cpu
python benchmarks/runs/merge_regression_v1/compare.py
```

The dispersion mode performs its moment calculations on CPU regardless of the
harness device option. No prior benchmark artifacts or user files were replaced.
