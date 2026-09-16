"""Bounded synthetic optimizer screen with physical stationarity diagnostics.

python benchmarks/bench_optimizer.py --out benchmarks/runs/repair_v1
Use --full-grid for the handoff's 11 penalty settings, --seeds 0 1 for two
starts. Each worker has a solver deadline and an outer process deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import time
import warnings


def clean(value):
    import math
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def worker(config, output):
    os.environ['OMP_NUM_THREADS'] = str(config['threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(config['threads'])
    import numpy as np
    import scipy.sparse as sp
    import torch
    from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf
    torch.set_num_threads(config['threads'])
    sim = simulate_nb_seminmf(p=config['p'], n=config['n'] + config['n_test'], k=5,
                             baseline_log_mean=-1.5, dispersion=5, random_state=0)
    # Held-out samples exercise transform against the same loadings.
    train = slice(0, config['n'])
    test = slice(config['n'], config['n'] + config['n_test'])
    x = sp.csc_matrix(sim.X[:, train])
    x_test = sp.csc_matrix(sim.X[:, test])
    # Two starts differ in factors, not merely in an effectively deterministic SVD seed.
    from glm_seminmf._init import initialize
    f, g, _, _ = initialize(x, sim.b[train], 5, 'svd', 0)
    rng = np.random.default_rng(config['seed'])
    f = f + rng.normal(0, .001, f.shape)
    g = np.maximum(g + rng.normal(0, .001, g.shape), 0)
    model = NBGLMSemiNMF(5, l1_F=config['cF'] * config['n'],
                        l1_G=config['l1_G'], l2_G=config['l2_G'],
                        dispersion=sim.theta, exposure=sim.b[train],
                        init=(f, g), dtype=config['dtype'], device=config['device'],
                        max_iter=config['max_iter'], max_seconds=config['seconds'],
                        stationarity_tol=config['stationarity_tol'], random_state=config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x)
    if config['device'] == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t

    # Transform is timed and certified separately from fit.
    t = time.monotonic()
    with warnings.catch_warnings(record=True) as caught_transform:
        warnings.simplefilter('always')
        model.transform(x_test, exposure=sim.b[test])
    if config['device'] == 'cuda':
        torch.cuda.synchronize()
    transform_elapsed = time.monotonic() - t

    result = dict(config=config, torch_version=torch.__version__, numpy_version=np.__version__,
                  density=x.nnz / (config['p'] * config['n']), wall_seconds=elapsed,
                  peak_cpu_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if config['device']=='cuda' else None,
                  initial_objective=model.initial_objective_, final_objective=model.final_objective_,
                  objective_components=model.objective_components_, stationarity=model.stationarity_,
                  n_iter=model.n_iter_, best_iteration=model.best_iteration_,
                  converged=model.converged_, timed_out=model.timed_out_, stop_reason=model.stop_reason_,
                  dispersion_status=model.dispersion_status_, theta_updates=model.theta_updates_,
                  zero_usage_fraction=float((model.G_ == 0).mean()),
                  warnings=[str(w.message) for w in caught],
                  phase_seconds=model.phase_seconds_,
                  safeguard_active=model.stationarity_['safeguard_active'],
                  predictor=model.stationarity_['predictor'],
                  transform=dict(wall_seconds=transform_elapsed,
                                 n_test=config['n_test'],
                                 n_iter=model.transform_n_iter_,
                                 converged=model.transform_converged_,
                                 stop_reason=model.transform_stop_reason_,
                                 max_residual=model.transform_stationarity_['max_residual'],
                                 objective=model.transform_final_objective_,
                                 warnings=[str(w.message) for w in caught_transform]))
    output.write_text(json.dumps(clean(result), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--genes', nargs='+', type=int, default=[500, 2000])
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--test-samples', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--stationarity-tol', type=float, default=.001)
    parser.add_argument('--full-grid', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)  # never overwrite earlier runs
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root/'src/glm_seminmf').glob('*.py')), Path(__file__).resolve()]
    protocol = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    note='Synthetic fixed-theta screen; unresolved budgets are not successes or biological negatives.')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    grid = [(c,0.,g) for c in [.0001,.001,.01] for g in [.01,.1,1.]] + [(.001,10.,0.),(.001,0.,0.)]
    if not args.full_grid:
        grid = [(.001,0.,.1)]
    results = []
    ctx = mp.get_context('spawn')
    for p in args.genes:
        for seed in args.seeds:
            for c, l1, l2 in grid:
                config = dict(p=p,n=args.samples,n_test=args.test_samples,seed=seed,cF=c,l1_G=l1,l2_G=l2,
                              threads=args.threads,dtype=args.dtype,device=args.device,
                              max_iter=args.max_iter,seconds=args.seconds,stationarity_tol=args.stationarity_tol)
                path = args.out/f'p{p}_seed{seed}_cF{c}_g1{l1}_g2{l2}.json'
                proc = ctx.Process(target=worker,args=(config,path))
                proc.start()
                # imports, init, final audit and the transform solve each get a
                # bounded allowance on top of the two solver deadlines.
                proc.join(2 * args.seconds + 60)
                if proc.is_alive():
                    proc.terminate(); proc.join(5)
                    if proc.is_alive():
                        proc.kill(); proc.join()
                    result = dict(config=config, stop_reason='process_timeout',converged=False,timed_out=True)
                elif proc.exitcode or not path.exists():
                    result = dict(config=config,stop_reason='worker_failed',exitcode=proc.exitcode,converged=False)
                else:
                    result = json.loads(path.read_text())
                path.write_text(json.dumps(clean(result),indent=2)+'\n')
                results.append(result)
                print(p,seed,result['stop_reason'],result.get('wall_seconds'),flush=True)
    (args.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
