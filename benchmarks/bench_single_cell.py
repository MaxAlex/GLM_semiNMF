"""Bounded single-cell screen, with held-out genes for activity evaluation.

Example: PYTHONPATH=src python benchmarks/bench_single_cell.py --out RUN
Use --real-npz with a fixture from prepare_single_cell.py. Every candidate is
scored at an independently fitted training-null dispersion. Budget exits are
recorded, never counted as certified fits. Optional modes isolate dispersion
calibration and fixed-iteration runtime without biological interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

from bench_optimizer import clean


PROFILES = {
    'legacy': dict(l1_G='auto', l2_G=0., init='svd', dispersion='trend'),
    'small_l1': dict(l1_G=.02, l2_G=0., init='svd', dispersion='trend'),
    'ridge': dict(l1_G=.02, l2_G=.1, init='svd', dispersion='trend'),
    'information': dict(l1_G='information', l2_G=.1, init='svd', dispersion='trend'),
    'pearson': dict(l1_G=.02, l2_G=.1, init='pearson', dispersion='trend'),
    'pooled': dict(l1_G=.02, l2_G=.1, init='svd', dispersion='trend_pooled'),
    'shared': dict(l1_G=.02, l2_G=.1, init='svd', dispersion='shared'),
    'combined': dict(l1_G='information', l2_G=.1, init='pearson', dispersion='trend_pooled'),
}


def synthetic(case, p, n, seed=41):
    import numpy as np
    from glm_seminmf import simulate_nb_seminmf
    # Fixed data seed across starts and configurations. Training and test cells
    # are independent NB observations with the same generating programs.
    sim = simulate_nb_seminmf(p=p, n=n, k=3, random_state=seed,
                             baseline_log_mean=-4 if case != 'moderate' else -2.5,
                             signal_strength=.8, exposure_sd=.8, n_batches=2,
                             batch_strength=.4, dispersion=10.)
    rng = np.random.default_rng(seed+1)
    if case == 'rare_correlated':
        sim.G[:, 2] *= rng.random(n) < .075  # ~3% active cells
        sim.F[:, 1] = .7*sim.F[:, 0] + .7*sim.F[:, 1]
        sim.F[:, 1] /= np.linalg.norm(sim.F[:, 1])
    eta = sim.a[:, None]+sim.b+sim.F@sim.G.T+sim.gamma@sim.Z.T
    mu = np.exp(np.clip(eta, -30, 12))
    x = rng.poisson(rng.gamma(sim.theta[:, None], mu/sim.theta[:, None]))
    test = np.arange(n) >= int(.75*n)
    return dict(X=x, b=sim.b, Z=sim.Z, test=test, F=sim.F, G=sim.G, theta=sim.theta)


def match(a, b):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a = a-a.mean(0); b = b-b.mean(0)
    corr = a.T@b / np.maximum(np.linalg.norm(a, axis=0)[:, None]*np.linalg.norm(b, axis=0), 1e-30)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    return float(np.abs(corr[rows, cols]).mean()), rows, cols


def load_data(config):
    import numpy as np
    if config['real_npz']:
        with np.load(config['real_npz']) as source:
            data = {k:source[k] for k in source.files}
        data['Z'] = None
        return data
    return synthetic(config['case'], config['genes'], config['cells'])


def calibration_worker(config, path):
    os.environ['OMP_NUM_THREADS'] = str(config['threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(config['threads'])
    import warnings
    import numpy as np
    import torch
    from scoring import fit_common_null
    torch.set_num_threads(config['threads'])
    data = load_data(config)
    train = ~data['test']
    z = None if data['Z'] is None else data['Z'][train]
    with warnings.catch_warnings(record=True):
        null = fit_common_null(data['X'][:, train], data['b'][train], z,
                               device=config['device'], max_seconds=config['seconds'])
    np.savez_compressed(path.with_suffix('.npz'), a=null.a_, theta=null.theta_,
                        gamma=np.empty((len(null.a_),0)) if null.gamma_ is None else null.gamma_)
    result = dict(stop_reason=null.stop_reason_, converged=null.converged_,
                  theta_sha256=hashlib.sha256(null.theta_.tobytes()).hexdigest(),
                  seconds=null.total_seconds_, config=config)
    path.write_text(json.dumps(clean(result),indent=2)+'\n')


def candidate_worker(config, path):
    os.environ['OMP_NUM_THREADS'] = str(config['threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(config['threads'])
    import warnings
    import numpy as np
    import torch
    from glm_seminmf import NBGLMSemiNMF
    from glm_seminmf._init import initialize
    from scoring import predictor, score_eta, subset_for_transform
    torch.set_num_threads(config['threads'])
    data = load_data(config)
    x, b, test, z = data['X'], data['b'], data['test'], data['Z']
    train = ~test
    ztrain, ztest = (None, None) if z is None else (z[train], z[test])
    p, n = x.shape
    profile = PROFILES[config['profile']].copy()
    # A perturbed spectral start is genuinely different, unlike svds seed alone.
    t = time.monotonic()
    f, g, a0, gamma0 = initialize(x[:, train], b[train], 3 if not config['real_npz'] else 5,
                            profile['init'], 0, ztrain)
    rng = np.random.default_rng(config['seed'])
    f += rng.normal(0, .02*max(np.abs(f).mean(), 1e-8), f.shape)
    g = np.maximum(g+rng.normal(0, .02*max(g.mean(), 1e-8), g.shape), 0)
    # Preserve nuisance starts from the chosen initializer as well: patching a
    # tuple would otherwise discard Pearson's fitted nuisance coefficients.
    # A worker-local wrapper preserves the nuisance start and avoids repeating
    # the spectral pass. Its separately measured time belongs to total time.
    import glm_seminmf.model as model_module
    original_initialize = model_module.initialize
    def perturbed(*args, **kwargs):
        return f.copy(), g.copy(), a0.copy(), None if gamma0 is None else gamma0.copy()
    model_module.initialize = perturbed
    init_seconds = time.monotonic()-t
    k = f.shape[1]
    model = NBGLMSemiNMF(k, **profile, l1_F=.001*int(train.sum()), exposure=b[train],
                        max_iter=config['max_iter'], max_seconds=config['seconds'],
                        random_state=config['seed'], device=config['device'],
                        reuse_step_size=config['reuse_step_size'])
    if config['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x[:, train], Z=ztrain)
    model_module.initialize = original_initialize
    fit_seconds = time.monotonic()-t
    peak = torch.cuda.max_memory_allocated()/2**20 if config['device']=='cuda' else None

    # The target genes' counts are not passed to transform. The fixed loading
    # rows for those genes were learned only from the training cells.
    gene_rng = np.random.default_rng(92)
    scored = np.zeros(p, dtype=bool)
    scored[gene_rng.choice(p, max(1, p//5), replace=False)] = True
    observed = ~scored
    projection = subset_for_transform(model, observed)
    with warnings.catch_warnings(record=True) as transform_warnings:
        usages = projection.transform(x[observed][:, test], Z=ztest, exposure=b[test])
    from types import SimpleNamespace
    calibration_path = Path(config['calibration_file'])
    with np.load(calibration_path.with_suffix('.npz')) as saved:
        null = SimpleNamespace(a_=saved['a'], theta_=saved['theta'],
                               gamma_=saved['gamma'] if saved['gamma'].shape[1] else None)
    eta = predictor(model, b[test], usages, ztest)
    null_eta = predictor(null, b[test], Z=ztest)
    score = score_eta(x[scored][:, test], eta[scored], null_eta[scored], null.theta_[scored])
    calibration = []
    gene_mean = x[:, train].mean(1)
    for lo, hi in [(0., .01), (.01, .1), (.1, 1.), (1., float('inf'))]:
        take = scored & (gene_mean >= lo) & (gene_mean < hi)
        if take.any():
            calibration.append(dict(mean_range=[lo, hi], genes=int(take.sum()),
                                    **score_eta(x[take][:, test], eta[take], null_eta[take], null.theta_[take])))
    result = dict(config=config, shape=[p,n], zero_fraction=float((x == 0).mean()),
                  mean_count=float(x.mean()), fit_seconds=fit_seconds, extra_start_seconds=init_seconds,
                  n_iter=model.n_iter_, converged=model.converged_, stop_reason=model.stop_reason_,
                  residual=model.stationarity_['max_residual'], work=model.work_,
                  l1_G=model.l1_G_, initial_objective=model.initial_objective_,
                  final_objective=model.final_objective_, peak_gpu_mb=peak,
                  collapsed=int(model.component_stats_.degenerate.sum()),
                  rare=int(model.component_stats_.rare.sum()),
                  zero_usage_fraction=float((model.G_ == 0).mean()),
                  common_null=json.loads(calibration_path.read_text()),
                  transform=dict(converged=projection.transform_converged_,
                                 stop_reason=projection.transform_stop_reason_,
                                 seconds=projection.transform_total_seconds_),
                  score=score, calibration=calibration,
                  warnings=[str(w.message) for w in [*caught, *transform_warnings]])
    if 'F' in data:
        recovery, rows, cols = match(data['F'], model.F_)
        result['loading_recovery'] = recovery
        result['usage_correlations'] = [float(np.corrcoef(data['G'][test, i], usages[:, j])[0,1])
                                       if np.std(usages[:, j]) > 0 and np.std(data['G'][test,i]) > 0 else None
                                       for i,j in zip(rows,cols)]
        result['log_theta_rmse'] = float(np.sqrt(np.mean(np.log(model.theta_/data['theta'])**2)))
    np.savez_compressed(path.with_suffix('.npz'), F=model.F_)
    path.write_text(json.dumps(clean(result), indent=2)+'\n')


def timing_worker(config, path):
    import numpy as np
    import torch
    from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf
    from glm_seminmf._fitting import DataSource, FitState, FitConfig, _step
    torch.set_num_threads(config['threads'])
    sim = simulate_nb_seminmf(p=config['genes'], n=config['cells'], k=5,
                             baseline_log_mean=-3., random_state=17)
    kw = dict(dtype=getattr(torch, config['dtype']), device=config['device'])
    t = lambda a:torch.as_tensor(a, **kw).clone()
    initial = FitState(t(sim.F), t(sim.G*.6), t(sim.a), t(sim.b), None, t(np.log(sim.theta)), False)
    data = DataSource(sim.X, None, config['device'], kw['dtype'], None)
    groups = [['G_raw'], ['F', 'a']]
    import copy
    cfg = FitConfig(lam=.001*config['cells'], lam_G=.02, lam_G2=.1, update_theta=False)
    if hasattr(cfg, 'reuse_step_size'):
        cfg.reuse_step_size = config['reuse_step_size']
    state = copy.deepcopy(initial)
    for group in groups:
        _step(state, data, cfg, group)
    times = []
    for repeat in range(3):
        state = copy.deepcopy(initial)
        if hasattr(cfg, 'step_sizes'):
            cfg.step_sizes.clear(); cfg.work.clear()
        if config['device']=='cuda': torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(5):
            for group in groups:
                for _ in range(5):
                    status = _step(state, data, cfg, group)
                    if status != 'accepted':
                        raise RuntimeError(status)
        if config['device']=='cuda': torch.cuda.synchronize()
        times.append(time.monotonic()-start)
    from glm_seminmf._fitting import full_objective
    result = dict(config=config, seconds=times, median_seconds=float(np.median(times)),
                  work=getattr(cfg, 'work', {}),
                  final_objective=full_objective(state, data, cfg.lam, cfg.lam_G, cfg.lam_G2),
                  torch_version=torch.__version__)
    path.write_text(json.dumps(clean(result), indent=2)+'\n')


def dispersion_screen(args):
    import numpy as np
    import torch
    from glm_seminmf._dispersion import dispersion_from_moments
    torch.set_num_threads(args.threads)
    rows = []
    # Both oracle means and fitted Poisson-intercept means. The latter changes
    # moment variances: do not label the working shrinkage statistic a z test.
    for theta in [10., 100., float('inf')]:
        for fitted in [False, True]:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                mu = np.exp(rng.uniform(np.log(.005), np.log(5.), (args.genes, 1)))
                exposure = np.exp(rng.normal(0, .7, args.cells))
                mean = mu*exposure
                x = rng.poisson(mean if np.isinf(theta) else rng.gamma(theta, mean/theta))
                estimated = x.sum(1)[:, None]/exposure.sum()*exposure if fitted else mean
                moments = [((x-estimated)**2).sum(1), estimated.sum(1), (estimated**2).sum(1)]
                for mode in ['trend', 'trend_pooled', 'shared']:
                    th = dispersion_from_moments(*[torch.tensor(v) for v in moments], mode).exp().numpy()
                    target = min(theta, 1e4)
                    for lo,hi in [(0.,.05),(.05,.5),(.5,float('inf'))]:
                        take = (mu[:,0]>=lo)&(mu[:,0]<hi)
                        rows.append(dict(theta=theta, fitted_mean=fitted, seed=seed, mode=mode,
                                         mean_range=[lo,hi], median_theta=float(np.median(th[take])),
                                         log_theta_rmse=float(np.sqrt(np.mean(np.log(th[take]/target)**2)))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['fit','timing','dispersion'], default='fit')
    ap.add_argument('--real-npz', type=Path)
    ap.add_argument('--genes', type=int, default=300)
    ap.add_argument('--cells', type=int, default=400)
    ap.add_argument('--cases', nargs='+', default=['sparse','moderate','rare_correlated'])
    ap.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0,1])
    ap.add_argument('--max-iter', type=int, default=1000)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--dtype', choices=['float32','float64'], default='float64')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--reuse-step-size', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    # Locate the imported package, including before-change timing checkouts.
    import importlib.util
    package = Path(importlib.util.find_spec('glm_seminmf').origin).parent
    files = sorted(package.glob('*.py')) + [Path(__file__).resolve(), Path(__file__).with_name('scoring.py')]
    protocol = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                    package_path=str(package), source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    note='20% of genes withheld from transform; common theta estimated on training cells only. '
                         'Real-data scores condition on observed full-library sizes; preselected panel is exploratory.')
    if args.real_npz:
        protocol['fixture_sha256'] = hashlib.sha256(args.real_npz.read_bytes()).hexdigest()
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if args.mode == 'dispersion':
        results = dispersion_screen(args)
    else:
        ctx = mp.get_context('spawn')
        results = []
        cases = ['real'] if args.real_npz else args.cases
        if args.mode == 'fit':
            for case in cases:
                config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
                config['case'] = case
                path = args.out/f'calibration_{case}.json'
                proc = ctx.Process(target=calibration_worker,args=(config,path))
                proc.start(); proc.join(args.seconds+60)
                if proc.is_alive():
                    proc.terminate(); proc.join(5)
                    if proc.is_alive(): proc.kill(); proc.join()
                    raise RuntimeError(f'Common-null calibration timed out for {case}')
                if proc.exitcode or not path.exists():
                    raise RuntimeError(f'Common-null calibration failed for {case}')
        tasks = [('timing',0,'timing')] if args.mode == 'timing' else [
            (case, seed, profile) for case in cases for seed in args.seeds for profile in args.profiles]
        for case,seed,profile in tasks:
            config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            config.update(case=case,seed=seed,profile=profile)
            config['calibration_file'] = str(args.out/f'calibration_{case}.json')
            path = args.out/f'{case}_{seed}_{profile}.json'
            proc = ctx.Process(target=timing_worker if args.mode=='timing' else candidate_worker,args=(config,path))
            proc.start()
            # Fit, transform, null fit, spectral start and imports each bounded
            # collectively by the parent, even if a solver deadline overruns.
            proc.join(3*args.seconds+90)
            if proc.is_alive():
                proc.terminate(); proc.join(5)
                if proc.is_alive(): proc.kill(); proc.join()
                result = dict(config=config, stop_reason='process_timeout', converged=False)
            elif proc.exitcode or not path.exists():
                result = dict(config=config,stop_reason='worker_failed',exitcode=proc.exitcode,converged=False)
            else:
                result = json.loads(path.read_text())
            path.write_text(json.dumps(clean(result),indent=2)+'\n')
            results.append(result)
            print(case,seed,profile,result.get('stop_reason','timed'),result.get('fit_seconds'),flush=True)
        if args.mode=='fit':
            import numpy as np
            stability = []
            for case in cases:
                for profile in args.profiles:
                    paths = [args.out/f'{case}_{s}_{profile}.npz' for s in args.seeds]
                    if len(paths)==2 and all(p.exists() for p in paths):
                        a,b = [np.load(p)['F'] for p in paths]
                        stability.append(dict(case=case,profile=profile,loading_agreement=match(a,b)[0]))
            (args.out/'stability.json').write_text(json.dumps(stability,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(clean(results),indent=2)+'\n')


if __name__ == '__main__':
    main()
