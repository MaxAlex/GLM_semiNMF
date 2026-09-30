"""Compare consensus-basis projection across revisions via PYTHONPATH.

Uses APIs common to both merge parents, each solver's native default step size,
fixed dispersion, and an independent NumPy/SciPy objective calculation. The
ordinary simulator is unchanged across parents, so seeds generate identical data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import gammaln
import torch

import glm_seminmf
from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf
from bench_optimizer import clean


def nll(x, eta, theta):
    delta = eta - np.log(theta)[:, None]
    t = theta[:, None]
    return float(np.sum(t * np.logaddexp(0, delta) + x * np.logaddexp(0, -delta)
                        - gammaln(x + t) + gammaln(t) + gammaln(x + 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    package = Path(glm_seminmf.__file__).parent
    protocol = dict(arguments={k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()},
                    package_path=str(package), torch_version=torch.__version__,
                    numpy_version=np.__version__,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [*sorted(package.glob('*.py')), Path(__file__)]},
                    note='Same data, penalties, precision, and iteration cap; native solver defaults. '
                         'Parent convergence flags use different definitions. Held-out NLL scores '
                         'new cells with all genes observed, not withheld-gene prediction.')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for seed in args.seeds:
        sim = simulate_nb_seminmf(p=150, n=300, k=3, n_batches=2,
                                 baseline_log_mean=-2., dispersion=10.,
                                 dispersion_spread=0., random_state=seed)
        train, test = slice(0, 220), slice(220, None)
        model = NBGLMSemiNMF(3, l1_F=.22, l1_G=.1, l2_G=.1,
                            exposure=sim.b[train], dispersion=10., dtype='float64',
                            max_iter=args.max_iter, random_state=seed, device=args.device)
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(sim.X[:, train], sim.Z[train], F_fixed=sim.F)
        fit_seconds = time.monotonic() - start
        eta = (model.a_[:, None] + model.b_ + model.F_ @ model.G_.T
               + model.gamma_ @ sim.Z[train].T)
        train_nll = nll(sim.X[:, train], eta, model.theta_)
        objective = (train_nll + .22 * np.abs(model.F_).sum()
                     + .1 * model.G_.sum() + .05 * np.square(model.G_).sum())
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as transform_warnings:
            warnings.simplefilter('always')
            g = model.transform(sim.X[:, test], sim.Z[test], exposure=sim.b[test])
        transform_seconds = time.monotonic() - start
        eta_test = (model.a_[:, None] + sim.b[test] + model.F_ @ g.T
                    + model.gamma_ @ sim.Z[test].T)
        row = dict(seed=seed, data_sha256=hashlib.sha256(sim.X.tobytes()).hexdigest(),
                   fit_seconds=fit_seconds, transform_seconds=transform_seconds,
                   final_objective=float(objective), train_nll=train_nll,
                   held_out_nll=nll(sim.X[:, test], eta_test, model.theta_),
                   basis_max_error=float(np.max(np.abs(model.F_ - sim.F))),
                   usage_correlations=[float(np.corrcoef(model.G_[:, j], sim.G[train, j])[0, 1])
                                       if np.std(model.G_[:, j]) > 0 else None
                                       for j in range(3)],
                   n_iter=model.n_iter_, converged=bool(model.converged_),
                   stop_reason=getattr(model, 'stop_reason_', None),
                   stationarity=getattr(model, 'stationarity_', None),
                   transform_converged=getattr(model, 'transform_converged_', None),
                   warnings=[str(w.message) for w in [*caught, *transform_warnings]])
        rows.append(row)
        (args.out / f'seed{seed}.json').write_text(json.dumps(clean(row), indent=2) + '\n')
        print(seed, row['final_objective'], row['stop_reason'], flush=True)
    (args.out / 'summary.json').write_text(json.dumps(clean(rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
