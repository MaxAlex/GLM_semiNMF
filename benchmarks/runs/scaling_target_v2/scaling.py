"""Bounded check of the spec's p=3000, n=20000, k=10 under-60-second target."""
import os, json, time, warnings, sys
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np, torch
torch.set_num_threads(4)
from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf

p, n, k = 3000, 20000, 10
sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=0)
out = []
for device, dtype in [('cuda', 'float32'), ('cuda', 'float64'), ('cpu', 'float32')]:
    warnings.filterwarnings('ignore')
    m = NBGLMSemiNMF(k, l1_F=0.001*n, dispersion=sim.theta, exposure=sim.b,
                     random_state=0, device=device, dtype=dtype,
                     max_iter=2000, max_seconds=300)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.monotonic(); m.fit(sim.X)
    if device == 'cuda': torch.cuda.synchronize()
    wall = time.monotonic() - t
    row = dict(device=device, dtype=dtype, wall_s=round(wall, 1),
               total_seconds=round(m.total_seconds_, 1), n_iter=m.n_iter_,
               stop_reason=m.stop_reason_, converged=bool(m.converged_),
               max_residual=float(m.stationarity_['max_residual']),
               s_per_iter=round(wall/max(m.n_iter_, 1), 3),
               peak_gpu_mb=round(torch.cuda.max_memory_allocated()/2**20, 1) if device=='cuda' else None,
               dev_expl=round(float(m.deviance_explained_), 4))
    out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=2)
