"""Extract a bounded discovery-donor panel without loading the full count matrix.

Requires h5py and parquet support (e.g. /data/agepath/venv/bin/python).
Writes a reusable, local NPZ fixture; never changes the source dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h5ad', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--gene-panel', type=Path, required=True)
    ap.add_argument('--label', default='CD4+ T cells_0')
    ap.add_argument('--donors', nargs='+', default=['A26', 'B17', 'E06', 'E18'])
    ap.add_argument('--cells-per-donor', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    panel = json.loads(args.gene_panel.read_text())
    discovery = set(panel['provenance']['discovery_donors'])
    if not set(args.donors) <= discovery:
        raise ValueError('This pilot only uses the predeclared discovery donors')
    meta = pd.read_parquet(args.metadata, columns=['cell_id', 'subcluster', 'donor_id', 'sample_id', 'n_counts'])
    rng = np.random.default_rng(args.seed)
    indices = []
    for donor in args.donors:
        eligible = np.flatnonzero((meta.subcluster == args.label) & (meta.donor_id == donor) & (meta.n_counts > 0))
        if len(eligible) < args.cells_per_donor:
            raise ValueError(f'Not enough cells for {donor}')
        indices.extend(rng.choice(eligible, args.cells_per_donor, replace=False))
    indices = np.sort(indices)
    selected = meta.iloc[indices]
    with h5py.File(args.h5ad, 'r') as h:
        # Verify row alignment rather than assuming the parquet and H5AD agree.
        obs_ids = h['obs/_index'].asstr()[indices]
        if not np.array_equal(obs_ids, selected.cell_id.to_numpy()):
            raise ValueError('Metadata/H5AD cell ordering differs')
        names = h['var/_index'].asstr()[:]
        lookup = {name: i for i, name in enumerate(names)}
        genes = np.array([lookup[g] for g in panel['genes']])
        x = h['X']
        if x.attrs['encoding-type'] != 'csr_matrix':
            raise ValueError('Expected CSR counts')
        # Read full rows to preserve the actual whole-transcriptome totals.
        starts, ends = x['indptr'][indices], x['indptr'][indices + 1]
        counts = np.empty((len(genes), len(indices)), dtype=np.int64)
        totals = np.empty(len(indices))
        for j, (start, end) in enumerate(zip(starts, ends)):
            values, columns = x['data'][start:end], x['indices'][start:end]
            if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                raise ValueError('Expected raw nonnegative integer counts')
            row = sp.csr_matrix((values, columns, [0, len(values)]), shape=(1, len(names)))
            counts[:, j] = row[:, genes].toarray().ravel()
            totals[j] = values.sum()
    if not np.array_equal(totals, selected.n_counts.to_numpy()):
        raise ValueError('Stored library sizes do not match count rows')
    # Entire donor held out for this pilot; no validation-arm donors are used.
    test = selected.donor_id.to_numpy() == args.donors[-1]
    reference = np.median(totals[~test])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=counts, b=np.log(totals/reference), test=test,
                        donor=selected.donor_id.to_numpy(dtype=str), genes=np.asarray(panel['genes']),
                        source_rows=indices)
    report = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  shape=list(counts.shape), zero_fraction=float((counts == 0).mean()),
                  mean_count=float(counts.mean()), median_panel_umi=float(np.median(counts.sum(0))),
                  median_full_umi=float(np.median(totals)), exposure_reference=float(reference),
                  n_train=int((~test).sum()), n_test=int(test.sum()),
                  fixture_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  note='Discovery-donor pilot; preselected gene panel is not independent of these donors.')
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
