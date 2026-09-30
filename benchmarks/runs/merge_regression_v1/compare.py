"""Recheck saved merge benchmark results; fail on a numerical regression."""
import json, math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
read=lambda group: json.loads((root/group/'summary.json').read_text())
comparison={}
for kind in ['optimizer','synthetic','real']:
    a,b=read(kind+'_local'),read(kind+('_merged_final' if kind=='optimizer' else '_merged'))
    assert len(a)==len(b)
    rows=[]
    for x,y in zip(a,b):
        keys=['initial_objective','final_objective','n_iter','converged','stop_reason']
        if kind=='optimizer': keys+=['stationarity','objective_components']
        else: keys+=['score','residual','work','l1_G','collapsed','rare','zero_usage_fraction']
        for key in keys:
            assert x[key]==y[key], (kind,key,x[key],y[key])
        for key in ['converged','stop_reason']:
            assert x['transform'][key]==y['transform'][key]
        if kind=='optimizer':
            assert x['transform']['objective']==y['transform']['objective']
        else:
            assert x['common_null']['theta_sha256']==y['common_null']['theta_sha256']
            case,seed,profile=[x['config'][key] for key in ['case','seed','profile']]
            filename=f'{case}_{seed}_{profile}.npz'
            np.testing.assert_array_equal(np.load(root/(kind+'_local')/filename)['F'],np.load(root/(kind+'_merged')/filename)['F'])
        rows.append(dict(case=x['config'].get('case',x['config'].get('p')),profile=x['config'].get('profile'),
                         final_objective=y['final_objective'],stop_reason=y['stop_reason'],n_iter=y['n_iter'],
                         objective_delta=y['final_objective']-x['final_objective'],
                         transform_stop_reason=y['transform']['stop_reason'],
                         score=y.get('score')))
    comparison[kind]=rows
assert read('dispersion_local')==read('dispersion_merged')
comparison['dispersion']=dict(rows=len(read('dispersion_local')),exact_match=True)
a,b=read('timing_local_final')[0],read('timing_merged_final')[0]
assert a['final_objective']==b['final_objective'] and a['work']==b['work']
comparison['timing']=dict(local_seconds=a['seconds'],merged_seconds=b['seconds'],
                         local_median=a['median_seconds'],merged_median=b['median_seconds'],
                         ratio=b['median_seconds']/a['median_seconds'],exact_objective=True,work=b['work'])
a,b,c=read('fixed_incoming'),read('fixed_local'),read('fixed_merged')
rows=[]
for old,local,merged in zip(a,b,c):
    assert old['data_sha256']==local['data_sha256']==merged['data_sha256']
    assert merged['basis_max_error']==0
    assert merged['final_objective'] <= local['final_objective']+1e-6
    assert merged['final_objective'] <= old['final_objective']
    assert merged['held_out_nll'] <= old['held_out_nll']
    rows.append(dict(seed=merged['seed'], incoming_objective=old['final_objective'],local_objective=local['final_objective'],
                     merged_objective=merged['final_objective'], incoming_held_out_nll=old['held_out_nll'],
                     merged_held_out_nll=merged['held_out_nll'],stop_reason=merged['stop_reason'],
                     basis_max_error=merged['basis_max_error'],merged_residual=merged['stationarity']['max_residual']))
comparison['fixed_basis']=rows
(root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
