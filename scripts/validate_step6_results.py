"""Read-only schema, reconstruction, invariance, and provenance validation."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from quantum_ess.ewl_resolution import RESIDENTS, is_self
from quantum_ess.ewl_invasion import bernstein_to_power_coefficients, classify_power_coefficients


def read_csv(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:return list(csv.DictReader(f))


def validate(root):
    root=Path(root);meta=json.loads((root/'run_metadata.json').read_text())
    assert meta['schema_version']==1
    assert meta['tolerance']==1e-8 and meta['future_steps_executed']==[]
    rows=read_csv(root/'mutant_coefficients.csv.gz'); assert len(rows)==17360
    residual=0.;counts={};max_errors={k:0. for k in ['focal_position_spread','placement_spread','probability_error','reconstruction_error']}
    for r in rows:
        name=r['resident'];k=RESIDENTS[name][0]
        assert int(r['k'])==k and not is_self(name,[float(r['mutant_theta']),float(r['mutant_phi'])])
        theta,phi=float(r['canonical_theta']),float(r['canonical_phi'])
        assert 0<=theta<=math.pi and 0<=phi<=math.pi/2
        if theta==math.pi:assert phi==0
        ur,um,b,a=[json.loads(r[key]) for key in ['u_R','u_M','bernstein_differences','power_coefficients']]
        assert len(ur)==len(um)==len(b)==len(a)==k
        for key in ['all_focal_u_R','all_focal_u_M']:
            matrix=json.loads(r[key]);assert len(matrix)==k and all(len(row)==k for row in matrix)
        assert all(math.isfinite(v) for group in [ur,um,b,a] for v in group)
        residual=max(residual,max(abs(v-u-bj) for u,v,bj in zip(ur,um,b)),max(abs(x-y) for x,y in zip(bernstein_to_power_coefficients(b),a)))
        status,order,leading=classify_power_coefficients(a)
        assert status==r['numerical_classification'] and ('' if order is None else str(order))==r['leading_order']
        assert leading==float(r['leading_coefficient'])
        diag=json.loads(r['diagnostic_delta'])
        for e,saved in zip(meta['diagnostic_frequencies'],diag):
            direct=sum(math.comb(k-1,j)*e**j*(1-e)**(k-1-j)*b[j] for j in range(k))
            residual=max(residual,abs(direct-saved))
        counts[name,r['stage']]=counts.get((name,r['stage']),0)+1
        for key in max_errors:max_errors[key]=max(max_errors[key],float(r[key]))
    assert residual<1e-8 and max(max_errors.values())<1e-8
    convergence=read_csv(root/'grid_convergence.csv');assert len(convergence)==12
    for r in convergence:
        expected={(21,11):221,(41,21):841,(81,41):3281}[int(r['theta_count']),int(r['phi_count'])]
        assert int(r['distinct_operations'])==expected
        assert counts[r['resident'],r['theta_count']+'x'+r['phi_count']]==expected-1
        assert int(r['apparent_invader_count'])==0
        selected=[x for x in rows if x['resident']==r['resident'] and x['stage']==r['theta_count']+'x'+r['phi_count']]
        assert max(float(x['delta_001']) for x in selected)==float(r['maximum_delta_001'])
    optimizers=read_csv(root/'coefficient_optimizer_runs.csv');assert len(optimizers)==meta['optimizer_runs']
    assert sum(r['success']=='True' for r in optimizers)==meta['optimizer_successes']
    for r in optimizers:
        assert r['seed'] and r['config'] and r['bounds'] and r['message']
        assert int(r['nfev'])>0 and math.isfinite(float(r['objective_value']))
        assert len(json.loads(r['power_coefficients']))==RESIDENTS[r['resident']][0]
        if r['domain']=='distance_ge_0.1':assert float(r['distance'])>=.1-1e-12
    cert=json.loads((root/'certification_summary.json').read_text());assert set(cert)==set(RESIDENTS)
    for r in cert.values():assert set(r['exact_checks'].values())=={'0'}
    precision=read_csv(root/'precision_validation.csv');assert len(precision)==26
    for r in precision:assert int(r['digits'])==80 and float(r['max_double_error'])<1e-8
    for name,data in json.loads((root/'local_derivative_analysis.json').read_text()).items():
        assert len(data['coefficients'])==RESIDENTS[name][0]
        assert max(x['maximum_error_to_symbolic_float'] for x in data['independent_check']['coefficients'])<1e-15
    for filename,digest in json.loads((root/'data_manifest.json').read_text()).items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==digest,filename
    for filename,digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    return dict(validation='passed',mutant_rows=len(rows),grid_summaries=len(convergence),reconstruction_error=residual,
        maximum_errors=max_errors,optimizer_runs=len(optimizers),
        global_converged=sum(r['success']=='True' and r['method']=='differential_evolution' for r in optimizers),
        local_failed=sum(r['success']=='False' and r['method']=='L-BFGS-B' for r in optimizers),
        high_precision_probes=len(precision),exact_identities=sum(len(c['exact_checks']) for c in cert.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',default='results/step6')
    print(json.dumps(validate(p.parse_args().results_dir),indent=2))
