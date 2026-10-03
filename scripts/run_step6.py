"""Regenerate Step 6 numerical data and exact certificates, from repository root."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import mpmath as mp
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint

from quantum_ess.ewl_resolution import (RESIDENTS, TOLERANCE, LOCAL_RADIUS,
    canonical_grid, is_self, exact_a0, tie_a1, batch_compositions,
    symbolic_certificate, symbolic_coefficients, local_derivatives,
    high_precision_coefficients)
from quantum_ess.ewl_invasion import DEFAULT_DIAGNOSTIC_FREQUENCIES, classify_power_coefficients

SEEDS = (20261002, 20261003, 20261004)
GRIDS = ((21,11),(41,21),(81,41))
DE = dict(maxiter=500, popsize=15, tol=1e-10, atol=1e-12, polish=False, workers=1, updating='immediate')
FREQUENCIES = (0., *DEFAULT_DIAGNOSTIC_FREQUENCIES)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def packed(q):
    return json.dumps(np.asarray(q).tolist(),separators=(',',':'),allow_nan=False)


def record(name, q, arrays, i, stage, aliases=None):
    k,rt,rp,payoff=RESIDENTS[name]
    a=arrays['a'][i,0];b=arrays['b'][i,0]
    classification,order,leading=classify_power_coefficients(a.tolist(),TOLERANCE)
    return dict(resident=name,k=k,resident_theta=rt,resident_phi=rp,resident_payoff=payoff,
        stage=stage,source_coordinates=packed(aliases if aliases is not None else [q]),
        mutant_theta=float(q[0]),mutant_phi=float(q[1]),canonical_theta=float(q[0]),canonical_phi=float(q[1]),
        distance=float(np.linalg.norm(np.array(q)-[rt,rp])),
        u_R=packed(arrays['u_R'][i,0]),u_M=packed(arrays['u_M'][i,0]),
        bernstein_differences=packed(b),power_coefficients=packed(a),
        all_focal_u_R=packed(arrays['u_R'][i]),all_focal_u_M=packed(arrays['u_M'][i]),
        first_above_threshold=order,leading_order=order,leading_coefficient=leading,
        numerical_classification=classification,threshold=TOLERANCE,
        diagnostic_delta=packed([sum(a[j]*e**j for j in range(k)) for e in FREQUENCIES]),
        delta_001=float(sum(a[j]*.01**j for j in range(k))),
        focal_position_spread=float(arrays['focal_spread'][i]),
        placement_spread=float(arrays['placement_spread'][i]),
        probability_error=float(arrays['probability_error'][i]),
        reconstruction_error=float(arrays['polynomial_error'][i]),
        exact_a0_evaluation=float(exact_a0(name,*q)))


def numerical_search():
    rows=[]
    def save(name,order,domain,method,seed,bounds,start,result,elapsed,decode):
        q=np.asarray(decode(result.x),float)
        # The canonical theta=pi edge represents one physical operation.
        if abs(q[0]-math.pi)<=16*math.ulp(math.pi):q=np.array([math.pi,0.])
        raw=record(name,q,batch_compositions(name,[q]),0,'optimizer')
        rows.append(dict(resident=name,coefficient_order=order,domain=domain,method=method,
            seed=seed,bounds=packed(bounds),start=packed(start) if start is not None else '',
            config=json.dumps(DE if method=='differential_evolution' else {'ftol':1e-14,'gtol':1e-10,'maxiter':1000}),
            success=bool(result.success),message=str(result.message),nit=int(result.nit),
            nfev=int(result.nfev),objective_value=float(-result.fun),
            constraint_violation=float(getattr(result,'maxcv',0.)),elapsed_seconds=elapsed,
            theta=float(q[0]),phi=float(q[1]),distance=raw['distance'],is_self=is_self(name,q),
            power_coefficients=raw['power_coefficients'],delta_001=raw['delta_001']))
    for name,(_,rt,rp,_) in RESIDENTS.items():
        bounds=[(0.,math.pi),(0.,math.pi/2)]
        objective=lambda q: -float(exact_a0(name,*q))
        for domain in ['full_rectangle','distance_ge_0.1']:
            constraints=() if domain=='full_rectangle' else (NonlinearConstraint(lambda q:np.linalg.norm(q-[rt,rp]),LOCAL_RADIUS,np.inf),)
            for seed in SEEDS:
                start=time.perf_counter()
                r=differential_evolution(objective,bounds,seed=seed,constraints=constraints,**DE)
                save(name,0,domain,'differential_evolution',seed,bounds,None,r,time.perf_counter()-start,lambda q:q)
        # Explicit boundary objectives, retaining corners plus optimized edges.
        for axis in (0,1):
            for fixed in bounds[axis]:
                other=1-axis
                def decode(z,axis=axis,fixed=fixed):
                    q=np.empty(2);q[axis]=fixed;q[1-axis]=z[0];return q
                seed=SEEDS[0];start=time.perf_counter()
                r=differential_evolution(lambda z:objective(decode(z)),[bounds[other]],seed=seed,**DE)
                save(name,0,f'boundary_axis{axis}_{fixed}','differential_evolution',seed,[bounds[other]],None,r,time.perf_counter()-start,decode)
        starts=np.vstack((np.array([[0,0],[0,math.pi/2],[math.pi,0],[math.pi,math.pi/2]]),np.random.default_rng(SEEDS[0]).random((4,2))*[math.pi,math.pi/2]))
        for q in starts:
            start=time.perf_counter();r=minimize(objective,q,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
            save(name,0,'full_rectangle','L-BFGS-B',SEEDS[0],bounds,q,r,time.perf_counter()-start,lambda q:q)
    for edge,bounds,domain in [
        ('theta_0',[(0,math.pi/2)],'B_Z0_theta0'),
        ('phi_pi_over_2',[(0,math.pi)],'B_Z0_phi_pi2'),
        ('phi_pi_over_2',[(0,math.pi/2-LOCAL_RADIUS)],'B_Z0_complement_left'),
        ('phi_pi_over_2',[(math.pi/2+LOCAL_RADIUS,math.pi)],'B_Z0_complement_right')]:
        for seed in SEEDS:
            start=time.perf_counter()
            r=differential_evolution(lambda z:-float(tie_a1(edge,z[0])),bounds,seed=seed,**DE)
            decode=(lambda z:[0.,z[0]]) if edge=='theta_0' else (lambda z:[z[0],math.pi/2])
            save('K4_B',1,domain,'differential_evolution',seed,bounds,None,r,time.perf_counter()-start,decode)
    return rows


def precision_checks(optimizers, summaries):
    rows=[];derivative_checks={}
    with mp.workdps(80):
        for name in RESIDENTS:
            rt=mp.pi/2 if name=='K4_B' else mp.mpf(0)
            rp=2*mp.pi/5 if name=='K5_A' else mp.pi/2
            probes=[('ordinary',mp.mpf('.83'),mp.mpf('.91')),
                ('near_theta_1e-8',rt+mp.mpf('1e-8'),rp),
                ('near_theta_1e-20',rt+mp.mpf('1e-20'),rp),
                ('near_phi_1e-20',rt,rp-mp.mpf('1e-20'))]
            # Recompute the strongest observed double candidate and finest-grid candidate.
            opt=max((r for r in optimizers if r['resident']==name and not r['is_self']),key=lambda r:r['delta_001'])
            fine=next(r for r in summaries if r['resident']==name)
            for label, tt, pp in [('optimizer_best_double',opt['theta'],opt['phi']),('finest_grid_best',fine['mutant_theta'],fine['mutant_phi'])]:
                t=mp.mpf(str(tt));p=mp.mpf(str(pp))
                if tt==0.:t=mp.mpf(0)
                if tt==math.pi:t=mp.pi
                if pp==math.pi/2:p=mp.pi/2
                if pp==RESIDENTS[name][2]:p=rp
                probes.append((label,t,p))
            if name=='K4_B':probes += [('tie_theta0',mp.mpf(0),mp.pi/7),('tie_phi_pi2',mp.pi/3,mp.pi/2)]
            for label,t,p in probes:
                high=high_precision_coefficients(name,t,p)
                ordinary=batch_compositions(name,[(float(t),float(p))])['a'][0,0]
                first=next((j for j,a in enumerate(high['a']) if abs(a)>mp.mpf('1e-60')),None)
                rows.append(dict(resident=name,probe=label,theta=mp.nstr(t,82),phi=mp.nstr(p,82),digits=80,
                    high_precision_coefficients=json.dumps([mp.nstr(a,75) for a in high['a']]),
                    double_coefficients=packed(ordinary),max_double_error=float(max(abs(mp.mpf(float(v))-a) for v,a in zip(ordinary,high['a']))),
                    high_precision_noise_threshold='1e-60 (diagnostic, not exact equality)',high_precision_leading_order=first,
                    high_precision_leading_value=mp.nstr(high['a'][first],75) if first is not None else '',
                    probability_error=mp.nstr(high['probability_error'],12),
                    ordinary_coordinates_collapse_to_self=is_self(name,[float(t),float(p)])))
            # Independent real centered finite differences of dense MP tensor payoffs.
            h=mp.mpf('1e-12');cache={}
            def at(i,j):
                if (i,j) not in cache:cache[i,j]=high_precision_coefficients(name,rt+i*h,rp+j*h)['a']
                return cache[i,j]
            symbolic=local_derivatives(name);checks=[]
            for m,d in enumerate(symbolic):
                g=[(at(1,0)[m]-at(-1,0)[m])/(2*h),(at(0,1)[m]-at(0,-1)[m])/(2*h)]
                hh=[[(at(1,0)[m]-2*at(0,0)[m]+at(-1,0)[m])/h**2,(at(1,1)[m]-at(1,-1)[m]-at(-1,1)[m]+at(-1,-1)[m])/(4*h*h)],
                    [0,(at(0,1)[m]-2*at(0,0)[m]+at(0,-1)[m])/h**2]];hh[1][0]=hh[0][1]
                err=max([abs(float(g[i])-d['gradient_float'][i]) for i in range(2)]+[abs(float(hh[i][j])-d['hessian_float'][i][j]) for i in range(2) for j in range(2)])
                assert err<1e-15,(name,m,err)
                checks.append(dict(order=m,gradient=[mp.nstr(v,40) for v in g],hessian=[[mp.nstr(v,40) for v in row] for row in hh],maximum_error_to_symbolic_float=err))
            derivative_checks[name]=dict(digits=80,step='1e-12',method='real centered finite differences of independent dense tensor-matrix payoffs',analytic_extension_for_derivatives_only=True,coefficients=checks)
    return rows,derivative_checks


def run(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    convergence=[];raw=[];summary=[]
    for name in RESIDENTS:
        best=None
        for nt,np_ in GRIDS:
            grid=canonical_grid(nt,np_);points=[q for q in grid if not is_self(name,q)]
            a=batch_compositions(name,points)
            rows=[record(name,q,a,i,f'{nt}x{np_}',grid[q]) for i,q in enumerate(points)]
            raw.extend(rows)
            best=max(rows,key=lambda row:row['delta_001'])
            # Different coefficient orders have different meanings; also retain per-order maxima.
            leading_best=max(rows,key=lambda row:row['leading_coefficient'])
            order_max={str(m):max((r['leading_coefficient'] for r in rows if r['leading_order']==m),default=None) for m in range(RESIDENTS[name][0])}
            convergence.append(dict(resident=name,theta_count=nt,phi_count=np_,distinct_operations=len(grid),mutants_excluding_self=len(rows),
                maximum_delta_001=best['delta_001'],theta=best['mutant_theta'],phi=best['mutant_phi'],distance=best['distance'],
                strongest_leading_coefficient=leading_best['leading_coefficient'],corresponding_order=leading_best['leading_order'],
                leading_theta=leading_best['mutant_theta'],leading_phi=leading_best['mutant_phi'],leading_distance=leading_best['distance'],
                per_order_maxima=json.dumps(order_max),apparent_neutral_count=sum(r['leading_order'] is None for r in rows),
                apparent_invader_count=sum(r['numerical_classification']=='rare_mutant_invades' for r in rows)))
            print(name,nt,np_,best['delta_001'],flush=True)
        summary.append({**best,'step5_status':'neutral_or_weak_candidate','step6_status':'proven_ESS_restricted_EWL',
            'strongest_definition':'maximum DeltaPi(0.01) among finest-grid nonself mutants; continuum supremum 0 has no nonself maximizer',
            'neutral_set':'none distinct; B has a0 tie edges resolved by a1' if name=='K4_B' else 'none distinct'})
    write_csv(out/'mutant_coefficients.csv.gz',raw)
    write_csv(out/'grid_convergence.csv',convergence)
    write_csv(out/'resident_summary.csv',summary)
    optimizers=numerical_search();write_csv(out/'coefficient_optimizer_runs.csv',optimizers)
    print('optimizer runs',len(optimizers),'successes',sum(r['success'] for r in optimizers),flush=True)
    cert=symbolic_certificate();dump(out/'certification_summary.json',cert)
    expressions={}
    for name in RESIDENTS:
        variables,ur,um,b,a=symbolic_coefficients(name)
        expressions[name]=dict(variables=list(map(str,variables)),u_R=list(map(str,ur)),u_M=list(map(str,um)),b=list(map(str,b)),a=list(map(str,a)))
    dump(out/'symbolic_coefficients.json',expressions)
    local={name:dict(coefficients=local_derivatives(name),feasible_cone=('dtheta>=0,dphi<=0' if name in ('K2_Q','K4_Q') else 'dphi<=0' if name=='K4_B' else 'dtheta>=0'),
        radius=LOCAL_RADIUS,eigendirections='coordinate axes; repeated zero eigenvalues allow any basis',
        a0_taylor=('9/4*dphi - 9/8*dtheta^2*dphi - 3/8*dphi^3 + O(||d||^5)' if name=='K4_B' else
            '-alpha/4*dtheta^2-beta*dphi^2+alpha/48*dtheta^4+beta/4*dtheta^2*dphi^2+beta/3*dphi^4+O(||d||^6)'),
        alpha=(3 if name=='K2_Q' else 9 if name=='K4_Q' else (9+5*math.sqrt(5))/2) if name!='K4_B' else None,
        beta=(2 if name=='K2_Q' else 6 if name=='K4_Q' else 8) if name!='K4_B' else None,
        null_direction_resolution='on dphi=0: a0 identically zero; a1=-3/4*sin(dtheta)^2=-3/4*dtheta^2+1/4*dtheta^4+O(dtheta^6)' if name=='K4_B' else 'negative definite a0 Hessian on all nonzero feasible directions') for name in RESIDENTS}
    precision,checks=precision_checks(optimizers,summary);write_csv(out/'precision_validation.csv',precision)
    for name in local:local[name]['independent_check']=checks[name]
    dump(out/'local_derivative_analysis.json',local)
    dump(out/'neutral_sets.json',dict(
        K2_Q={'Z0_including_self':'{Q}','distinct_neutral_set':'empty'},
        K4_Q={'Z0_including_self':'{Q}','distinct_neutral_set':'empty'},
        K5_A={'Z0_including_self':'{(0,2*pi/5)}','distinct_neutral_set':'empty'},
        K4_B={'Z0_including_self':'{theta=0,0<=phi<=pi/2} union {phi=pi/2,0<=theta<=pi}; theta=pi aliases canonical D',
              'a1_theta0':'-3/4','a1_phi_pi2':'-3/4*cos(theta)^2','Z1_including_self':'{B}',
              'distinct_neutral_set':'empty','equality_evidence':'exact polynomial remainders zero; independent 80-digit tensor evaluations'}))
    metadata=dict(schema_version=1,utc_finished=datetime.now(timezone.utc).isoformat(),
        command='PYTHONPATH=src studio-python scripts/run_step6.py --output-dir '+str(out),
        git_head_at_run=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_tracked_files_at_start=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines(),
        step5_merge_commit='655c86574b6c989f25153c7ad500b5b15d6e8316',step5_validated_commit='b174ec298187655e8d2eb143fe5509d7fcbc60d8',
        baseline_commit='5827d9aaf631f67a02713eb8643609a7730707d2',python=sys.version,executable=sys.executable,
        platform=platform.platform(),hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),cpu_count=os.cpu_count(),
        workers=1,grids=GRIDS,seeds=SEEDS,optimizer_parameters=DE,diagnostic_frequencies=FREQUENCIES,
        tolerance=TOLERANCE,precision_digits=80,precision_diagnostic_threshold='1e-60',local_radius=LOCAL_RADIUS,
        runtime_seconds=time.perf_counter()-started,mutant_rows=len(raw),optimizer_runs=len(optimizers),
        optimizer_successes=sum(r['success'] for r in optimizers),
        dependency_versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','sympy','mpmath','matplotlib','pandas','nbformat','nbclient']},
        model='unchanged restricted two-parameter EWL; gamma=pi/2; global parity-adjusted entangler; pairwise summed PD; infinite well-mixed population',
        certification='exact symbolic identities and analytic sign proof; optimizers/grids are cross-checks, not the proof',
        future_steps_executed=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/quantum_ess/ewl_resolution.py'),Path('pyproject.toml')]})
    dump(out/'run_metadata.json',metadata)
    dump(out/'data_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='data_manifest.json'})
    print(json.dumps({'runtime_seconds':metadata['runtime_seconds'],'mutant_rows':len(raw),'optimizer_runs':len(optimizers)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path('results/step6'))
    run(parser.parse_args().output_dir)
