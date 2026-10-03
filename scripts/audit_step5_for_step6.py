"""Reconstruct both frozen Step 5 datasets and freshly check selected pairs."""
import csv,gzip,json,math,hashlib,argparse
from pathlib import Path
from quantum_ess import EWLStrategy,KPlayerEWLGame,analyze_resident_mutant
from quantum_ess.ewl_invasion import bernstein_to_power_coefficients,classify_power_coefficients,population_fitness

def audit(root):
    root=Path(root)
    with gzip.open(root/'resident_mutant_pair_results.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    ce=fe=ee=0.
    for r in rows:
        comp=json.loads(r['composition_payoffs']);b=[c['mutant_payoff']-c['resident_payoff'] for c in comp]
        a=bernstein_to_power_coefficients(b);ce=max(ce,max(abs(x-y) for x,y in zip(a,json.loads(r['power_coefficients']))))
        assert classify_power_coefficients(a)[0]==r['rare_mutant_classification']
        for d in json.loads(r['diagnostic_frequencies']):
            for typ in ['resident','mutant']:fe=max(fe,abs(population_fitness([c[typ+'_payoff'] for c in comp],d['epsilon'])-d[typ+'_payoff']))
    summaries=list(csv.DictReader(open(root/'resident_summary.csv')))
    expected=['neutral_or_weak_candidate','not_ESS','neutral_or_weak_candidate','neutral_or_weak_candidate','neutral_or_weak_candidate','not_ESS']
    assert [s['resident_status'] for s in summaries]==expected
    for s,expected_payoff in zip(summaries,[3,4.5,9,6.75,12,9]):
        k=int(s['k']);rt=float(s['resident_theta']);rp=float(s['resident_phi']);R=EWLStrategy(rt,rp)
        game=KPlayerEWLGame(k,gamma=math.pi/2)
        assert abs(game.expected_payoffs([R]*k)[0]-expected_payoff)<1e-8
        best=max((r for r in rows if int(r['k'])==k and float(r['resident_theta'])==rt and float(r['resident_phi'])==rp),key=lambda r:float(r['DeltaPi_epsilon_reference']))
        pair=analyze_resident_mutant(game,R,EWLStrategy(float(best['mutant_theta']),float(best['mutant_phi'])),.01)
        for v in pair.position_results:ee=max(ee,max(abs(x-y) for x,y in zip(v.power_coefficients,json.loads(best['power_coefficients']))))
    opt=list(csv.DictReader(open(root/'continuous_optimizer_runs.csv')))
    assert len(opt)==16 and all(r['differential_evolution_success']=='True' for r in opt)
    assert max(ce,fe,ee)<1e-8
    return dict(rows=len(rows),coefficient_error=ce,frequency_error=fe,six_pairs_engine_error=ee,statuses=expected,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*') if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat-dir',default='results/step5_macmini_20260828');p.add_argument('--output',default='results/step6/baseline_audit.json');args=p.parse_args()
    results={'step5':audit('results/step5'),'step5_macmini_20260828':audit(args.repeat_dir)}
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    print('Both Step 5 datasets reproduce; six classifications and payoffs unchanged.')
