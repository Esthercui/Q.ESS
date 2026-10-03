"""Saved research results must be reproducible, internally consistent, and exact."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import differential_evolution
from quantum_ess.ewl_resolution import exact_a0, tie_a1, symbolic_certificate

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/step6'


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Step6ArtifactTests(unittest.TestCase):
    def test_full_saved_artifact_schema(self):
        result=script('validate_step6_results').validate(RESULTS)
        self.assertEqual(result['validation'],'passed')
        self.assertEqual(result['mutant_rows'],17360)

    def test_every_exact_saved_certificate_rederives(self):
        saved=json.loads((RESULTS/'certification_summary.json').read_text())
        self.assertEqual(saved,symbolic_certificate())

    def test_global_seed_reproduction_and_metadata(self):
        rows=script('validate_step6_results').read_csv(RESULTS/'coefficient_optimizer_runs.csv')
        global_rows=[r for r in rows if r['method']=='differential_evolution']
        self.assertEqual(len(global_rows),52)
        self.assertTrue(all(r['success']=='True' for r in global_rows))
        # Reproduce a full-domain coefficient run and a nontrivial exact-tie run.
        for row in [next(r for r in global_rows if r['resident']=='K5_A' and r['domain']=='full_rectangle'),
                    next(r for r in global_rows if r['domain']=='B_Z0_complement_left')]:
            fn=(lambda z:-float(tie_a1('phi_pi_over_2',z[0]))) if int(row['coefficient_order'])==1 else (lambda z:-float(exact_a0(row['resident'],*z)))
            result=differential_evolution(fn,json.loads(row['bounds']),seed=int(row['seed']),**json.loads(row['config']))
            self.assertEqual(result.success,row['success']=='True')
            self.assertEqual(result.nfev,int(row['nfev']))
            self.assertAlmostEqual(-result.fun,float(row['objective_value']),delta=1e-14)

    def test_no_distinct_neutral_in_exact_result(self):
        sets=json.loads((RESULTS/'neutral_sets.json').read_text())
        self.assertTrue(all(v['distinct_neutral_set']=='empty' for v in sets.values()))
        rows=script('validate_step6_results').read_csv(RESULTS/'precision_validation.csv')
        for r in rows:
            self.assertNotEqual(r['high_precision_leading_order'],'')
            self.assertLess(float(r['high_precision_leading_value']),0)

    def test_grid_maxima_converge_to_self_without_becoming_positive(self):
        rows=script('validate_step6_results').read_csv(RESULTS/'grid_convergence.csv')
        for name in ['K2_Q','K4_Q','K4_B','K5_A']:
            v=[r for r in rows if r['resident']==name]
            ds=[float(r['maximum_delta_001']) for r in v]
            self.assertTrue(ds[0]<ds[1]<ds[2]<0)
            self.assertTrue(all(int(r['apparent_invader_count'])==0 for r in v))
            self.assertAlmostEqual(float(v[-1]['distance']),math.pi/80)


if __name__=='__main__':unittest.main()
