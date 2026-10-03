"""Independent Step 6 checks; existing Step 1--5 tests are untouched."""
import math
import unittest

import mpmath as mp
import numpy as np
import sympy as S

from quantum_ess import EWLStrategy, KPlayerEWLGame, analyze_resident_mutant
from quantum_ess.ewl_invasion import bernstein_to_power_coefficients
from quantum_ess.ewl_resolution import (RESIDENTS, canonical_grid, is_self, feasible,
    batch_compositions, exact_a0, tie_a1, symbolic_coefficients,
    symbolic_certificate, local_derivatives, high_precision_coefficients)


class ResolutionTests(unittest.TestCase):
    def test_canonical_counts_and_aliases(self):
        for nt,np_,count in [(21,11,221),(41,21,841),(81,41,3281)]:
            grid=canonical_grid(nt,np_)
            self.assertEqual(len(grid),count)
            self.assertEqual(sum(map(len,grid.values())),nt*np_)
            self.assertEqual(len(grid[(math.pi,0.)]),np_)
            for name in RESIDENTS:
                self.assertEqual(sum(is_self(name,q) for q in grid),1)

    def test_maynard_smith_identity(self):
        rr,mr,rm,mm=S.symbols('rr mr rm mm')
        a0=mr-rr; a1=(mm-rm)-(mr-rr)
        self.assertEqual(S.expand(a0-(mr-rr)),0)
        self.assertEqual(S.expand(a1.subs(mr,rr)-(mm-rm)),0)
        game=KPlayerEWLGame(2,gamma=math.pi/2)
        resident=EWLStrategy(.45,.61);mutant=EWLStrategy(.82,.94)
        result=analyze_resident_mutant(game,resident,mutant,.01).position_results[0]
        rr_=game.expected_payoffs([resident,resident])[0]
        mr_=game.expected_payoffs([mutant,resident])[0]
        rm_=game.expected_payoffs([resident,mutant])[0]
        mm_=game.expected_payoffs([mutant,mutant])[0]
        np.testing.assert_allclose(result.power_coefficients,[mr_-rr_,mm_-rm_-(mr_-rr_)],atol=1e-14,rtol=0)
        # The actual implementation must implement b1-b0 (not b1 off the tie).
        self.assertEqual(bernstein_to_power_coefficients([2.,5.]),(2.,3.))
        for mr_,rr_,rm_,mm_ in [(2,2,4,3),(5,3,2,1)]:
            a=bernstein_to_power_coefficients([mr_-rr_,mm_-rm_])
            self.assertEqual(a[0],mr_-rr_)
            self.assertEqual(a[1],mm_-rm_-a[0])

    def test_all_positions_match_frozen_engine(self):
        rng=np.random.default_rng(20261002)
        for name,(k,t,p,_) in RESIDENTS.items():
            points=rng.random((3,2))*[math.pi,math.pi/2]
            fast=batch_compositions(name,points)
            for i,q in enumerate(points):
                old=analyze_resident_mutant(KPlayerEWLGame(k,gamma=math.pi/2),EWLStrategy(t,p),EWLStrategy(*q),.01)
                for f,v in enumerate(old.position_results):
                    np.testing.assert_allclose(fast['a'][i,f],v.power_coefficients,atol=3e-12,rtol=0)
                    np.testing.assert_allclose(fast['u_R'][i,f],[c.resident_payoff for c in v.composition_payoffs],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(fast['u_M'][i,f],[c.mutant_payoff for c in v.composition_payoffs],atol=1e-12,rtol=0)
            for metric in ['probability_error','placement_spread','focal_spread','polynomial_error']:
                self.assertLess(max(fast[metric]),1e-11)

    def test_symbolic_identities_and_zero_set(self):
        cert=symbolic_certificate()
        self.assertEqual(len(cert),4)
        for c in cert.values():self.assertEqual(set(c['exact_checks'].values()),{'0'})
        # In the positive octant, B's sx=0 is exactly the union of two planes.
        x,y,s=S.symbols('x y s',nonnegative=True)
        self.assertEqual(S.solve(2*y*y-1,y),[S.sqrt(2)/2])
        # x=0 and y=s=1/sqrt(2) gives B, not a distinct neutral mutant.

    def test_symbolic_full_coefficients_match_independent_batch(self):
        q=(.83,.91)
        coords=[math.cos(q[0]/2)*math.cos(q[1]),math.cos(q[0]/2)*math.sin(q[1]),math.sin(q[0]/2)]
        for name in RESIDENTS:
            v,_,_,_,a=symbolic_coefficients(name)
            exact=[float(p.subs(dict(zip(v,coords)))) for p in a]
            np.testing.assert_allclose(exact,batch_compositions(name,[q])['a'][0,0],atol=2e-12,rtol=0)

    def test_prior_grid_has_no_robust_invader(self):
        for name in RESIDENTS:
            points=[q for q in canonical_grid(21,11) if not is_self(name,q)]
            a=batch_compositions(name,points)['a'][:,0,:]
            for row in a:
                leading=next((v for v in row if abs(v)>1e-8),0)
                self.assertLessEqual(leading,0)

    def test_boundary_cones(self):
        cases={'K2_Q':([(1,-1),(0,-1)], [(-1,0),(0,1)]),
               'K4_Q':([(1,-1),(1,0)], [(-1,-1),(1,1)]),
               'K4_B':([(-1,-1),(1,0)], [(0,1),(1,1)]),
               'K5_A':([(1,-1),(1,1)], [(-1,0),(-1,1)])}
        for name,(yes,no) in cases.items():
            for d in yes:self.assertTrue(feasible(name,d))
            for d in no:self.assertFalse(feasible(name,d))

    def test_derivatives_independent_real_finite_difference(self):
        # Analytic continuation outside rectangle only estimates derivatives;
        # feasibility is tested separately. No extrapolated mutant is admitted.
        h=1e-4
        for name,(_,t,p,_) in RESIDENTS.items():
            center=np.array([t,p]); base=batch_compositions(name,[center])['a'][0,0]
            derivative=local_derivatives(name)
            for axis in range(2):
                e=np.eye(2)[axis]*h
                minus,plus=batch_compositions(name,[center-e,center+e])['a'][:,0,:]
                gradient=(plus-minus)/(2*h);hessian=(plus-2*base+minus)/h**2
                for m, d in enumerate(derivative):
                    self.assertAlmostEqual(gradient[m],d['gradient_float'][axis],delta=2e-5)
                    self.assertAlmostEqual(hessian[m],d['hessian_float'][axis][axis],delta=2e-4)
            offsets=np.array([[h,h],[h,-h],[-h,h],[-h,-h]])
            vals=batch_compositions(name,center+offsets)['a'][:,0,:]
            cross=(vals[0]-vals[1]-vals[2]+vals[3])/(4*h*h)
            for m,d in enumerate(derivative):self.assertAlmostEqual(cross[m],d['hessian_float'][0][1],delta=2e-4)

    def test_high_precision_away_from_noise(self):
        with mp.workdps(80):
            for name in RESIDENTS:
                precise=high_precision_coefficients(name,mp.mpf('.83'),mp.mpf('.91'))
                ordinary=batch_compositions(name,[(.83,.91)])
                np.testing.assert_allclose(list(map(float,precise['a'])),ordinary['a'][0,0],atol=2e-12,rtol=0)
                self.assertLess(precise['probability_error'],mp.mpf('1e-75'))

    def test_b_tie_edges_independent_evaluation(self):
        with mp.workdps(80):
            for t,p,expected in [(mp.pi/3,mp.pi/2,mp.mpf('-0.1875')),(mp.mpf(0),mp.pi/7,mp.mpf('-.75'))]:
                a=high_precision_coefficients('K4_B',t,p)['a']
                self.assertLess(abs(a[0]),mp.mpf('1e-75'))
                self.assertLess(abs(a[1]-expected),mp.mpf('1e-75'))
                self.assertLess(a[1],0)

    def test_exact_a0_and_probability_conversion(self):
        points=np.array([[0,0],[math.pi/2,math.pi/2],[.17,.39],[math.pi,0]])
        for name in RESIDENTS:
            result=batch_compositions(name,points)
            np.testing.assert_allclose(result['a'][:,0,0],exact_a0(name,points[:,0],points[:,1]),atol=1e-12,rtol=0)
            for i in range(len(points)):
                np.testing.assert_allclose(result['a'][i,0],bernstein_to_power_coefficients(result['b'][i,0].tolist()),atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
