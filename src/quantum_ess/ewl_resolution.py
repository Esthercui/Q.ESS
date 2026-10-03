"""Step 6: independent exact and batched checks of the frozen EWL model.

No Step 1--5 implementation or numerical classification threshold is changed.
Coordinates x=cos(theta/2)cos(phi), y=cos(theta/2)sin(phi), s=sin(theta/2)
lie in the positive octant of x*x+y*y+s*s=1.
"""
from functools import lru_cache
from itertools import product
from math import comb, pi

import numpy as np

from .ewl import EWLStrategy
from .ewl_invasion import (canonicalize_ewl_strategy, ewl_strategies_equivalent,
                           DEFAULT_DIAGNOSTIC_FREQUENCIES)
from .k_player_ewl_grid import ewl_strategy_grid

RESIDENTS = {
    'K2_Q': (2, 0., pi/2, 3.),
    'K4_Q': (4, 0., pi/2, 9.),
    'K4_B': (4, pi/2, pi/2, 6.75),
    'K5_A': (5, 0., 2*pi/5, 12.),
}
TOLERANCE = 1e-8
LOCAL_RADIUS = .1


def canonical_grid(nt, np_):
    """Keep every source coordinate alias, using the frozen Step 5 rule."""
    points = {}
    for u in ewl_strategy_grid(nt, np_):
        v = canonicalize_ewl_strategy(u)
        points.setdefault((v.theta, v.phi), []).append([u.theta, u.phi])
    return points


def is_self(name, angles):
    _, t, p, _ = RESIDENTS[name]
    return ewl_strategies_equivalent(EWLStrategy(t, p), EWLStrategy(*angles))


def feasible(name, direction):
    """Tangent cone of the parameter rectangle at the specified resident."""
    dt, dp = direction
    return ((dt >= 0 and dp <= 0) if name in ('K2_Q', 'K4_Q') else
            (dp <= 0) if name == 'K4_B' else dt >= 0)


def exact_a0(name, theta, phi):
    """Stable evaluation of analytically certified a0, no payoff subtraction."""
    if name in ('K2_Q', 'K4_Q'):
        factor = 1 if name == 'K2_Q' else 3
        return -factor*(3*np.sin(theta/2)**2 + 2*np.cos(theta/2)**2*np.cos(phi)**2)
    if name == 'K4_B':
        return -2.25*np.sin(theta)*np.cos(phi)
    alpha = (9+5*np.sqrt(5))/2
    return -alpha*np.sin(theta/2)**2 - 8*np.cos(theta/2)**2*np.sin(phi-2*pi/5)**2


def tie_a1(edge, parameter):
    """Exact B tie-set objectives; equality is imposed by parametrization."""
    if edge == 'phi_pi_over_2':
        return -.75*np.cos(parameter)**2
    if edge == 'theta_0':
        return np.zeros_like(parameter) - .75
    raise ValueError(edge)


def batch_compositions(name, angles):
    """All type placements and focal positions, using closed endpoint amplitudes.

    Returns shape (mutants, focal positions, co-player count) for each type.
    This independently implements J^dagger tensor(U) J, without clipping small
    probabilities. It is checked against the unchanged scalar Step 5 engine.
    """
    k, rt, rp, _ = RESIDENTS[name]
    q = np.asarray(angles, dtype=float).reshape(-1, 2)
    size = len(q)
    def matrices(t, p):
        c, s = np.cos(t/2), np.sin(t/2)
        u = np.empty((size, 2, 2), complex)
        u[:, 0, 0] = c*np.exp(1j*p); u[:, 0, 1] = s
        u[:, 1, 0] = -s; u[:, 1, 1] = c*np.exp(-1j*p)
        return u
    mats = [matrices(rt, rp), matrices(q[:, 0], q[:, 1])]
    bits = np.array(list(product((0, 1), repeat=k)))
    outcomes = len(bits)
    phase = 1j if k % 2 else 1
    g0 = phase*(-1)**k
    gcomp = phase*(-1.)**bits.sum(axis=1)
    table = np.array([[sum(((3, 0), (5, 1))[row[f]][row[j]]
                           for j in range(k) if j != f) for f in range(k)] for row in bits])
    values = np.empty((outcomes, size, k))
    prob_error = np.zeros(size)
    for mask, types in enumerate(bits):
        p0 = np.ones((size, outcomes), complex)
        p1 = p0.copy()
        for i, kind in enumerate(types):
            p0 *= mats[kind][:, bits[:, i], 0]
            p1 *= mats[kind][:, bits[:, i], 1]
        v = p0 + 1j*g0*p1
        amp = (v - 1j*gcomp*v[:, ::-1])/2
        prob = abs(amp)**2
        prob_error = np.maximum(prob_error, abs(prob.sum(axis=1)-1))
        values[mask] = prob @ table
    ur = np.empty((size, k, k)); um = ur.copy()
    placement = np.zeros(size)
    for f in range(k):
        for j in range(k):
            for kind, dest in ((0, ur), (1, um)):
                selected = values[(bits[:, f] == kind) & (bits.sum(axis=1) == j+kind), :, f]
                dest[:, f, j] = selected.mean(axis=0)
                placement = np.maximum(placement, np.ptp(selected, axis=0))
    b = um-ur
    conversion = np.array([[comb(k-1,m)*(-1)**(m-j)*comb(m,j) if j <= m else 0
                            for j in range(k)] for m in range(k)])
    a = b @ conversion.T
    spread = np.maximum(np.ptp(ur, axis=1).max(axis=1), np.ptp(um, axis=1).max(axis=1))
    reconstruction = np.zeros(size)
    for e in (0., *DEFAULT_DIAGNOSTIC_FREQUENCIES):
        w = np.array([comb(k-1,j)*e**j*(1-e)**(k-1-j) for j in range(k)])
        direct = um @ w - ur @ w
        power = a @ np.array([e**j for j in range(k)])
        reconstruction = np.maximum(reconstruction, abs(direct-power).max(axis=1))
    return dict(u_R=ur, u_M=um, b=b, a=a, probability_error=prob_error,
                placement_spread=placement, focal_spread=spread,
                polynomial_error=reconstruction)


@lru_cache(None)
def symbolic_coefficients(name):
    """Exact polynomial composition payoffs from endpoint state amplitudes."""
    import sympy as S
    x, y, s = S.symbols('x y s', real=True)
    M = S.Matrix([[x+S.I*y, s], [-s, x-S.I*y]])
    k = RESIDENTS[name][0]
    if name == 'K4_B':
        R = S.Matrix([[S.I, 1], [-1, -S.I]])/S.sqrt(2)
    elif name == 'K5_A':
        p = 2*S.pi/5
        R = S.diag(S.cos(p)+S.I*S.sin(p), S.cos(p)-S.I*S.sin(p))
    else:
        R = S.diag(S.I, -S.I)
    phase = S.I if k % 2 else 1
    g0 = phase*(-1)**k
    def payoff(profile):
        out = 0
        for bits in product((0, 1), repeat=k):
            def endpoint(bs, col):
                return S.prod(u[b,col] for u,b in zip(profile,bs))
            reverse = tuple(1-b for b in bits)
            v = endpoint(bits,0)+S.I*g0*endpoint(bits,1)
            vc = endpoint(reverse,0)+S.I*g0*endpoint(reverse,1)
            amp = S.expand((v-S.I*phase*(-1)**sum(bits)*vc)/2)
            prob = S.expand(S.re(amp)**2+S.im(amp)**2)
            payoff_value = sum(((3,0),(5,1))[bits[0]][b] for b in bits[1:])
            out += prob*payoff_value
        return S.expand(out)
    ur, um = [], []
    for j in range(k):
        co = [M]*j+[R]*(k-1-j)
        ur.append(payoff([R]+co)); um.append(payoff([M]+co))
    b = [S.expand(v-u) for u,v in zip(ur,um)]
    a = [S.expand(comb(k-1,m)*sum((-1)**(m-j)*comb(m,j)*b[j] for j in range(m+1))) for m in range(k)]
    return (x,y,s), ur, um, b, a


def symbolic_certificate():
    """Polynomial-ideal identities plus explicit sign/zero-set arguments.

    The normalization relation is the only reduction. Exact expressions, not
    floating tolerance, decide every certificate assertion.
    """
    import sympy as S
    certificates = {}
    for name in RESIDENTS:
        (x,y,s), ur, um, b, a = symbolic_coefficients(name)
        alpha = (9+5*S.sqrt(5))/2
        target = (-3*s*s-2*x*x if name == 'K2_Q' else
                  -9*s*s-6*x*x if name == 'K4_Q' else
                  -S.Rational(9,2)*s*x if name == 'K4_B' else
                  -alpha*s*s-8*(x*S.sin(2*S.pi/5)-y*S.cos(2*S.pi/5))**2)
        remainder = S.simplify(S.reduced(S.expand(a[0]-target), [s*s+x*x+y*y-1], s,x,y)[1])
        assert remainder == 0, (name, remainder)
        checks = {'a0_identity_remainder': str(remainder)}
        if name == 'K4_B':
            checks['a1_x0_remainder'] = str(S.factor(S.reduced(S.expand(a[1].subs(x,0)+S.Rational(3,4)*(2*y*y-1)**2), [s*s+y*y-1], s,y)[1]))
            checks['a1_s0_remainder'] = str(S.factor(S.reduced(S.expand(a[1].subs(s,0)+S.Rational(3,4)), [x*x+y*y-1], x,y)[1]))
            assert set(checks.values()) == {'0'}
        certificates[name] = dict(a0_cartesian=str(target), exact_checks=checks,
            status='proven_ESS_restricted_EWL',
            scope='pointwise rare-mutant ESS for every fixed distinct pure mutant; restricted family only',
            method='exact polynomial identities and sign/zero-set argument over positive unit octant')
    return certificates


def local_derivatives(name):
    """Exact chain rule for every a_m; no complex-step/holomorphic assumption."""
    import sympy as S
    variables, _, _, _, a = symbolic_coefficients(name)
    t, p = S.symbols('theta phi', real=True)
    v = S.Matrix([S.cos(t/2)*S.cos(p), S.cos(t/2)*S.sin(p), S.sin(t/2)])
    rt = S.pi/2 if name == 'K4_B' else 0
    rp = 2*S.pi/5 if name == 'K5_A' else S.pi/2
    at = {t:rt, p:rp}
    coords = {x:S.simplify(q.subs(at)) for x,q in zip(variables,v)}
    J = v.jacobian([t,p]).subs(at)
    result = []
    for m, poly in enumerate(a):
        g = S.Matrix([S.diff(poly,x) for x in variables]).subs(coords).applyfunc(S.simplify)
        H = S.hessian(poly,variables).subs(coords).applyfunc(S.simplify)
        grad = (J.T*g).applyfunc(S.simplify)
        hess = (J.T*H*J + sum((g[i]*S.hessian(v[i],[t,p]).subs(at) for i in range(3)), S.zeros(2))).applyfunc(S.simplify)
        result.append(dict(order=m, gradient=[str(q) for q in grad],
                           hessian=[[str(hess[i,j]) for j in range(2)] for i in range(2)],
                           gradient_float=[float(q) for q in grad],
                           hessian_float=np.array(hess).astype(float).tolist(),
                           eigenvalues=[str(q) for q in hess.eigenvals()]))
    return result


def high_precision_coefficients(name, theta, phi, dps=80):
    """Independent dense tensor-matrix evaluation at arbitrary precision.

    theta/phi can be mp numbers (do not downcast exact-angle expressions).
    Co-player placement invariance follows from the symmetric tensor circuit;
    the separate batch audit enumerates every placement and focal position.
    """
    import mpmath as mp
    with mp.workdps(dps):
        k = RESIDENTS[name][0]
        rt = mp.pi/2 if name == 'K4_B' else mp.mpf(0)
        rp = 2*mp.pi/5 if name == 'K5_A' else mp.pi/2
        def kron(a,b):
            return mp.matrix([[a[i//b.rows,j//b.cols]*b[i%b.rows,j%b.cols]
                               for j in range(a.cols*b.cols)] for i in range(a.rows*b.rows)])
        def tensor(ms):
            out = mp.matrix([[1]])
            for m in ms: out = kron(out,m)
            return out
        def unitary(t,p):
            c,s = mp.cos(t/2),mp.sin(t/2)
            return mp.matrix([[mp.exp(1j*p)*c,s],[-s,mp.exp(-1j*p)*c]])
        R,M = unitary(rt,rp),unitary(theta,phi)
        D = mp.matrix([[0,1],[-1,0]])
        G = tensor([D]*k)*(1j if k%2 else 1)
        J = (mp.eye(2**k)+1j*G)/mp.sqrt(2)
        initial = J*mp.matrix([1]+[0]*(2**k-1))
        norm_error = mp.mpf(0)
        def payoff(profile):
            nonlocal norm_error
            state = J.H*tensor(profile)*initial
            probs = [abs(z)**2 for z in state]
            norm_error = max(norm_error,abs(mp.fsum(probs)-1))
            return mp.fsum(probs[z]*sum(((3,0),(5,1))[(z>>(k-1))&1][(z>>(k-1-j))&1]
                                       for j in range(1,k)) for z in range(2**k))
        ur,um = [],[]
        for j in range(k):
            co = [M]*j+[R]*(k-1-j)
            ur.append(payoff([R]+co));um.append(payoff([M]+co))
        b = [v-u for u,v in zip(ur,um)]
        a = [comb(k-1,m)*mp.fsum((-1)**(m-j)*comb(m,j)*b[j] for j in range(m+1)) for m in range(k)]
        return dict(u_R=ur,u_M=um,b=b,a=a,probability_error=norm_error)
