# Step 6: restricted-EWL ESS resolution and certification

## Frozen problem and provenance

Steps 1-4 are frozen at `5827d9aaf631f67a02713eb8643609a7730707d2`.
Step 5 was validated at `b174ec298187655e8d2eb143fe5509d7fcbc60d8`, then squash
merged by PR #5 as `655c86574b6c989f25153c7ad500b5b15d6e8316`. The merged tree is
byte-identical to the validated Step 5 tree. This branch starts at that merge.
No Step 1-5 source, tests, saved classifications, or original artifacts change.

The model is the project's parity-adjusted global K-body EWL extension, at
`gamma=pi/2`, with the restricted strategy

\[
U(\theta,\phi)=\begin{pmatrix}e^{i\phi}\cos(\theta/2)&\sin(\theta/2)\\
-\sin(\theta/2)&e^{-i\phi}\cos(\theta/2)\end{pmatrix},\quad
0\leq\theta\leq\pi,\quad0\leq\phi\leq\pi/2.
\]

Outcome payoffs sum pairwise PD `(R,S,T,P)=(3,0,5,1)` over the other `K-1`
players. The population is infinite, well mixed, and monomorphic except for a
single pure mutant type at frequency epsilon. Co-players are independently
sampled. No parameter, strategy access, population assumption, or Step 5
classification tolerance (`1e-8`) changes.

For each fixed physically distinct mutant the required condition is a negative
first nonzero coefficient of the invasion polynomial. This is the question's
pointwise rare-mutant definition: the sufficiently small frequency may depend
on the mutant. We do not claim a common invasion barrier, stability against
arbitrary mixtures of mutant types, or a dynamics theorem.

## Population polynomial and K=2 check

Set `n=K-1`, and let `u_X(j)` be the payoff to a focal X with j mutant
co-players. With `b_j=u_M(j)-u_R(j)`, the polynomial is

\[
\Delta\Pi(\epsilon)=\sum_{j=0}^{n}\binom nj b_j\epsilon^j(1-\epsilon)^{n-j}
 =\sum_{m=0}^{n}a_m\epsilon^m,\qquad
 a_m=\binom nm\sum_{j=0}^{m}(-1)^{m-j}\binom mj b_j.
\]

For K=2, `a0=u(M,R)-u(R,R)` and
`a1=u(M,M)-u(R,M)-a0`. **Only on the exact tie `a0=0`** does this reduce to
`a1=u(M,M)-u(R,M)`. Thus the ordered-coefficient test is exactly the two-player
Maynard-Smith condition. Omitting `-a0` away from the tie would be incorrect.
Regression tests check both cases.

## Exact algebraic certificate

Define nonnegative Cartesian coordinates

\[
x=\cos(\theta/2)\cos\phi,\quad y=\cos(\theta/2)\sin\phi,\quad
s=\sin(\theta/2),\qquad x^2+y^2+s^2=1.
\]

The strategy matrix is `[[x+i y,s],[-s,x-i y]]`. This maps the positive unit
octant onto the restricted family; the single point `s=1` is the physically
redundant theta=pi edge, canonicalized to `(pi,0)` by the Step 5 rule.

Let `h=1` for even K and `h=i` for odd K, and let `D=[[0,1],[-1,0]]`.
Then `G=h D^tensorK`, `G|z>=h(-1)^(K-|z|)|complement(z)>`, and
`J=(I+iG)/sqrt(2)`. For a profile define
`P_l(z)=product_i U_i[z_i,l]`, `v(z)=P_0(z)+i h(-1)^K P_1(z)`.
The final amplitude is exactly

\[
 A(z)=\tfrac12\{v(z)-i h(-1)^{|z|}v(\bar z)\}.
\]

Squaring its real and imaginary parts, multiplying by the explicit pairwise
payoff, and summing over outcomes yields exact polynomials for every `u_R(j)`,
`u_M(j)`, `b_j`, and `a_m`. The tensor circuit, initial state, and payoff rule
commute with a common player permutation. Therefore a representative type
placement suffices for the symbolic derivation. The numerical audit separately
enumerates *all* type placements and *all* focal positions.

`scripts/run_step6.py` saves the full symbolic polynomials, including radical
coefficients for K=5. `symbolic_certificate()` verifies the following identities
by polynomial division by `s^2+x^2+y^2-1`; all remainders simplify **exactly to
zero**. These are algebraic equalities, not fitted formulas or tolerance tests.

| Resident | Exact a0 | Exact a0=0 set |
|---|---|---|
| K=2 Q | `-3s^2-2x^2` | self only |
| K=4 Q | `-9s^2-6x^2` | self only |
| K=5 A=(0,2pi/5) | `-alpha s^2-8(x sin r-y cos r)^2`, `alpha=(9+5sqrt(5))/2`, `r=2pi/5` | self only |
| K=4 B | `-(9/2)s x=-(9/4)sin(theta)cos(phi)` | `s=0` union `x=0` |

The first three expressions are strictly negative for every non-self mutant.
For K=5, equality forces `s=0` and `sin(phi-r)=0`; on the specified phase
interval the only solution is `phi=r`. For Q equality forces `s=x=0,y=1`.
This proves both local and global exclusion without a floating-point search.

For B, the complete tie set is the theta=0 edge plus the phi=pi/2 edge, with
the theta=pi edge represented by the canonical D point. Exact reductions give

\[
 a_1|_{s=0}=-\tfrac34,\qquad
 a_1|_{x=0}=-\tfrac34(2y^2-1)^2=-\tfrac34\cos^2\theta.
\]

The latter can vanish only when `y=s=1/sqrt(2)`, which is B itself. Every
non-self mutant on the tie set is therefore rejected at order 1; off the set
it is rejected at order 0. There is **no distinct exact neutral mutant** for
any of the four residents. In particular, a unilateral tie manifold is not
an identically neutral invasion manifold. The only recursive Z1 for B is self;
a2 and a3 need not decide any distinct mutant.

For any fixed mutant, a finite polynomial with a negative leading coefficient
is negative at all sufficiently small positive epsilon. The identities and
zero-set arguments cover the entire admissible domain. Consequently all four
residents receive `proven_ESS_restricted_EWL` for the stated definition. This is
an analytic proof supported by exact symbolic checks, not a claim of formal
proof-assistant verification or a proof inferred from optimizer success.

## Feasible local derivatives and complement

All derivatives are of individual coefficient functions, not just DeltaPi at
0.01. Exact chain-rule gradients and Hessians of *every* a_m are saved. The
independent check uses real centered finite differences of an 80-digit dense
tensor-matrix calculation, with step `1e-12`. Evaluations outside the rectangle
are used only for the smooth analytic derivative extension, never as admitted
mutants. Complex-step differentiation is not used.

| Resident | Feasible local directions | grad a0 | Hessian a0 |
|---|---|---|---|
| K=2 Q | dtheta>=0, dphi<=0 | (0,0) | diag(-3/2,-4) |
| K=4 Q | dtheta>=0, dphi<=0 | (0,0) | diag(-9/2,-12) |
| K=5 A | dtheta>=0, dphi free | (0,0) | diag(-(9+5sqrt(5))/4,-16) |
| K=4 B | dtheta free, dphi<=0 | (0,9/4) | zero matrix |

For the three isolated a0 zeros write `alpha=3,9,(9+5sqrt(5))/2` and
`beta=2,6,8` respectively. With resident phase p_R, the exact formula is
`-alpha sin^2(dtheta/2)-beta cos^2(dtheta/2) sin^2(dphi)`. Its local expansion is

\[
 -\alpha d\theta^2/4-\beta d\phi^2
 +\alpha d\theta^4/48+\beta d\theta^2d\phi^2/4
 +\beta d\phi^4/3+O(\|d\|^6).
\]

The a0 Hessians are negative definite; their eigendirections are the coordinate
axes. For B, `a0=(9/4)cos(dtheta)sin(dphi)`. It decreases linearly into the
feasible interior; its zero Hessian alone would be inconclusive. Along the
boundary tangent `dphi=0`, a0 vanishes identically and
`a1=-(3/4)sin^2(dtheta)=-(3/4)dtheta^2+(1/4)dtheta^4+O(dtheta^6)`.
The a0 expansion is `(9/4)dphi-(9/8)dtheta^2 dphi-(3/8)dphi^3+O(||d||^5)`.
There is no unhandled feasible null direction.

Use local radius `r0=0.1` in Euclidean angle coordinates. The exact identities
control both the local ball and its complement. For the three isolated zeros,
a concrete global-complement upper bound on a0 is `-min(A,B)<0`, where
`A=alpha sin^2(r0/(2sqrt(2)))` and
`B=beta cos^2(r0/(2sqrt(2))) sin^2(r0/sqrt(2))`. Indeed either theta is at least
`r0/sqrt(2)`, or the phase distance is; phase distances never exceed pi/2.
For B no strictly negative a0 margin exists on the complement because its tie
edges remain. On the phi=pi/2 tie edge outside the ball,
`a1<=-(3/4)sin^2(r0)<0`; on theta=0 it is `-3/4`. Off both edges a0<0.
Thus the complement is covered lexicographically. Interval branch-and-bound is
unnecessary here because the exact global identities already give the sign.

## Numerical convergence, optimization, and precision

The grids `(21,11),(41,21),(81,41)` contain 221, 841, 3281 physical operations;
self exclusion leaves 220, 840, 3280 per resident. Source-coordinate aliases
are preserved, including every redundant coordinate on theta=pi. Each of the
17,360 rows contains all composition payoffs, Bernstein and power coefficients,
all focal-position composition arrays, leading coefficient at the unchanged
`1e-8` threshold, diagnostics at 0 and the eight Step 5 frequencies, and
probability, placement, focal, and polynomial residuals. The floating threshold
labels are computational diagnostics and never determine the exact certificate.

The NumPy batch evaluator uses the exact endpoint-amplitude formula without
clipping probabilities. The frozen Step 5 engine clips probabilities below
`1e-15`; it is not modified. Independent regression comparisons and high-
precision calculations quantify the difference, especially near self.

DE maximizes a0 on the full rectangle and the distance>=0.1 complement from
three seeds `20261002,20261003,20261004`. Each rectangle edge is also searched
explicitly. Eight L-BFGS-B starts per resident include the four corners and four
seeded interior points. For B, stage 1 maximizes a1 on *exact parametrizations*
of both Z0 edges, then on both portions of the phi=pi/2 edge outside the local
ball. The theta=0 edge is entirely outside that ball. No loose equality
constraint substitutes for the tie set. Stage 0 searches may return self,
which is flagged and is not treated as a non-self mutant. No later stage is
needed for the other three residents because their Z0 consists only of self.

Every run saves bounds, seed, configuration, start (where applicable), status,
message, iterations, evaluations, objective, constraint violation, runtime,
coordinates, and full ordinary-precision coefficients. The configuration is
DE maxiter=500, popsize=15, tol=1e-10, atol=1e-12, polish=False, one worker,
immediate updating. L-BFGS-B uses ftol=1e-14, gtol=1e-10, maxiter=1000.
One local refinement reports an abnormal line-search stop near self; retain it
as a failure, not convergence. All global searches converge. The exact proof
is independent of either numerical outcome.

Dense 80-digit tensor products independently recompute ordinary points, theta
and phi displacements down to `1e-20`, tie-edge witnesses, the finest-grid best
mutant, and the strongest observed double-precision optimizer candidate.
`1e-60` is only a high-precision diagnostic threshold. Exact boundaries are
constructed with mp.pi, rather than converting a binary64 approximation of pi.
Probes that collapse to self in ordinary coordinates are explicitly flagged.
The exact identities, not numerical tiny coefficients, establish equality.

There is no strongest distinct mutant in the continuous domain under the
reference-frequency ordering near these residents: differences approach zero
as mutants approach self. The summary therefore reports the strongest *finite
81x41-grid* mutant, and precision tables separately show near-self optimizer
candidates, including spurious floating zeros/positive residuals.

## Literature and future scope

The project literature reference is A. Iqbal and A.H. Toor, *Evolutionarily
Stable Strategies in Quantum Games*, Physics Letters A 280 (5-6), 249-256
(2001), [arXiv:quant-ph/0007100v3](https://arxiv.org/abs/quant-ph/0007100v3),
[DOI 10.1016/S0375-9601(01)00082-2](https://doi.org/10.1016/S0375-9601(01)00082-2).
The stored September review identifies this exact paper. Its symmetric
Prisoner's Dilemma discussion distinguishes available strategy families and
uses the two-player ESS test. The K=2 result here is derived independently,
not presented as a new discovery. Its multiplayer conclusions cannot be
imported into this project's different K-body construction.

The related project reference is S.C. Benjamin and P.M. Hayden, *Comment on
"Quantum Games and Quantum Strategies"*, Physical Review Letters 87, 069801
(2001), [arXiv:quant-ph/0003036](https://arxiv.org/abs/quant-ph/0003036).
The admissible quantum strategy set can change equilibrium and stability
claims. This motivates Step 7; none of this pass's proofs automatically covers
full SU(2). A paper-specific conclusion transfers only after matching payoff,
entangler, game, strategy domain, and evolutionary definition.

Steps 7-10 are documented only: full-SU(2) unilateral Nash screen first, then
population ESS for its survivors; entanglement sweeps with new Nash
identification at each gamma; explicit noise models and stability thresholds;
then replicator dynamics. Polymorphic/asymmetric residents remain a later
project. No such experiment was run in Step 6.

## Reproduction and artifact contracts

From the repository root, use `studio-python` on this host. Elsewhere create an
isolated Python environment, install `requirements-step6.txt`, and substitute
that environment's Python. Optional SymPy and mpmath are the only additions to
the research dependency group: they provide exact identities and independent
arbitrary precision. Exact runtime versions, hardware, source hashes, seeds,
workers, elapsed time, and the experiment commit are recorded in run metadata.

```sh
PYTHONPATH=src studio-python -m unittest discover -s tests
studio-python scripts/validate_step5_results.py
studio-python scripts/validate_step5_results.py --results-dir results/step5_macmini_20260828
PYTHONPATH=src studio-python scripts/audit_step5_for_step6.py
PYTHONPATH=src studio-python scripts/run_step6.py --output-dir results/step6
PYTHONPATH=src studio-python scripts/validate_step6_results.py
PYTHONPATH=src studio-python scripts/build_step6_report.py
PYTHONPATH=src studio-python scripts/build_step6_notebook.py
```

The Mac Mini repeat is included as preserved baseline evidence in the delivery
package; its existing local folder is not overwritten. Run the experiment
before artifact-dependent tests when recreating artifacts from scratch.
Results include `run_metadata.json`, `resident_summary.csv`,
`grid_convergence.csv`, `mutant_coefficients.csv.gz`,
`coefficient_optimizer_runs.csv`, `local_derivative_analysis.json`,
`neutral_sets.json`, `precision_validation.csv`, `symbolic_coefficients.json`,
`certification_summary.json`, `baseline_audit.json`, `data_manifest.json`, and
plots. `data_manifest.json` hashes the numerical files; report and notebook
products are derived from those data. The saved artifact validator checks both
file hashes and the generating source hashes, reconstruction, schema, and
invariance. Deliberate code changes require regenerating affected data.
