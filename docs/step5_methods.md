# Step 5 Resident-Mutant Invasion Method

## Scope and authority

This analysis uses GitHub main commit
`5827d9aaf631f67a02713eb8643609a7730707d2` as the authoritative Step 1-4
baseline. The older donation-game/noise implementation is not imported into
this baseline. It remains separate for a later robustness analysis.

The present search covers pure strategies in the restricted two-parameter EWL
family for `K=2,3,4,5`, initially at `gamma=pi/2`. Mixed population strategies,
general `SU(2)` operations, noise, finite shots, and finite populations are not
part of Step 5.

## Audited quantum game

### Local strategy

The exact implemented operation is

```math
U(\theta,\phi)=
\begin{pmatrix}
e^{i\phi}\cos(\theta/2) & \sin(\theta/2)\\
-\sin(\theta/2) & e^{-i\phi}\cos(\theta/2)
\end{pmatrix},
```

with

```math
0\leq\theta\leq\pi,\qquad 0\leq\phi\leq\pi/2.
```

The embedded classical operations are `C=U(0,0)` and `D=U(pi,0)`, while
`Q=U(0,pi/2)`. At `theta=pi`, both diagonal entries vanish, so `phi` has no
effect. All coordinates `(pi,phi)` are therefore canonicalized to `(pi,0)`.
This is a coordinate duplication, not a payoff-based approximation. The two
`phi` edges at `0` and `pi/2` are retained as distinct boundaries of this
restricted strategy set.

### Entangler and execution order

Let `D=U(pi,0)` and define the Hermitian involution

```math
G_K=
\begin{cases}
D^{\otimes K}, & K\text{ even},\\
iD^{\otimes K}, & K\text{ odd}.
\end{cases}
```

The implemented entangler is

```math
J_K(\gamma)=\cos(\gamma/2)I+i\sin(\gamma/2)G_K.
```

The final state is evaluated in this order:

```math
|\psi_f\rangle=
J_K(\gamma)^\dagger
\left(\bigotimes_{i=1}^{K}U_i\right)
J_K(\gamma)|0\ldots0\rangle.
```

In code, the local operations are applied one player at a time after `J_K`;
because they act on different qubits, this is equivalent to their tensor
product. This entangler is **this project's global parity-adjusted K-body EWL
extension**. It must not be described as the uniquely standard multiplayer EWL
construction. The original EWL paper defines the two-player framework, while
other multiplayer quantizations exist in the literature.

For each bitstring `z`, the engine uses exact statevector probabilities

```math
p(z)=|\langle z|\psi_f\rangle|^2.
```

There is no shot sampling, seed, or confidence interval in the payoff engine.
The reported uncertainty field therefore records normalization, polynomial,
and permutation residuals rather than sampling uncertainty.

### Payoff rule

Outcome bit `0` is cooperation-like and bit `1` is defection-like. Each player
plays the Prisoner's Dilemma against every other player with

```math
(R,S,T,P)=(3,0,5,1).
```

For outcome `z`, player `i` receives the pairwise sum

```math
g_i(z)=\sum_{\ell\neq i}g(z_i,z_\ell),
```

where `g(0,0)=3`, `g(0,1)=0`, `g(1,0)=5`, and `g(1,1)=1`. The expected payoff is

```math
u_i=\sum_{z\in\{0,1\}^K}p(z)g_i(z).
```

The circuit, measurement rule, and payoff tensor are invariant under a common
permutation of players. Step 5 nevertheless evaluates every focal position and
reports the maximum positional discrepancy rather than assuming symmetry.

### Reuse and audit findings

Step 5 reuses `EWLStrategy.matrix`, `KPlayerEWLGame.run`,
`KPlayerEWLGame.expected_payoffs`, `k_player_ewl_pd_payoff`, and
`ewl_strategy_grid` without changing them. The population, search, and reporting
logic lives in new modules, and the Step 1-4 regression suite remains intact.

The audit identified three distinctions that matter for interpretation:

- The original Step 4 `11 x 6` grid contains 66 coordinate rows but only 61
  distinct operations after the five redundant `theta=pi` phase coordinates
  are collapsed. Step 4 output is preserved; Step 5 canonicalizes before search.
- The two-player EWL paper does not specify this project's odd-`K`
  parity-adjusted global entangler. Results must be attributed to the explicit
  construction above, not to a unique multiplayer standard.
- The older donation-game/noise implementation uses a different payoff and
  physical model. Combining its assumptions with the exact pairwise-PD baseline
  would invalidate the comparison, so it is deliberately excluded here.

## Population model

The population is infinite and well mixed. Residents use `R`; mutants use `M`
at population frequency `epsilon`. Groups of `K` are formed randomly and each
of a focal player's `K-1` co-players is sampled independently. Thus the number
`j` of mutant co-players is binomial:

```math
\Pr(j)=\binom{K-1}{j}\epsilon^j(1-\epsilon)^{K-1-j}.
```

Let `u_X(j;R,M)` be the expected EWL payoff of a focal type `X` in `{R,M}`
when exactly `j` co-players use `M`. The implementation averages over all
`C(K-1,j)` placements of those co-player types and records their spread as a
permutation audit. Population fitness is

```math
\Pi_R(\epsilon)=\sum_{j=0}^{K-1}
\binom{K-1}{j}\epsilon^j(1-\epsilon)^{K-1-j}u_R(j;R,M),
```

```math
\Pi_M(\epsilon)=\sum_{j=0}^{K-1}
\binom{K-1}{j}\epsilon^j(1-\epsilon)^{K-1-j}u_M(j;R,M),
```

and

```math
\Delta\Pi(R,M;\epsilon)=\Pi_M(\epsilon)-\Pi_R(\epsilon).
```

This differs from putting exactly one mutant into one fixed `K`-player group.
That fixed-group experiment compares the mutant with residents conditioned on
the same mixed group. Population ESS compares the expected fitness of both
types under population-wide random grouping. The code exposes the fixed-group
quantity only as `exactly_one_mutant_group_diagnostic`.

For example, with `K=5`, `D` in one otherwise-`C` group earns `20` while each
group resident earns `9`, a fixed-group difference of `11`. In the rare
population limit, however, a resident focal individual almost always meets
four residents and earns `12`; the mutant focal individual meets four residents
and earns `20`. The leading population invasion advantage is therefore `8`.

## Rare-limit polynomial

Set

```math
b_j=u_M(j;R,M)-u_R(j;R,M).
```

Then `DeltaPi` is a Bernstein polynomial of degree at most `n=K-1`:

```math
\Delta\Pi(\epsilon)=\sum_{j=0}^{n}
b_j\binom{n}{j}\epsilon^j(1-\epsilon)^{n-j}.
```

The exact basis conversion used by the code is

```math
a_m=\binom{n}{m}\sum_{j=0}^{m}
(-1)^{m-j}\binom{m}{j}b_j,
\qquad
\Delta\Pi(\epsilon)=\sum_{m=0}^{n}a_m\epsilon^m.
```

With tolerance `tau=1e-8`, the first coefficient `a_m` satisfying
`|a_m|>tau` determines the sufficiently rare-mutant sign:

- positive: `rare_mutant_invades`;
- negative: `rare_mutant_rejected`;
- no coefficient above tolerance: `neutral`;
- numerical reliability residual above tolerance: `indeterminate`.

`DeltaPi` is also reported at `epsilon_ref=0.01` and at
`10^-8,10^-7,...,10^-1`. These are diagnostics; they do not replace the
leading-coefficient classification.

### K=2 reduction

For `K=2`,

```math
a_0=u(M,R)-u(R,R).
```

If `a_0=0`, then

```math
a_1=u(M,M)-u(R,M).
```

Consequently, rejection requires either
`u(R,R)>u(M,R)`, or equality followed by `u(R,M)>u(M,M)`. These are the
standard two-player ESS conditions, and both branches are covered by tests.

## Terminology

For this project, a resident is evolutionarily stable against a particular
non-self mutant if there exists `epsilon_bar>0` such that
`DeltaPi(R,M;epsilon)<0` for every `0<epsilon<epsilon_bar`. An ESS must satisfy
that condition against every admissible mutant. A strict computational result
would reject every non-self mutant. A neutral pair has an identically zero
invasion polynomial within declared tolerance.

The phrase "weakly stable" has several meanings in the literature and is not
used as a formal theorem here. `neutral_or_weak_candidate` is only a
conservative computational status: no positive invader was found, but at least
one distinct tested mutant had every coefficient within tolerance.

The reporting statuses are:

- `not_ESS`: at least one robust rare-limit invader was found;
- `grid_ESS_candidate`: no invader was found on the current finite grid;
- `numerically_supported_ESS`: no invader or neutral mutant was found after
  refinement and continuous search, subject to tolerance;
- `neutral_or_weak_candidate`: no positive invader was found, but numerical
  neutrality prevents a strict ESS claim;
- `indeterminate`: numerical reliability or optimizer convergence is
  insufficient.

## Search design

Residents and mutants use the same continuous domain. Grid resolution is a
numerical search choice, not a restriction on one population type.

1. **Screening.** Symmetric pure Nash candidates are found on the validated
   `11 x 6` grid. Canonicalizing the duplicated `theta=pi` edge reduces the 66
   coordinates to 61 distinct strategies. A monomorphic resident population
   corresponds to a symmetric profile, so asymmetric Step 4 Nash profiles are
   not resident candidates in this step.
2. **Coarse mutant search.** Every candidate is tested against every non-self
   canonical strategy on that full-domain grid.
3. **Fine mutant search.** Apparent survivors are tested on a `21 x 11`
   full-domain grid augmented around the resident, the strongest coarse mutant,
   and all four boundaries.
4. **Continuous search.** Bounded differential evolution is run at
   `epsilon=0,10^-6,10^-4,0.01`, followed by multistart L-BFGS-B refinement.
   Each resulting candidate is reclassified from its full rare-limit
   polynomial. Seeds, bounds, objective frequencies, convergence flags, and
   evaluation counts are saved.
5. **Position audit.** All `K` focal qubit positions are evaluated. The
   worst-case mutant advantage and maximum spread are reported.

The continuous scalar objectives search several small frequencies rather than
symbolically maximizing a lexicographic vector of polynomial coefficients.
This is a practical global search, not an analytical proof that every
higher-order neutral manifold has been exhausted.

## Baseline results

The reproducible run used `gamma=pi/2`, `epsilon_ref=0.01`, tolerance `1e-8`,
eight grid workers, eight differential-evolution workers, and seed `20260826`.
All 16 global optimizer runs reported convergence.

| K | Resident `(theta,phi)` | Monomorphic payoff | Final status | Maximum `DeltaPi(0.01)` | Interpretation |
|---:|---|---:|---|---:|---|
| 2 | `(0,pi/2)` | 3.00 | `neutral_or_weak_candidate` | `-1.36e-11` | No positive invader found; near-resident numerical neutrality remains. |
| 3 | `(pi/2,pi/2)` | 4.50 | `not_ESS` | `2.33156e-2` | Mutant `(0,3pi/10)` invades at leading order 1 with coefficient `2.33156`. |
| 4 | `(0,pi/2)` | 9.00 | `neutral_or_weak_candidate` | `-3.40e-12` | No positive invader found; near-resident numerical neutrality remains. |
| 4 | `(pi/2,pi/2)` | 6.75 | `neutral_or_weak_candidate` | `2.66e-15` | No robust positive invader found; neutral candidates remain within tolerance. |
| 5 | `(0,2pi/5)` | 12.00 | `neutral_or_weak_candidate` | `3.55e-15` | Best-payoff candidate survives the numerical search, without a strict proof. |
| 5 | `(pi/2,pi/2)` | 9.00 | `not_ESS` | `2.41211e-6` | Mutant `(pi/2,pi/5)` invades first at order 3, coefficient `2.42597`. |

The `K=3` and second `K=5` rows demonstrate why Nash equilibrium and ESS cannot
be equated. A unilateral mutant is tied when rare at order zero, yet mutant
fitness becomes positive at a higher order as mutants sometimes meet one
another.

The maximum saved residuals were approximately `3.33e-15` for probability
normalization, `3.55e-15` for polynomial reconstruction, and `2.49e-14` for
permutation invariance, all below `1e-8`.

## Reproducibility and outputs

Install the optional analysis dependencies and run:

```bash
python -m pip install -e '.[research]'
python examples/step5_ess_analysis.py --workers 8 --continuous-workers 8
python scripts/validate_step5_results.py
python scripts/build_step5_notebook.py
```

The complete position-resolved pair table is compressed separately from the
concise resident summary. `run_metadata.json` records the authoritative
baseline, command, interpreter, platform, grids, optimizer settings, and seed.

## Limitations

- The results are numerical evidence in a restricted two-parameter pure
  strategy family, not an analytical ESS proof and not a result over full
  `SU(2)` or mixed strategies.
- A finite grid can miss narrow invading regions. The continuous optimizer
  reduces this risk but cannot prove global nonexistence of an invader.
- Smooth payoffs allow distinct strategies arbitrarily close to a resident to
  have differences below any fixed tolerance. Such cases are conservatively
  reported as `neutral_or_weak_candidate`; they are not evidence of exact
  analytical neutrality.
- The baseline is exact and noiseless. There is no finite-shot uncertainty.
  Noise and finite sampling belong to the later robustness step.
- The project-specific global K-body entangler is one multiplayer extension.
  Conclusions are conditional on that topology and parity convention.
- Only monomorphic symmetric resident populations are screened. Asymmetric
  Nash profiles are not population strategies in this one-resident model.

## Paper-ready Methods subsection

### Resident-mutant invasion analysis

We analyzed invasion in an infinite well-mixed population playing random
`K`-player groups (`K=2,3,4,5`). Each pure strategy was an EWL operation
`U(theta,phi)` on the restricted domain `[0,pi] x [0,pi/2]`; the duplicated
`theta=pi` boundary was represented once because its operation is independent
of `phi`. For a focal type `X` and `j` mutant co-players, we calculated the exact
statevector EWL payoff `u_X(j;R,M)` for every placement of those co-players.
Population fitness was the binomial average over `j`, and mutant advantage was
`DeltaPi=Pi_M-Pi_R`. Because this difference is a polynomial of degree at most
`K-1`, rare-mutant behavior was classified by the sign of its first power-basis
coefficient exceeding `10^-8`, rather than by a single chosen mutation
frequency. We additionally reported payoffs at `epsilon=0.01` and over
log-spaced small frequencies. Symmetric Nash residents were screened on a
canonical `11 x 6` grid, apparent survivors were tested on an augmented
`21 x 11` grid, and continuous mutant advantage was searched by bounded
differential evolution with multistart L-BFGS-B refinement. Every candidate was
evaluated at all focal qubit positions; probability normalization, polynomial
reconstruction, optimizer convergence, and permutation invariance were saved
with the results. Finite-grid survivors were termed grid-ESS candidates, and no
analytical ESS claim was made from numerical search alone.

## Primary references

- J. Maynard Smith and G. R. Price, ["The Logic of Animal Conflict"](https://doi.org/10.1038/246015a0), *Nature* 246, 15-18 (1973).
- M. Broom, C. Cannings, and G. T. Vickers, ["Multi-player matrix games"](https://doi.org/10.1016/S0092-8240(97)00041-4), *Bulletin of Mathematical Biology* 59, 931-952 (1997).
- J. Eisert, M. Wilkens, and M. Lewenstein, ["Quantum Games and Quantum Strategies"](https://doi.org/10.1103/PhysRevLett.83.3077), *Physical Review Letters* 83, 3077 (1999).
- S. C. Benjamin and P. M. Hayden, ["Multiplayer quantum games"](https://doi.org/10.1103/PhysRevA.64.030301), *Physical Review A* 64, 030301(R) (2001).
