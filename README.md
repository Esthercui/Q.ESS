# Q.ESS

Code for the Quantum Evolutionarily Stable Strategy research project, starting with classical Prisoner's Dilemma baselines and building toward quantum EWL games against mutants.

## Step 1: Classical 2 Player Baseline

Implemented outputs:

- Classical two-player Prisoner's Dilemma payoff matrix
- Pure Nash equilibrium detection
- Pure resident ESS checks against mutant strategies
- Invasion strength, defined as `payoff(mutant, resident) - payoff(resident, resident)`

Default payoff matrix:

```text
CC: (3, 3)
CD: (0, 5)
DC: (5, 0)
DD: (1, 1)
```

Expected result:

- Pure Nash equilibrium: `(D, D)`
- Cooperation `C` is not ESS because defection invades with advantage `2`
- Defection `D` is ESS because cooperation has advantage `-1`

## Step 2: Classical K-Player Baseline

Implemented outputs:

- K-player payoff as a function of the total number of cooperators
- Pure action-profile payoff calculation for any K-player `C`/`D` profile
- Resident-mutant invasion test for one or more mutants
- Invasion strength, defined as `mutant_payoff - resident_payoff`

The K-player model is a pairwise extension of the Step 1 Prisoner's Dilemma: each player plays the same two-player game against every other player, and the K-player payoff is the sum of those pairwise payoffs. For `K=2`, this exactly recovers the Step 1 payoff matrix.

For the default `K=5` case:

- One `D` mutant in `C` residents invades with strength `11`
- One `C` mutant in `D` residents does not invade; its strength is `-8`

## Step 3: 2-Player EWL Engine

Implemented outputs:

- Two-player EWL strategy family `U(theta, phi)`
- Classical `C` and `D` as special quantum operations
- Entangle, apply local strategies, disentangle, then measure
- Measurement probabilities for `CC`, `CD`, `DC`, and `DD` outcomes
- Expected payoffs from the Step 1 Prisoner's Dilemma payoff matrix

The strategy matrix is:

```text
U(theta, phi) = [[exp(i phi) cos(theta/2),  sin(theta/2)],
                 [-sin(theta/2),            exp(-i phi) cos(theta/2)]]
```

The entangler is:

```text
J = cos(gamma/2) I + i sin(gamma/2) (D tensor D)
```

Validation checks covered by tests:

- When `gamma = 0`, classical C/D behavior is recovered
- Classical `C = U(0, 0)` and `D = U(pi, 0)` are special EWL operations
- Measurement probabilities sum to `1`
- Classical C/D profiles match expected classical payoffs in classical limits
- Classical C/D profiles remain deterministic under entanglement

## Step 4: K-Player EWL Engine and Strategy Grid

Implemented outputs:

- K-player EWL engine for `K = 2, 3, 4, 5`
- Expected payoff for each player under arbitrary K-player strategy profiles
- K-player measurement probabilities over all bitstring outcomes
- Resident-mutant profile helpers with arbitrary mutant indices
- Average resident and mutant payoff calculation for later invasion logic
- Finite EWL strategy grids over `theta` and `phi`
- Exact pure Nash equilibrium search on a K-player grid, including `K=5`
- One result row for every ordered Nash profile
- Direct player-by-player verification against every unilateral grid deviation
- Per-player angles, expected payoffs, best responses, and maximum deviation gains
- Both total pairwise-summed payoff and average payoff per opponent
- Optional multiprocessing for large grid searches

The K-player EWL engine reuses the Step 3 strategy family `U(theta, phi)` and extends the entangler to a full K-player state vector. For odd `K`, the generator is phase-adjusted so it remains Hermitian and unitary-safe. Classical C/D profiles still recover deterministic classical outcomes after disentangling.

One grid point `U(theta, phi)` is one pure strategy available to one player. An ordered tuple of `K` grid points is a strategy profile. The default research grid matches the earlier notebook:

```text
theta_count = 11
phi_count = 6
strategies = 66
```

Thus every player has 66 possible pure strategies. For `K=5`, the finite game contains:

```text
66^5 = 1,252,332,576
```

The search is exact: a profile is Nash only if no player can improve by changing alone to any of the 66 grid strategies. Internally, the default game is symmetric, so identical opponent multisets share one best-response calculation. This reduces repeated work to:

```text
C(66 + 5 - 1, 5) = C(70, 5) = 12,103,014
```

This symmetry calculation does not remove strategies, profiles, or deviations
from the finite game. Confirmed equilibria are expanded back into every distinct
ordered player profile, and every returned profile receives an independent
unilateral-deviation audit.

The engine's primary payoff is the sum over all pairwise interactions, matching
the Step 2 classical K-player baseline. Results also report that total divided
by `K - 1` as the average payoff per opponent. For example, four-player mutual
cooperation is total payoff `9` and average payoff `3`; these are the same
payoff convention at different scales, and the scaling does not change Nash
equilibrium membership.

The grid resolution is configurable. For example, `21` theta values and `11`
phi values give 231 strategies per player. Grid refinement should be reported
in the research because a finite-grid equilibrium can change when additional
strategies become available.

Validation checks covered by tests:

- `K=2` matches the Step 3 two-player EWL engine
- `K=2,3,4,5` probability distributions sum to `1`
- `gamma = 0` recovers classical C/D behavior
- Classical C/D profiles recover classical outcomes under entanglement
- Arbitrary quantum profiles return one expected payoff per player
- Resident-mutant profile helpers return grouped average payoffs
- Small-grid Nash search recovers all-defection for classical `K=5`
- The profile `(D,C,C,C,C)` returns `(20,9,9,9,9)` but fails the Nash audit
- Every returned equilibrium passes a direct all-players/all-deviations audit

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Run Demos

```bash
PYTHONPATH=src python3 examples/classical_baseline_demo.py
PYTHONPATH=src python3 examples/k_player_baseline_demo.py
PYTHONPATH=src python3 examples/two_player_ewl_demo.py
PYTHONPATH=src python3 examples/k_player_ewl_demo.py
PYTHONPATH=src python3 examples/k_player_ewl_grid_search_demo.py
```

For a larger `K=5` grid search on a 25-core machine:

```bash
PYTHONPATH=src python3 examples/k_player_ewl_grid_search_demo.py \
  --k 5 \
  --gamma 1.5707963267948966 \
  --theta-count 11 \
  --phi-count 6 \
  --workers 25 \
  --chunksize 256 \
  --output k5_ewl_nash_results.json
```

To refine the grid beyond the earlier notebook, change the counts explicitly:

```bash
PYTHONPATH=src python3 examples/k_player_ewl_grid_search_demo.py \
  --k 5 \
  --theta-count 21 \
  --phi-count 11 \
  --workers 25 \
  --output k5_ewl_nash_dense_results.json
```

## Use From Python

```python
import math

from quantum_ess import (
    EWL_C,
    EWL_D,
    EWL_Q,
    EWLTwoPlayerGame,
    KPlayerEWLGame,
    classical_k_player_baseline,
    classical_two_player_baseline,
    ewl_strategy_grid,
    find_pure_nash_equilibria_on_grid,
    k_player_resident_mutant_profile,
)

step_1 = classical_two_player_baseline()
print(step_1.payoff_matrix)
print(step_1.nash_equilibria)
print([(result.resident, result.is_ess) for result in step_1.ess_results])

step_2 = classical_k_player_baseline(k=5)
print(step_2.payoff_table)
print(step_2.one_defector_in_cooperators)
print(step_2.one_cooperator_in_defectors)

step_3 = EWLTwoPlayerGame(gamma=math.pi / 2)
print(step_3.run(EWL_C, EWL_D).probabilities)
print(step_3.run(EWL_Q, EWL_Q).expected_payoffs)

step_4 = KPlayerEWLGame(k=5, gamma=math.pi / 4)
profile = k_player_resident_mutant_profile(
    k=5,
    resident=EWL_C,
    mutant=EWL_D,
    mutant_indices=(0,),
)
result = step_4.run(profile)
print(result.probabilities)
print(result.expected_payoffs)

grid = ewl_strategy_grid(theta_count=11, phi_count=6)
nash = find_pure_nash_equilibria_on_grid(
    k=5,
    gamma=math.pi / 2,
    strategies=grid.strategies,
    workers=1,
)
print(nash.ordered_profile_count)
for equilibrium in nash.equilibria:
    print(equilibrium.strategy_indices)
    print(equilibrium.angles())
    print(equilibrium.payoffs)
    print(equilibrium.unilateral_gains)
```
