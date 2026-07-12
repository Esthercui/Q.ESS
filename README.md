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

## Step 4: K-Player EWL Engine

Implemented outputs:

- K-player EWL engine for `K = 2, 3, 4, 5`
- Expected payoff for each player under arbitrary K-player strategy profiles
- K-player measurement probabilities over all bitstring outcomes
- Resident-mutant profile helpers with arbitrary mutant indices
- Average resident and mutant payoff calculation for later invasion logic

The K-player EWL engine reuses the Step 3 strategy family `U(theta, phi)` and extends the entangler to a full K-player state vector. For odd `K`, the generator is phase-adjusted so it remains Hermitian and unitary-safe. Classical C/D profiles still recover deterministic classical outcomes after disentangling.

Validation checks covered by tests:

- `K=2` matches the Step 3 two-player EWL engine
- `K=2,3,4,5` probability distributions sum to `1`
- `gamma = 0` recovers classical C/D behavior
- Classical C/D profiles recover classical outcomes under entanglement
- Arbitrary quantum profiles return one expected payoff per player
- Resident-mutant profile helpers return grouped average payoffs

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
```
