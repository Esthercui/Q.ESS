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

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Run Demos

```bash
PYTHONPATH=src python3 examples/classical_baseline_demo.py
PYTHONPATH=src python3 examples/k_player_baseline_demo.py
```

## Use From Python

```python
from quantum_ess import classical_two_player_baseline, classical_k_player_baseline

step_1 = classical_two_player_baseline()
print(step_1.payoff_matrix)
print(step_1.nash_equilibria)
print([(result.resident, result.is_ess) for result in step_1.ess_results])

step_2 = classical_k_player_baseline(k=5)
print(step_2.payoff_table)
print(step_2.one_defector_in_cooperators)
print(step_2.one_cooperator_in_defectors)
```
