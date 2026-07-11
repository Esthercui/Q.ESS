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

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Run Step 1 Demo

```bash
PYTHONPATH=src python3 examples/classical_baseline_demo.py
```

## Use From Python

```python
from quantum_ess import classical_two_player_baseline

baseline = classical_two_player_baseline()
print(baseline.payoff_matrix)
print(baseline.nash_equilibria)
print([(result.resident, result.is_ess) for result in baseline.ess_results])
```
