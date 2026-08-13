"""K-player Eisert-Wilkens-Lewenstein quantum game engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ewl import EWL_C, EWL_D, EWL_Q, EWLStrategy

BitStringK = Tuple[int, ...]
PayoffFunction = Callable[[BitStringK], Sequence[float]]

SUPPORTED_K_VALUES = (2, 3, 4, 5)


@dataclass(frozen=True)
class KPlayerEWLResult:
    """Result of one K-player EWL simulation."""

    k: int
    gamma: float
    strategies: Tuple[EWLStrategy, ...]
    probabilities: Dict[BitStringK, float]
    expected_payoffs: Tuple[float, ...]
    amplitudes: Tuple[complex, ...]
    probability_sum: float

    def most_likely_outcome(self) -> Tuple[BitStringK, float]:
        """Return the measured outcome with highest probability."""
        return max(self.probabilities.items(), key=lambda item: item[1])


class KPlayerEWLGame:
    """EWL quantum engine for K-player games.

    The research scope currently supports K=2,3,4,5. The engine keeps the full
    state vector in memory, so K=5 is only 32 amplitudes and remains easy to
    inspect without external numerical dependencies.

    Outcome convention:
        bit 0 means the classical cooperation-like measurement outcome.
        bit 1 means the classical defection-like measurement outcome.
    """

    def __init__(
        self,
        k: int,
        payoff_fn: Optional[PayoffFunction] = None,
        gamma: float = 0.0,
    ) -> None:
        _validate_k(k)
        if not math.isfinite(gamma):
            raise ValueError(f"gamma must be finite; got {gamma!r}")

        self.k = k
        self.payoff_fn = payoff_fn if payoff_fn is not None else k_player_ewl_pd_payoff(k)
        self.gamma = float(gamma)

    def with_gamma(self, gamma: float) -> "KPlayerEWLGame":
        """Return a copy of the game with a different entanglement value."""
        return KPlayerEWLGame(k=self.k, payoff_fn=self.payoff_fn, gamma=gamma)

    def run(self, strategies: Sequence[EWLStrategy]) -> KPlayerEWLResult:
        """Run one K-player EWL game and return probabilities and payoffs."""
        strategies_tuple = tuple(strategies)
        if len(strategies_tuple) != self.k:
            raise ValueError(f"expected {self.k} strategies, got {len(strategies_tuple)}")

        state = self._initial_state()
        state = self._apply_entangler(state, dagger=False)
        for player_index, strategy in enumerate(strategies_tuple):
            state = self._apply_local_unitary(state, strategy.matrix(), player_index)
        state = self._apply_entangler(state, dagger=True)

        probabilities = self._measurement_probabilities(state)
        expected_payoffs = self._expected_payoffs(probabilities)

        return KPlayerEWLResult(
            k=self.k,
            gamma=self.gamma,
            strategies=strategies_tuple,
            probabilities=probabilities,
            expected_payoffs=expected_payoffs,
            amplitudes=tuple(state),
            probability_sum=sum(probabilities.values()),
        )

    def expected_payoffs(self, strategies: Sequence[EWLStrategy]) -> Tuple[float, ...]:
        """Return only the expected payoff vector for one strategy profile."""
        return self.run(strategies).expected_payoffs

    def resident_mutant_payoffs(
        self,
        resident: EWLStrategy,
        mutant: EWLStrategy,
        mutant_indices: Sequence[int] = (0,),
    ) -> Tuple[float, float, KPlayerEWLResult]:
        """Return average mutant payoff, average resident payoff, and full result."""
        mutant_indices_tuple = _validate_mutant_indices(self.k, mutant_indices)
        profile = k_player_resident_mutant_profile(
            k=self.k,
            resident=resident,
            mutant=mutant,
            mutant_indices=mutant_indices_tuple,
        )
        result = self.run(profile)
        mutant_set = set(mutant_indices_tuple)
        mutant_payoffs = [
            payoff for player, payoff in enumerate(result.expected_payoffs)
            if player in mutant_set
        ]
        resident_payoffs = [
            payoff for player, payoff in enumerate(result.expected_payoffs)
            if player not in mutant_set
        ]
        return (
            sum(mutant_payoffs) / len(mutant_payoffs),
            sum(resident_payoffs) / len(resident_payoffs),
            result,
        )

    def _initial_state(self) -> List[complex]:
        state = [0j for _ in range(1 << self.k)]
        state[0] = 1.0 + 0j
        return state

    def _apply_entangler(self, state: Sequence[complex], dagger: bool) -> List[complex]:
        c = math.cos(self.gamma / 2.0)
        s = math.sin(self.gamma / 2.0)
        generated = self._apply_entangler_generator(state)
        sign = -1j if dagger else 1j
        return [
            c * amplitude + sign * s * generated_amplitude
            for amplitude, generated_amplitude in zip(state, generated)
        ]

    def _apply_entangler_generator(self, state: Sequence[complex]) -> List[complex]:
        """Apply the Hermitian K-player EWL generator to the state vector."""
        size = 1 << self.k
        out = [0j for _ in range(size)]
        odd_k_phase = 1j if self.k % 2 else 1.0 + 0j
        for index, amplitude in enumerate(state):
            if amplitude == 0:
                continue
            bits = index_to_bits(index, self.k)
            zeros = self.k - sum(bits)
            phase = odd_k_phase * ((-1) ** zeros)
            out[complement_index(index, self.k)] += phase * amplitude
        return out

    def _apply_local_unitary(
        self,
        state: Sequence[complex],
        matrix: Tuple[Tuple[complex, complex], Tuple[complex, complex]],
        player_index: int,
    ) -> List[complex]:
        if not 0 <= player_index < self.k:
            raise ValueError(f"player_index out of range: {player_index!r}")

        out = list(state)
        target_shift = self.k - 1 - player_index
        target_mask = 1 << target_shift
        for base in range(1 << self.k):
            if base & target_mask:
                continue
            i0 = base
            i1 = base | target_mask
            a0 = state[i0]
            a1 = state[i1]
            out[i0] = matrix[0][0] * a0 + matrix[0][1] * a1
            out[i1] = matrix[1][0] * a0 + matrix[1][1] * a1
        return out

    def _measurement_probabilities(self, state: Sequence[complex]) -> Dict[BitStringK, float]:
        probabilities: Dict[BitStringK, float] = {}
        for index, amplitude in enumerate(state):
            probability = float((amplitude.conjugate() * amplitude).real)
            if abs(probability) < 1e-15:
                probability = 0.0
            probabilities[index_to_bits(index, self.k)] = probability
        return probabilities

    def _expected_payoffs(self, probabilities: Mapping[BitStringK, float]) -> Tuple[float, ...]:
        totals = [0.0 for _ in range(self.k)]
        for outcome, probability in probabilities.items():
            payoffs = tuple(float(value) for value in self.payoff_fn(outcome))
            if len(payoffs) != self.k:
                raise ValueError(
                    f"payoff_fn returned {len(payoffs)} payoffs for K={self.k}"
                )
            if not all(math.isfinite(value) for value in payoffs):
                raise ValueError(f"payoff_fn returned non-finite payoffs for {outcome!r}")
            for player, payoff in enumerate(payoffs):
                totals[player] += probability * payoff
        return tuple(totals)


def bits_to_index(bits: Sequence[int]) -> int:
    """Convert a big-endian bitstring to a basis-state index."""
    index = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError(f"bits must contain only 0 or 1; got {bits!r}")
        index = (index << 1) | bit
    return index


def index_to_bits(index: int, k: int) -> BitStringK:
    """Convert a basis-state index to a big-endian K-bit tuple."""
    if index < 0 or index >= (1 << k):
        raise ValueError(f"index {index!r} is outside the {k}-player state space")
    return tuple((index >> shift) & 1 for shift in range(k - 1, -1, -1))


def complement_index(index: int, k: int) -> int:
    """Flip every bit in a basis-state index."""
    return ((1 << k) - 1) ^ index


def k_player_ewl_pd_payoff(
    k: int,
    reward: float = 3.0,
    sucker: float = 0.0,
    temptation: float = 5.0,
    punishment: float = 1.0,
) -> PayoffFunction:
    """Return a K-player pairwise Prisoner's Dilemma payoff function."""
    _validate_k(k)
    payoffs = (reward, sucker, temptation, punishment)
    if not all(math.isfinite(value) for value in payoffs):
        raise ValueError("payoffs must be finite real numbers")

    def payoff(outcome: BitStringK) -> Tuple[float, ...]:
        if len(outcome) != k:
            raise ValueError(f"expected a {k}-bit outcome, got {outcome!r}")
        if any(bit not in (0, 1) for bit in outcome):
            raise ValueError(f"outcome bits must be 0 or 1, got {outcome!r}")

        cooperators = outcome.count(0)
        defectors = k - cooperators
        values = []
        for bit in outcome:
            if bit == 0:
                values.append(reward * (cooperators - 1) + sucker * defectors)
            else:
                values.append(temptation * cooperators + punishment * (defectors - 1))
        return tuple(float(value) for value in values)

    return payoff


def k_player_classical_profile(actions: Sequence[str]) -> Tuple[EWLStrategy, ...]:
    """Convert classical C/D labels into K-player EWL strategies."""
    profile = []
    for action in actions:
        normalized = action.upper()
        if normalized == "C":
            profile.append(EWL_C)
        elif normalized == "D":
            profile.append(EWL_D)
        else:
            raise ValueError(f"classical action must be 'C' or 'D'; got {action!r}")
    return tuple(profile)


def all_k_player_classical_profiles(k: int) -> Iterable[Tuple[EWLStrategy, ...]]:
    """Yield every K-player classical C/D EWL profile."""
    _validate_k(k)
    for index in range(1 << k):
        bits = index_to_bits(index, k)
        yield tuple(EWL_C if bit == 0 else EWL_D for bit in bits)


def k_player_resident_mutant_profile(
    k: int,
    resident: EWLStrategy,
    mutant: EWLStrategy,
    mutant_indices: Sequence[int] = (0,),
) -> Tuple[EWLStrategy, ...]:
    """Build a K-player strategy profile with residents and mutants."""
    mutant_indices_tuple = _validate_mutant_indices(k, mutant_indices)
    mutant_set = set(mutant_indices_tuple)
    return tuple(mutant if player in mutant_set else resident for player in range(k))


def k_player_ewl_game(k: int = 5, gamma: float = 0.0) -> KPlayerEWLGame:
    """Return the default K-player EWL Prisoner's Dilemma game."""
    return KPlayerEWLGame(k=k, gamma=gamma)


def _validate_k(k: int) -> None:
    if k not in SUPPORTED_K_VALUES:
        raise ValueError(f"K must be one of {SUPPORTED_K_VALUES}; got {k!r}")


def _validate_mutant_indices(k: int, mutant_indices: Sequence[int]) -> Tuple[int, ...]:
    _validate_k(k)
    indices = tuple(mutant_indices)
    if not indices:
        raise ValueError("at least one mutant index is required")
    if len(set(indices)) != len(indices):
        raise ValueError(f"mutant_indices must be unique; got {indices!r}")
    if len(indices) >= k:
        raise ValueError("resident-mutant profiles require at least one resident")
    for index in indices:
        if not 0 <= index < k:
            raise ValueError(f"mutant index out of range: {index!r}")
    return indices
