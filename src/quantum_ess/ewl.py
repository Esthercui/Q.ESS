"""Two-player Eisert-Wilkens-Lewenstein quantum game engine."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

BitString = Tuple[int, int]
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]
Payoff = Tuple[float, float]

DEFAULT_PD_PAYOFFS: Mapping[BitString, Payoff] = {
    (0, 0): (3.0, 3.0),
    (0, 1): (0.0, 5.0),
    (1, 0): (5.0, 0.0),
    (1, 1): (1.0, 1.0),
}


@dataclass(frozen=True)
class EWLStrategy:
    """Two-parameter EWL strategy U(theta, phi)."""

    theta: float
    phi: float = 0.0
    label: Optional[str] = None

    def matrix(self) -> Matrix2:
        """Return the 2x2 unitary matrix for this strategy."""
        half_theta = self.theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        phase = cmath.exp(1j * self.phi)
        phase_conjugate = cmath.exp(-1j * self.phi)
        return (
            (phase * c, s),
            (-s, phase_conjugate * c),
        )

    @property
    def name(self) -> str:
        if self.label is not None:
            return self.label
        return f"U(theta={self.theta:.6g}, phi={self.phi:.6g})"


EWL_C = EWLStrategy(theta=0.0, phi=0.0, label="C")
EWL_D = EWLStrategy(theta=math.pi, phi=0.0, label="D")
EWL_Q = EWLStrategy(theta=0.0, phi=math.pi / 2.0, label="Q")


@dataclass(frozen=True)
class EWLResult:
    """Result of one two-player EWL simulation."""

    gamma: float
    strategies: Tuple[EWLStrategy, EWLStrategy]
    probabilities: Dict[BitString, float]
    expected_payoffs: Payoff
    amplitudes: Tuple[complex, complex, complex, complex]
    probability_sum: float

    def most_likely_outcome(self) -> Tuple[BitString, float]:
        """Return the measured outcome with highest probability."""
        return max(self.probabilities.items(), key=lambda item: item[1])


class EWLTwoPlayerGame:
    """Two-player EWL quantum game.

    Outcome convention:
        bit 0 means the classical cooperation-like measurement outcome.
        bit 1 means the classical defection-like measurement outcome.

    The entangler is:
        J = cos(gamma / 2) I + i sin(gamma / 2) (D tensor D)

    with D = U(pi, 0). Classical C/D profiles therefore recover the classical
    deterministic outcomes after J dagger for every gamma.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        payoff_matrix: Mapping[BitString, Payoff] = DEFAULT_PD_PAYOFFS,
    ) -> None:
        if not math.isfinite(gamma):
            raise ValueError(f"gamma must be finite; got {gamma!r}")
        self.gamma = float(gamma)
        self.payoff_matrix = dict(payoff_matrix)
        self._validate_payoff_matrix()

    def with_gamma(self, gamma: float) -> "EWLTwoPlayerGame":
        """Return a copy of the game with a different entanglement value."""
        return EWLTwoPlayerGame(gamma=gamma, payoff_matrix=self.payoff_matrix)

    def run(
        self,
        player_0_strategy: EWLStrategy,
        player_1_strategy: EWLStrategy,
    ) -> EWLResult:
        """Run one EWL game and return amplitudes, probabilities, and payoffs."""
        state = (1.0 + 0j, 0j, 0j, 0j)
        state = self._apply_entangler(state, dagger=False)
        state = self._apply_local_unitary(state, player_0_strategy.matrix(), player=0)
        state = self._apply_local_unitary(state, player_1_strategy.matrix(), player=1)
        state = self._apply_entangler(state, dagger=True)

        probabilities = self._measurement_probabilities(state)
        expected_payoffs = self._expected_payoffs(probabilities)

        return EWLResult(
            gamma=self.gamma,
            strategies=(player_0_strategy, player_1_strategy),
            probabilities=probabilities,
            expected_payoffs=expected_payoffs,
            amplitudes=state,
            probability_sum=sum(probabilities.values()),
        )

    def expected_payoffs(
        self,
        player_0_strategy: EWLStrategy,
        player_1_strategy: EWLStrategy,
    ) -> Payoff:
        """Return only the expected payoff pair."""
        return self.run(player_0_strategy, player_1_strategy).expected_payoffs

    def _apply_entangler(
        self,
        state: Tuple[complex, complex, complex, complex],
        dagger: bool,
    ) -> Tuple[complex, complex, complex, complex]:
        c = math.cos(self.gamma / 2.0)
        s = math.sin(self.gamma / 2.0)
        generated = self._apply_dd_generator(state)
        sign = -1j if dagger else 1j
        return tuple(
            c * amplitude + sign * s * generated_amplitude
            for amplitude, generated_amplitude in zip(state, generated)
        )

    def _apply_dd_generator(
        self,
        state: Tuple[complex, complex, complex, complex],
    ) -> Tuple[complex, complex, complex, complex]:
        # D tensor D maps |00> -> |11>, |01> -> -|10>,
        # |10> -> -|01>, and |11> -> |00>.
        return (state[3], -state[2], -state[1], state[0])

    def _apply_local_unitary(
        self,
        state: Tuple[complex, complex, complex, complex],
        matrix: Matrix2,
        player: int,
    ) -> Tuple[complex, complex, complex, complex]:
        if player == 0:
            return (
                matrix[0][0] * state[0] + matrix[0][1] * state[2],
                matrix[0][0] * state[1] + matrix[0][1] * state[3],
                matrix[1][0] * state[0] + matrix[1][1] * state[2],
                matrix[1][0] * state[1] + matrix[1][1] * state[3],
            )
        if player == 1:
            return (
                matrix[0][0] * state[0] + matrix[0][1] * state[1],
                matrix[1][0] * state[0] + matrix[1][1] * state[1],
                matrix[0][0] * state[2] + matrix[0][1] * state[3],
                matrix[1][0] * state[2] + matrix[1][1] * state[3],
            )
        raise ValueError(f"player must be 0 or 1; got {player!r}")

    def _measurement_probabilities(
        self,
        state: Sequence[complex],
    ) -> Dict[BitString, float]:
        outcomes = ((0, 0), (0, 1), (1, 0), (1, 1))
        probabilities: Dict[BitString, float] = {}
        for outcome, amplitude in zip(outcomes, state):
            probability = float((amplitude.conjugate() * amplitude).real)
            if abs(probability) < 1e-15:
                probability = 0.0
            probabilities[outcome] = probability
        return probabilities

    def _expected_payoffs(self, probabilities: Mapping[BitString, float]) -> Payoff:
        player_0_total = 0.0
        player_1_total = 0.0
        for outcome, probability in probabilities.items():
            payoff_0, payoff_1 = self.payoff_matrix[outcome]
            player_0_total += probability * payoff_0
            player_1_total += probability * payoff_1
        return (float(player_0_total), float(player_1_total))

    def _validate_payoff_matrix(self) -> None:
        expected_outcomes = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if set(self.payoff_matrix) != expected_outcomes:
            raise ValueError(
                "payoff_matrix must define exactly (0,0), (0,1), (1,0), and (1,1)"
            )
        for outcome, payoff in self.payoff_matrix.items():
            if len(payoff) != 2:
                raise ValueError(f"outcome {outcome!r} must have two payoffs")
            if not all(math.isfinite(value) for value in payoff):
                raise ValueError(f"outcome {outcome!r} has non-finite payoffs")


def classical_action_to_ewl_strategy(action: str) -> EWLStrategy:
    """Convert a classical C/D action into its EWL strategy operation."""
    normalized = action.upper()
    if normalized == "C":
        return EWL_C
    if normalized == "D":
        return EWL_D
    raise ValueError(f"classical action must be 'C' or 'D'; got {action!r}")


def two_player_ewl_game(gamma: float = 0.0) -> EWLTwoPlayerGame:
    """Return the default two-player EWL Prisoner's Dilemma game."""
    return EWLTwoPlayerGame(gamma=gamma)
