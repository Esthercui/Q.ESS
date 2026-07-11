"""Classical K-player Prisoner's Dilemma baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Action = str


@dataclass(frozen=True)
class KPlayerPayoffRow:
    """Payoffs for a total number of cooperators in a K-player group."""

    cooperators: int
    cooperator_payoff: Optional[float]
    defector_payoff: Optional[float]


@dataclass(frozen=True)
class KPlayerInvasionResult:
    """Resident-mutant invasion result for a classical K-player group."""

    k: int
    resident: Action
    mutant: Action
    resident_count: int
    mutant_count: int
    cooperators: int
    resident_payoff: float
    mutant_payoff: float
    invasion_strength: float
    invades: bool


@dataclass(frozen=True)
class ClassicalKPlayerBaselineResult:
    """Step 2 output for the classical K-player baseline."""

    k: int
    payoff_table: Tuple[KPlayerPayoffRow, ...]
    one_defector_in_cooperators: KPlayerInvasionResult
    one_cooperator_in_defectors: KPlayerInvasionResult


class ClassicalKPlayerGame:
    """K-player pairwise extension of the classical Prisoner's Dilemma.

    Each player plays the same two-player Prisoner's Dilemma against every other
    player, and that player's K-player payoff is the sum of those pairwise
    payoffs. For K=2 this exactly recovers the Step 1 payoff matrix.
    """

    def __init__(
        self,
        k: int,
        reward: float = 3.0,
        sucker: float = 0.0,
        temptation: float = 5.0,
        punishment: float = 1.0,
    ) -> None:
        if k < 2:
            raise ValueError(f"k must be at least 2; got {k!r}")
        payoffs = (reward, sucker, temptation, punishment)
        if not all(math.isfinite(value) for value in payoffs):
            raise ValueError("payoffs must be finite real numbers")

        self.k = k
        self.reward = float(reward)
        self.sucker = float(sucker)
        self.temptation = float(temptation)
        self.punishment = float(punishment)

    def payoff_for_action(self, action: Action, cooperators: int) -> float:
        """Return focal payoff given total cooperators in the group.

        Args:
            action: Focal player action, either "C" or "D".
            cooperators: Total number of cooperators in the whole K-player group,
                including the focal player when action is "C".
        """
        self._validate_action(action)
        self._validate_cooperator_count(cooperators)

        if action == "C":
            if cooperators == 0:
                raise ValueError("a focal cooperator requires at least one cooperator")
            cooperator_opponents = cooperators - 1
            defector_opponents = self.k - cooperators
            return (
                self.reward * cooperator_opponents
                + self.sucker * defector_opponents
            )

        if cooperators == self.k:
            raise ValueError("a focal defector requires at least one defector")
        cooperator_opponents = cooperators
        defector_opponents = self.k - cooperators - 1
        return (
            self.temptation * cooperator_opponents
            + self.punishment * defector_opponents
        )

    def payoff_table(self) -> Tuple[KPlayerPayoffRow, ...]:
        """Return payoffs as a function of the total number of cooperators."""
        rows = []
        for cooperators in range(self.k + 1):
            cooperator_payoff = None
            defector_payoff = None
            if cooperators > 0:
                cooperator_payoff = self.payoff_for_action("C", cooperators)
            if cooperators < self.k:
                defector_payoff = self.payoff_for_action("D", cooperators)
            rows.append(
                KPlayerPayoffRow(
                    cooperators=cooperators,
                    cooperator_payoff=cooperator_payoff,
                    defector_payoff=defector_payoff,
                )
            )
        return tuple(rows)

    def profile_payoffs(self, actions: Sequence[Action]) -> Tuple[float, ...]:
        """Return one payoff per player for a pure K-player action profile."""
        if len(actions) != self.k:
            raise ValueError(f"expected {self.k} actions, got {len(actions)}")
        normalized = tuple(action.upper() for action in actions)
        for action in normalized:
            self._validate_action(action)
        cooperators = normalized.count("C")
        return tuple(
            self.payoff_for_action(action, cooperators)
            for action in normalized
        )

    def resident_mutant_invasion(
        self,
        resident: Action,
        mutant: Action,
        mutant_count: int = 1,
        epsilon: float = 1e-8,
    ) -> KPlayerInvasionResult:
        """Test whether rare mutants beat residents in a K-player group."""
        resident = resident.upper()
        mutant = mutant.upper()
        self._validate_action(resident)
        self._validate_action(mutant)
        if not 1 <= mutant_count < self.k:
            raise ValueError(
                f"mutant_count must be between 1 and {self.k - 1}; got {mutant_count!r}"
            )

        resident_count = self.k - mutant_count
        cooperators = 0
        if resident == "C":
            cooperators += resident_count
        if mutant == "C":
            cooperators += mutant_count

        resident_payoff = self.payoff_for_action(resident, cooperators)
        mutant_payoff = self.payoff_for_action(mutant, cooperators)
        invasion_strength = mutant_payoff - resident_payoff

        return KPlayerInvasionResult(
            k=self.k,
            resident=resident,
            mutant=mutant,
            resident_count=resident_count,
            mutant_count=mutant_count,
            cooperators=cooperators,
            resident_payoff=resident_payoff,
            mutant_payoff=mutant_payoff,
            invasion_strength=invasion_strength,
            invades=invasion_strength > epsilon,
        )

    def baseline(self, epsilon: float = 1e-8) -> ClassicalKPlayerBaselineResult:
        """Return the complete Step 2 classical K-player baseline."""
        return ClassicalKPlayerBaselineResult(
            k=self.k,
            payoff_table=self.payoff_table(),
            one_defector_in_cooperators=self.resident_mutant_invasion(
                resident="C",
                mutant="D",
                epsilon=epsilon,
            ),
            one_cooperator_in_defectors=self.resident_mutant_invasion(
                resident="D",
                mutant="C",
                epsilon=epsilon,
            ),
        )

    def _validate_action(self, action: Action) -> None:
        if action not in ("C", "D"):
            raise ValueError(f"action must be 'C' or 'D'; got {action!r}")

    def _validate_cooperator_count(self, cooperators: int) -> None:
        if not 0 <= cooperators <= self.k:
            raise ValueError(
                f"cooperators must be between 0 and {self.k}; got {cooperators!r}"
            )


def classical_k_player_game(k: int = 5) -> ClassicalKPlayerGame:
    """Return the default classical K-player Prisoner's Dilemma game."""
    return ClassicalKPlayerGame(k=k)


def classical_k_player_baseline(k: int = 5, epsilon: float = 1e-8) -> ClassicalKPlayerBaselineResult:
    """Return the Step 2 baseline for the default K-player game."""
    return classical_k_player_game(k=k).baseline(epsilon=epsilon)
