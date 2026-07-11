"""Classical two-player baselines for ESS research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

Action = str
Profile = Tuple[Action, Action]
Payoff = Tuple[float, float]

DEFAULT_ACTIONS: Tuple[Action, Action] = ("C", "D")


@dataclass(frozen=True)
class MutantESSComparison:
    """One pure-mutant ESS comparison against a resident strategy."""

    resident: Action
    mutant: Action
    resident_against_resident: float
    mutant_against_resident: float
    resident_against_mutant: float
    mutant_against_mutant: float
    invasion_strength: float
    blocks_ess: bool
    reason: str


@dataclass(frozen=True)
class ClassicalESSResult:
    """Pure-strategy ESS result for one resident action."""

    resident: Action
    is_ess: bool
    comparisons: Tuple[MutantESSComparison, ...]

    @property
    def max_mutant_advantage(self) -> float:
        """Return the strongest rare-mutant payoff advantage."""
        if not self.comparisons:
            return 0.0
        return max(comparison.invasion_strength for comparison in self.comparisons)


@dataclass(frozen=True)
class ClassicalBaselineResult:
    """Step 1 output: payoff matrix, pure Nash equilibria, and ESS checks."""

    payoff_matrix: Mapping[Profile, Payoff]
    nash_equilibria: Tuple[Profile, ...]
    ess_results: Tuple[ClassicalESSResult, ...]

    def ess_for(self, resident: Action) -> ClassicalESSResult:
        """Return the ESS result for a resident action."""
        for result in self.ess_results:
            if result.resident == resident:
                return result
        raise KeyError(f"unknown resident action {resident!r}")


class ClassicalTwoPlayerGame:
    """Finite two-player normal-form game with pure-strategy analysis helpers."""

    def __init__(
        self,
        payoff_matrix: Mapping[Profile, Payoff],
        actions: Tuple[Action, ...] = DEFAULT_ACTIONS,
    ) -> None:
        if len(actions) < 2:
            raise ValueError("at least two actions are required")
        if len(set(actions)) != len(actions):
            raise ValueError(f"actions must be unique; got {actions!r}")

        self.actions = tuple(actions)
        self.payoff_matrix = dict(payoff_matrix)
        self._validate_payoff_matrix()

    def payoff(self, profile: Profile) -> Payoff:
        """Return the ordered payoff pair for a pure action profile."""
        if profile not in self.payoff_matrix:
            raise KeyError(f"unknown action profile {profile!r}")
        return self.payoff_matrix[profile]

    def pure_nash_equilibria(self, epsilon: float = 1e-8) -> Tuple[Profile, ...]:
        """Return all pure Nash equilibria."""
        equilibria = []
        for row_action in self.actions:
            for column_action in self.actions:
                profile = (row_action, column_action)
                current = self.payoff(profile)
                row_can_improve = any(
                    self.payoff((alternative, column_action))[0]
                    > current[0] + epsilon
                    for alternative in self.actions
                    if alternative != row_action
                )
                column_can_improve = any(
                    self.payoff((row_action, alternative))[1]
                    > current[1] + epsilon
                    for alternative in self.actions
                    if alternative != column_action
                )
                if not row_can_improve and not column_can_improve:
                    equilibria.append(profile)
        return tuple(equilibria)

    def ess_results(self, epsilon: float = 1e-8) -> Tuple[ClassicalESSResult, ...]:
        """Return pure-strategy ESS results for every resident action."""
        return tuple(
            self.ess_result(resident=resident, epsilon=epsilon)
            for resident in self.actions
        )

    def ess_result(
        self,
        resident: Action,
        epsilon: float = 1e-8,
    ) -> ClassicalESSResult:
        """Evaluate whether one pure resident action is an ESS.

        The check uses the standard symmetric-game pure ESS condition:
        resident R resists mutant M when either u(R, R) > u(M, R), or when
        those are tied and u(R, M) > u(M, M).
        """
        self._validate_action(resident)
        comparisons = []
        for mutant in self.actions:
            if mutant == resident:
                continue

            resident_against_resident = self.payoff((resident, resident))[0]
            mutant_against_resident = self.payoff((mutant, resident))[0]
            resident_against_mutant = self.payoff((resident, mutant))[0]
            mutant_against_mutant = self.payoff((mutant, mutant))[0]
            invasion_strength = mutant_against_resident - resident_against_resident

            if invasion_strength > epsilon:
                blocks_ess = True
                reason = "mutant earns more against resident than resident does"
            elif abs(invasion_strength) <= epsilon:
                blocks_ess = resident_against_mutant <= mutant_against_mutant + epsilon
                reason = (
                    "resident loses the second-order ESS tie-break"
                    if blocks_ess
                    else "resident wins the second-order ESS tie-break"
                )
            else:
                blocks_ess = False
                reason = "resident earns more against itself than mutant does"

            comparisons.append(
                MutantESSComparison(
                    resident=resident,
                    mutant=mutant,
                    resident_against_resident=resident_against_resident,
                    mutant_against_resident=mutant_against_resident,
                    resident_against_mutant=resident_against_mutant,
                    mutant_against_mutant=mutant_against_mutant,
                    invasion_strength=invasion_strength,
                    blocks_ess=blocks_ess,
                    reason=reason,
                )
            )

        return ClassicalESSResult(
            resident=resident,
            is_ess=not any(comparison.blocks_ess for comparison in comparisons),
            comparisons=tuple(comparisons),
        )

    def baseline(self, epsilon: float = 1e-8) -> ClassicalBaselineResult:
        """Return the complete Step 1 classical baseline."""
        return ClassicalBaselineResult(
            payoff_matrix=dict(self.payoff_matrix),
            nash_equilibria=self.pure_nash_equilibria(epsilon=epsilon),
            ess_results=self.ess_results(epsilon=epsilon),
        )

    def _validate_action(self, action: Action) -> None:
        if action not in self.actions:
            raise ValueError(
                f"unknown action {action!r}; expected one of {self.actions!r}"
            )

    def _validate_payoff_matrix(self) -> None:
        for row_action in self.actions:
            for column_action in self.actions:
                profile = (row_action, column_action)
                if profile not in self.payoff_matrix:
                    raise ValueError(f"missing payoff for profile {profile!r}")
                payoff = self.payoff_matrix[profile]
                if len(payoff) != 2:
                    raise ValueError(f"profile {profile!r} must have two payoffs")
                if not all(math.isfinite(value) for value in payoff):
                    raise ValueError(f"profile {profile!r} has non-finite payoffs")


def prisoners_dilemma_game(
    reward: float = 3.0,
    sucker: float = 0.0,
    temptation: float = 5.0,
    punishment: float = 1.0,
) -> ClassicalTwoPlayerGame:
    """Return the default two-player Prisoner's Dilemma normal-form game."""
    return ClassicalTwoPlayerGame(
        payoff_matrix={
            ("C", "C"): (reward, reward),
            ("C", "D"): (sucker, temptation),
            ("D", "C"): (temptation, sucker),
            ("D", "D"): (punishment, punishment),
        }
    )


def classical_two_player_baseline(epsilon: float = 1e-8) -> ClassicalBaselineResult:
    """Return the Step 1 baseline for the default two-player Prisoner's Dilemma."""
    return prisoners_dilemma_game().baseline(epsilon=epsilon)
