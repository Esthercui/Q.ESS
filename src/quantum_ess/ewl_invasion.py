"""Rare-mutant invasion analysis for symmetric K-player EWL games.

The population model is infinite and well mixed. Conditional on a focal
individual, each of its K-1 co-players is independently mutant with frequency
epsilon. The resulting co-player mutant count is therefore binomial.

This module deliberately keeps population ESS fitness separate from the
diagnostic experiment that inserts exactly one mutant into one fixed group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Optional, Sequence, Tuple

from .ewl import EWLStrategy
from .k_player_ewl import KPlayerEWLGame

DEFAULT_TOLERANCE = 1e-8
DEFAULT_DIAGNOSTIC_FREQUENCIES = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
)

RARE_MUTANT_INVADES = "rare_mutant_invades"
RARE_MUTANT_REJECTED = "rare_mutant_rejected"
NEUTRAL = "neutral"
INDETERMINATE = "indeterminate"

# This is only for recognizing floating-point representations of exact domain
# endpoints. It is intentionally much smaller than the payoff tolerance so a
# genuinely nearby mutant is not collapsed into the resident strategy.
ANGLE_ENDPOINT_TOLERANCE = 16.0 * math.ulp(math.pi)


@dataclass(frozen=True)
class FrequencyDiagnostic:
    """Population fitness comparison at one declared mutant frequency."""

    epsilon: float
    resident_payoff: float
    mutant_payoff: float
    mutant_advantage: float


@dataclass(frozen=True)
class FocalCompositionPayoff:
    """Type-conditioned focal payoffs for one co-player composition."""

    mutant_co_players: int
    resident_payoff: float
    mutant_payoff: float
    resident_positional_spread: float
    mutant_positional_spread: float
    max_probability_error: float
    conditional_profile_count: int


@dataclass(frozen=True)
class PopulationInvasionResult:
    """Rare-mutant result for one resident, mutant, and focal qubit position."""

    k: int
    gamma: float
    resident: EWLStrategy
    mutant: EWLStrategy
    mutant_position: int
    epsilon_reference: float
    resident_payoff_reference: float
    mutant_payoff_reference: float
    mutant_advantage_reference: float
    composition_payoffs: Tuple[FocalCompositionPayoff, ...]
    bernstein_advantages: Tuple[float, ...]
    power_coefficients: Tuple[float, ...]
    leading_nonzero_order: Optional[int]
    leading_coefficient: float
    rare_mutant_classification: str
    invades_at_epsilon_reference: bool
    diagnostic_frequencies: Tuple[FrequencyDiagnostic, ...]
    tolerance: float
    permutation_error: float
    probability_error: float
    polynomial_reconstruction_error: float
    uncertainty_or_confidence_interval: str


@dataclass(frozen=True)
class ResidentMutantAnalysis:
    """Position-resolved population analysis for one resident-mutant pair."""

    k: int
    gamma: float
    resident: EWLStrategy
    mutant: EWLStrategy
    position_results: Tuple[PopulationInvasionResult, ...]
    rare_mutant_classification: str
    permutation_error: float

    @property
    def worst_reference_result(self) -> PopulationInvasionResult:
        """Return the position with the largest reference-frequency advantage."""
        return max(
            self.position_results,
            key=lambda result: result.mutant_advantage_reference,
        )


@dataclass(frozen=True)
class ExactlyOneMutantGroupDiagnostic:
    """Payoffs in one fixed group containing exactly one mutant.

    This is not the standard infinite-population ESS comparison because the
    resident and mutant fitnesses are conditioned on different population
    sampling events in the well-mixed model.
    """

    k: int
    gamma: float
    resident: EWLStrategy
    mutant: EWLStrategy
    mutant_position: int
    resident_average_payoff: float
    mutant_payoff: float
    mutant_minus_group_resident: float
    probability_error: float


def canonicalize_ewl_strategy(strategy: EWLStrategy) -> EWLStrategy:
    """Validate the restricted EWL domain and collapse the duplicated D edge."""
    theta = _canonical_domain_value(strategy.theta, 0.0, math.pi, "theta")
    phi = _canonical_domain_value(strategy.phi, 0.0, math.pi / 2.0, "phi")
    if abs(theta - math.pi) <= ANGLE_ENDPOINT_TOLERANCE:
        theta = math.pi
        phi = 0.0
    return EWLStrategy(theta=theta, phi=phi, label=strategy.label)


def deduplicate_ewl_strategies(
    strategies: Iterable[EWLStrategy],
) -> Tuple[EWLStrategy, ...]:
    """Return canonical strategies with exact coordinate duplicates removed."""
    unique = []
    seen = set()
    for strategy in strategies:
        canonical = canonicalize_ewl_strategy(strategy)
        key = (canonical.theta, canonical.phi)
        if key in seen:
            continue
        seen.add(key)
        unique.append(canonical)
    return tuple(unique)


def ewl_strategies_equivalent(
    first: EWLStrategy,
    second: EWLStrategy,
) -> bool:
    """Return whether two restricted-domain coordinates denote the same strategy."""
    first_canonical = canonicalize_ewl_strategy(first)
    second_canonical = canonicalize_ewl_strategy(second)
    return (
        abs(first_canonical.theta - second_canonical.theta)
        <= ANGLE_ENDPOINT_TOLERANCE
        and abs(first_canonical.phi - second_canonical.phi)
        <= ANGLE_ENDPOINT_TOLERANCE
    )


def population_fitness(
    composition_payoffs: Sequence[float],
    epsilon: float,
) -> float:
    """Return binomially averaged focal fitness at mutant frequency epsilon."""
    _validate_frequency(epsilon, "epsilon")
    if not composition_payoffs:
        raise ValueError("at least one composition payoff is required")
    n = len(composition_payoffs) - 1
    terms = (
        math.comb(n, j)
        * (epsilon ** j)
        * ((1.0 - epsilon) ** (n - j))
        * float(payoff)
        for j, payoff in enumerate(composition_payoffs)
    )
    return float(math.fsum(terms))


def bernstein_to_power_coefficients(
    bernstein_values: Sequence[float],
) -> Tuple[float, ...]:
    """Convert degree-n Bernstein values into coefficients about epsilon=0."""
    if not bernstein_values:
        raise ValueError("at least one Bernstein value is required")
    n = len(bernstein_values) - 1
    coefficients = []
    for order in range(n + 1):
        finite_difference = math.fsum(
            ((-1) ** (order - j))
            * math.comb(order, j)
            * float(bernstein_values[j])
            for j in range(order + 1)
        )
        coefficients.append(float(math.comb(n, order) * finite_difference))
    return tuple(coefficients)


def evaluate_power_polynomial(coefficients: Sequence[float], epsilon: float) -> float:
    """Evaluate a power-basis polynomial using Horner's rule."""
    _validate_frequency(epsilon, "epsilon")
    value = 0.0
    for coefficient in reversed(tuple(coefficients)):
        value = value * epsilon + float(coefficient)
    return float(value)


def classify_power_coefficients(
    coefficients: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
    reliability_error: float = 0.0,
) -> Tuple[str, Optional[int], float]:
    """Classify the rare-mutant limit by its first reliable coefficient."""
    _validate_tolerance(tolerance)
    if reliability_error < 0.0 or not math.isfinite(reliability_error):
        raise ValueError("reliability_error must be finite and nonnegative")
    if reliability_error > tolerance:
        return (INDETERMINATE, None, 0.0)
    for order, coefficient in enumerate(coefficients):
        value = float(coefficient)
        if not math.isfinite(value):
            return (INDETERMINATE, None, 0.0)
        if abs(value) <= tolerance:
            continue
        classification = RARE_MUTANT_INVADES if value > 0.0 else RARE_MUTANT_REJECTED
        return (classification, order, value)
    return (NEUTRAL, None, 0.0)


def analyze_resident_mutant(
    game: KPlayerEWLGame,
    resident: EWLStrategy,
    mutant: EWLStrategy,
    epsilon_reference: float,
    tolerance: float = DEFAULT_TOLERANCE,
    diagnostic_frequencies: Sequence[float] = DEFAULT_DIAGNOSTIC_FREQUENCIES,
    positions: Optional[Sequence[int]] = None,
) -> ResidentMutantAnalysis:
    """Evaluate one resident-mutant pair under random well-mixed grouping.

    For each focal position and each j in 0,...,K-1, this function averages over
    every placement of j mutant co-players. In the symmetric baseline those
    conditional values should agree to floating-point precision.
    """
    _validate_tolerance(tolerance)
    _validate_frequency(epsilon_reference, "epsilon_reference")
    frequencies = tuple(float(value) for value in diagnostic_frequencies)
    if not frequencies:
        raise ValueError("at least one diagnostic frequency is required")
    for value in frequencies:
        _validate_frequency(value, "diagnostic frequency")

    resident_canonical = canonicalize_ewl_strategy(resident)
    mutant_canonical = canonicalize_ewl_strategy(mutant)
    focal_positions = _validate_positions(game.k, positions)
    raw_results = tuple(
        _analyze_position(
            game=game,
            resident=resident_canonical,
            mutant=mutant_canonical,
            mutant_position=position,
            epsilon_reference=epsilon_reference,
            diagnostic_frequencies=frequencies,
            tolerance=tolerance,
        )
        for position in focal_positions
    )

    permutation_error = _position_permutation_error(raw_results)
    results = []
    for result in raw_results:
        reliability_error = max(
            permutation_error,
            result.probability_error,
            result.polynomial_reconstruction_error,
        )
        classification, order, coefficient = classify_power_coefficients(
            result.power_coefficients,
            tolerance=tolerance,
            reliability_error=reliability_error,
        )
        uncertainty = (
            "exact_statevector; "
            f"permutation_error={permutation_error:.17g}; "
            f"probability_error={result.probability_error:.17g}; "
            "sampling_confidence_interval=not_applicable"
        )
        results.append(
            replace(
                result,
                leading_nonzero_order=order,
                leading_coefficient=coefficient,
                rare_mutant_classification=classification,
                permutation_error=permutation_error,
                uncertainty_or_confidence_interval=uncertainty,
            )
        )

    classifications = {result.rare_mutant_classification for result in results}
    pair_classification = (
        classifications.pop() if len(classifications) == 1 else INDETERMINATE
    )
    return ResidentMutantAnalysis(
        k=game.k,
        gamma=game.gamma,
        resident=resident_canonical,
        mutant=mutant_canonical,
        position_results=tuple(results),
        rare_mutant_classification=pair_classification,
        permutation_error=permutation_error,
    )


def exactly_one_mutant_group_diagnostic(
    game: KPlayerEWLGame,
    resident: EWLStrategy,
    mutant: EWLStrategy,
    mutant_position: int = 0,
) -> ExactlyOneMutantGroupDiagnostic:
    """Compare types inside one fixed K-player group containing one mutant."""
    if not 0 <= mutant_position < game.k:
        raise ValueError(f"mutant_position out of range: {mutant_position!r}")
    resident_canonical = canonicalize_ewl_strategy(resident)
    mutant_canonical = canonicalize_ewl_strategy(mutant)
    profile = [resident_canonical] * game.k
    profile[mutant_position] = mutant_canonical
    result = game.run(profile)
    resident_payoffs = tuple(
        payoff
        for position, payoff in enumerate(result.expected_payoffs)
        if position != mutant_position
    )
    resident_average = float(math.fsum(resident_payoffs) / len(resident_payoffs))
    mutant_payoff = float(result.expected_payoffs[mutant_position])
    return ExactlyOneMutantGroupDiagnostic(
        k=game.k,
        gamma=game.gamma,
        resident=resident_canonical,
        mutant=mutant_canonical,
        mutant_position=mutant_position,
        resident_average_payoff=resident_average,
        mutant_payoff=mutant_payoff,
        mutant_minus_group_resident=mutant_payoff - resident_average,
        probability_error=abs(result.probability_sum - 1.0),
    )


def _analyze_position(
    game: KPlayerEWLGame,
    resident: EWLStrategy,
    mutant: EWLStrategy,
    mutant_position: int,
    epsilon_reference: float,
    diagnostic_frequencies: Tuple[float, ...],
    tolerance: float,
) -> PopulationInvasionResult:
    other_positions = tuple(
        position for position in range(game.k) if position != mutant_position
    )
    composition_rows = []
    for mutant_co_players in range(game.k):
        resident_samples = []
        mutant_samples = []
        probability_errors = []
        for mutant_positions in combinations(other_positions, mutant_co_players):
            resident_profile = [resident] * game.k
            mutant_profile = [resident] * game.k
            mutant_profile[mutant_position] = mutant
            for position in mutant_positions:
                resident_profile[position] = mutant
                mutant_profile[position] = mutant

            resident_result = game.run(resident_profile)
            mutant_result = game.run(mutant_profile)
            resident_samples.append(resident_result.expected_payoffs[mutant_position])
            mutant_samples.append(mutant_result.expected_payoffs[mutant_position])
            probability_errors.extend(
                (
                    abs(resident_result.probability_sum - 1.0),
                    abs(mutant_result.probability_sum - 1.0),
                )
            )

        composition_rows.append(
            FocalCompositionPayoff(
                mutant_co_players=mutant_co_players,
                resident_payoff=float(
                    math.fsum(resident_samples) / len(resident_samples)
                ),
                mutant_payoff=float(math.fsum(mutant_samples) / len(mutant_samples)),
                resident_positional_spread=_spread(resident_samples),
                mutant_positional_spread=_spread(mutant_samples),
                max_probability_error=max(probability_errors, default=0.0),
                conditional_profile_count=len(resident_samples),
            )
        )

    resident_values = tuple(row.resident_payoff for row in composition_rows)
    mutant_values = tuple(row.mutant_payoff for row in composition_rows)
    bernstein_advantages = tuple(
        mutant_payoff - resident_payoff
        for resident_payoff, mutant_payoff in zip(resident_values, mutant_values)
    )
    power_coefficients = bernstein_to_power_coefficients(bernstein_advantages)
    resident_reference = population_fitness(resident_values, epsilon_reference)
    mutant_reference = population_fitness(mutant_values, epsilon_reference)
    advantage_reference = mutant_reference - resident_reference
    diagnostics = tuple(
        _frequency_diagnostic(resident_values, mutant_values, epsilon)
        for epsilon in diagnostic_frequencies
    )
    reconstruction_error = max(
        abs(
            diagnostic.mutant_advantage
            - evaluate_power_polynomial(power_coefficients, diagnostic.epsilon)
        )
        for diagnostic in diagnostics
    )
    probability_error = max(
        (row.max_probability_error for row in composition_rows),
        default=0.0,
    )

    # Final classification is assigned after results from every requested
    # position are available and their permutation error can be measured.
    return PopulationInvasionResult(
        k=game.k,
        gamma=game.gamma,
        resident=resident,
        mutant=mutant,
        mutant_position=mutant_position,
        epsilon_reference=epsilon_reference,
        resident_payoff_reference=resident_reference,
        mutant_payoff_reference=mutant_reference,
        mutant_advantage_reference=advantage_reference,
        composition_payoffs=tuple(composition_rows),
        bernstein_advantages=bernstein_advantages,
        power_coefficients=power_coefficients,
        leading_nonzero_order=None,
        leading_coefficient=0.0,
        rare_mutant_classification=INDETERMINATE,
        invades_at_epsilon_reference=advantage_reference > tolerance,
        diagnostic_frequencies=diagnostics,
        tolerance=tolerance,
        permutation_error=0.0,
        probability_error=probability_error,
        polynomial_reconstruction_error=reconstruction_error,
        uncertainty_or_confidence_interval="pending_position_audit",
    )


def _frequency_diagnostic(
    resident_values: Sequence[float],
    mutant_values: Sequence[float],
    epsilon: float,
) -> FrequencyDiagnostic:
    resident_payoff = population_fitness(resident_values, epsilon)
    mutant_payoff = population_fitness(mutant_values, epsilon)
    return FrequencyDiagnostic(
        epsilon=epsilon,
        resident_payoff=resident_payoff,
        mutant_payoff=mutant_payoff,
        mutant_advantage=mutant_payoff - resident_payoff,
    )


def _position_permutation_error(
    results: Sequence[PopulationInvasionResult],
) -> float:
    errors = []
    for composition_index in range(results[0].k):
        resident_values = [
            result.composition_payoffs[composition_index].resident_payoff
            for result in results
        ]
        mutant_values = [
            result.composition_payoffs[composition_index].mutant_payoff
            for result in results
        ]
        errors.extend((_spread(resident_values), _spread(mutant_values)))
        errors.extend(
            result.composition_payoffs[composition_index].resident_positional_spread
            for result in results
        )
        errors.extend(
            result.composition_payoffs[composition_index].mutant_positional_spread
            for result in results
        )
    return max(errors, default=0.0)


def _validate_positions(k: int, positions: Optional[Sequence[int]]) -> Tuple[int, ...]:
    values = tuple(range(k)) if positions is None else tuple(positions)
    if not values:
        raise ValueError("at least one focal position is required")
    if len(set(values)) != len(values):
        raise ValueError(f"positions must be unique; got {values!r}")
    if any(position < 0 or position >= k for position in values):
        raise ValueError(f"position outside K={k} state space: {values!r}")
    return values


def _canonical_domain_value(
    value: float,
    lower: float,
    upper: float,
    name: str,
) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    outside_domain = (
        value < lower - ANGLE_ENDPOINT_TOLERANCE
        or value > upper + ANGLE_ENDPOINT_TOLERANCE
    )
    if outside_domain:
        raise ValueError(f"{name} must be in [{lower}, {upper}]; got {value!r}")
    if abs(value - lower) <= ANGLE_ENDPOINT_TOLERANCE:
        return float(lower)
    if abs(value - upper) <= ANGLE_ENDPOINT_TOLERANCE:
        return float(upper)
    return float(value)


def _spread(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _validate_frequency(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")


def _validate_tolerance(tolerance: float) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and nonnegative; got {tolerance!r}")
