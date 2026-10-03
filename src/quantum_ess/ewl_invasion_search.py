"""Grid and continuous mutant searches for K-player EWL invasion analysis."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .ewl import EWLStrategy
from .ewl_invasion import (
    DEFAULT_DIAGNOSTIC_FREQUENCIES,
    DEFAULT_TOLERANCE,
    INDETERMINATE,
    NEUTRAL,
    RARE_MUTANT_INVADES,
    ResidentMutantAnalysis,
    analyze_resident_mutant,
    canonicalize_ewl_strategy,
    deduplicate_ewl_strategies,
    ewl_strategies_equivalent,
)
from .k_player_ewl import KPlayerEWLGame
from .k_player_ewl_grid import ewl_strategy_grid

NOT_ESS = "not_ESS"
GRID_ESS_CANDIDATE = "grid_ESS_candidate"
NUMERICALLY_SUPPORTED_ESS = "numerically_supported_ESS"
NEUTRAL_OR_WEAK_CANDIDATE = "neutral_or_weak_candidate"


@dataclass(frozen=True)
class SymmetricNashCandidate:
    """One symmetric pure Nash candidate on a finite strategy grid."""

    k: int
    gamma: float
    resident: EWLStrategy
    resident_payoff: float
    maximum_unilateral_gain: float
    best_response_mutants: Tuple[EWLStrategy, ...]
    grid_strategy_count: int


@dataclass(frozen=True)
class MutantGridSearchResult:
    """Complete resident-versus-mutant results for one finite grid stage."""

    stage: str
    k: int
    gamma: float
    resident: EWLStrategy
    mutant_strategy_count: int
    pair_results: Tuple[ResidentMutantAnalysis, ...]
    resident_status: str
    maximum_mutant_advantage_at_epsilon_reference: float
    strongest_mutant: EWLStrategy
    strongest_mutant_position: int
    number_of_robust_invaders: int
    number_of_neutral_mutants: int
    number_of_indeterminate_mutants: int
    maximum_permutation_error: float
    tolerance: float

    @property
    def strongest_pair(self) -> ResidentMutantAnalysis:
        """Return the mutant pair with the largest reference-frequency advantage."""
        return max(
            self.pair_results,
            key=lambda pair: pair.worst_reference_result.mutant_advantage_reference,
        )


@dataclass(frozen=True)
class ContinuousOptimizerRun:
    """One global-then-local bounded optimization objective."""

    objective_epsilon: float
    differential_evolution_success: bool
    differential_evolution_message: str
    differential_evolution_iterations: int
    differential_evolution_evaluations: int
    best_theta: float
    best_phi: float
    best_objective_advantage: float
    local_start_count: int


@dataclass(frozen=True)
class ContinuousMutantSearchResult:
    """Numerical continuous-domain validation for one resident."""

    k: int
    gamma: float
    resident: EWLStrategy
    seed: int
    objective_frequencies: Tuple[float, ...]
    optimizer_runs: Tuple[ContinuousOptimizerRun, ...]
    candidate_results: Tuple[ResidentMutantAnalysis, ...]
    resident_status: str
    maximum_mutant_advantage_at_epsilon_reference: float
    strongest_mutant: EWLStrategy
    strongest_mutant_position: int
    number_of_robust_invaders: int
    number_of_neutral_mutants: int
    maximum_permutation_error: float
    all_global_runs_converged: bool
    tolerance: float


@dataclass(frozen=True)
class _ContinuousObjective:
    k: int
    gamma: float
    resident: EWLStrategy
    epsilon: float
    tolerance: float

    def __call__(self, angles: Sequence[float]) -> float:
        mutant = EWLStrategy(theta=float(angles[0]), phi=float(angles[1]))
        game = KPlayerEWLGame(k=self.k, gamma=self.gamma)
        analysis = analyze_resident_mutant(
            game=game,
            resident=self.resident,
            mutant=mutant,
            epsilon_reference=self.epsilon,
            tolerance=self.tolerance,
            diagnostic_frequencies=(self.epsilon,),
            positions=(0,),
        )
        return -analysis.worst_reference_result.mutant_advantage_reference


def symmetric_nash_candidates_on_grid(
    k: int,
    gamma: float,
    strategies: Sequence[EWLStrategy],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[SymmetricNashCandidate, ...]:
    """Return canonical symmetric Nash candidates on a finite mutant grid."""
    strategy_set = deduplicate_ewl_strategies(strategies)
    game = KPlayerEWLGame(k=k, gamma=gamma)
    candidates = []
    for resident in strategy_set:
        resident_payoff = game.expected_payoffs((resident,) * k)[0]
        mutant_rows = []
        for mutant in strategy_set:
            profile = (mutant,) + (resident,) * (k - 1)
            gain = game.expected_payoffs(profile)[0] - resident_payoff
            mutant_rows.append((gain, mutant))
        maximum_gain = max(gain for gain, _mutant in mutant_rows)
        if maximum_gain > tolerance:
            continue
        best_mutants = tuple(
            mutant
            for gain, mutant in mutant_rows
            if abs(gain - maximum_gain) <= tolerance
        )
        candidates.append(
            SymmetricNashCandidate(
                k=k,
                gamma=gamma,
                resident=resident,
                resident_payoff=float(resident_payoff),
                maximum_unilateral_gain=float(maximum_gain),
                best_response_mutants=best_mutants,
                grid_strategy_count=len(strategy_set),
            )
        )
    return tuple(candidates)


def refined_mutant_grid(
    theta_count: int,
    phi_count: int,
    centers: Sequence[EWLStrategy] = (),
    local_levels: int = 2,
) -> Tuple[EWLStrategy, ...]:
    """Build a full-domain grid augmented near selected points and boundaries."""
    if theta_count < 2 or phi_count < 2:
        raise ValueError("refined grids require at least two values on each axis")
    if local_levels < 0:
        raise ValueError("local_levels must be nonnegative")

    base = list(
        ewl_strategy_grid(
            theta_count=theta_count,
            phi_count=phi_count,
            label_prefix="F",
        ).strategies
    )
    theta_step = math.pi / (theta_count - 1)
    phi_step = (math.pi / 2.0) / (phi_count - 1)
    canonical_centers = tuple(canonicalize_ewl_strategy(center) for center in centers)

    for center in canonical_centers:
        for level in range(1, local_levels + 1):
            theta_delta = theta_step / (2 ** level)
            phi_delta = phi_step / (2 ** level)
            for theta_offset in (-theta_delta, 0.0, theta_delta):
                for phi_offset in (-phi_delta, 0.0, phi_delta):
                    theta = min(math.pi, max(0.0, center.theta + theta_offset))
                    phi = min(math.pi / 2.0, max(0.0, center.phi + phi_offset))
                    base.append(EWLStrategy(theta=theta, phi=phi, label="local"))

    # Increase resolution along all four domain edges. The two phi edges are
    # boundaries, not periodic aliases, in the restricted EWL strategy set.
    edge_count = max(theta_count, 2 * phi_count - 1)
    for index in range(edge_count):
        fraction = index / (edge_count - 1)
        theta = fraction * math.pi
        phi = fraction * math.pi / 2.0
        base.extend(
            (
                EWLStrategy(theta=theta, phi=0.0, label="edge"),
                EWLStrategy(theta=theta, phi=math.pi / 2.0, label="edge"),
                EWLStrategy(theta=0.0, phi=phi, label="edge"),
                EWLStrategy(theta=math.pi, phi=phi, label="edge"),
            )
        )
    return deduplicate_ewl_strategies(base)


def search_mutants_on_grid(
    k: int,
    gamma: float,
    resident: EWLStrategy,
    mutant_strategies: Sequence[EWLStrategy],
    epsilon_reference: float,
    stage: str,
    tolerance: float = DEFAULT_TOLERANCE,
    diagnostic_frequencies: Sequence[float] = DEFAULT_DIAGNOSTIC_FREQUENCIES,
    positions: Optional[Sequence[int]] = None,
    workers: int = 1,
    chunksize: int = 8,
) -> MutantGridSearchResult:
    """Test one resident against every distinct non-self mutant on a grid."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if chunksize < 1:
        raise ValueError("chunksize must be at least one")
    resident_canonical = canonicalize_ewl_strategy(resident)
    mutants = tuple(
        mutant
        for mutant in deduplicate_ewl_strategies(mutant_strategies)
        if not ewl_strategies_equivalent(resident_canonical, mutant)
    )
    if not mutants:
        raise ValueError("the mutant grid contains no distinct non-self strategies")
    position_tuple = tuple(range(k)) if positions is None else tuple(positions)
    frequencies = tuple(float(value) for value in diagnostic_frequencies)
    tasks = tuple(
        (
            k,
            gamma,
            resident_canonical,
            mutant,
            epsilon_reference,
            tolerance,
            frequencies,
            position_tuple,
        )
        for mutant in mutants
    )

    if workers == 1:
        pair_results = tuple(_analyze_grid_mutant(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pair_results = tuple(
                executor.map(_analyze_grid_mutant, tasks, chunksize=chunksize)
            )
    return _summarize_grid_search(
        stage=stage,
        resident=resident_canonical,
        pair_results=pair_results,
        tolerance=tolerance,
    )


def continuous_mutant_search(
    k: int,
    gamma: float,
    resident: EWLStrategy,
    epsilon_reference: float,
    seed_strategies: Sequence[EWLStrategy] = (),
    tolerance: float = DEFAULT_TOLERANCE,
    diagnostic_frequencies: Sequence[float] = DEFAULT_DIAGNOSTIC_FREQUENCIES,
    objective_frequencies: Sequence[float] = (0.0, 1e-6, 1e-4, 1e-2),
    seed: int = 20260826,
    differential_maxiter: int = 80,
    differential_popsize: int = 12,
    local_start_limit: int = 10,
    optimizer_tolerance: float = 1e-10,
    workers: int = 1,
) -> ContinuousMutantSearchResult:
    """Search the continuous mutant domain with global and local optimizers.

    Separate objectives maximize DeltaPi at several rare frequencies. Every
    optimizer candidate is then classified from the full rare-limit polynomial,
    so these scalar objectives are search devices rather than ESS definitions.
    """
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "continuous search requires the optional research dependencies; "
            "install with `python -m pip install -e '.[research]'`"
        ) from exc

    if differential_maxiter < 1 or differential_popsize < 2:
        raise ValueError(
            "differential evolution iteration and population limits are too small"
        )
    if local_start_limit < 1:
        raise ValueError("local_start_limit must be positive")
    resident_canonical = canonicalize_ewl_strategy(resident)
    frequencies = tuple(sorted(set(float(value) for value in objective_frequencies)))
    if epsilon_reference not in frequencies:
        frequencies = tuple(sorted(frequencies + (float(epsilon_reference),)))

    base_starts = list(deduplicate_ewl_strategies(seed_strategies))
    base_starts.extend(
        (
            resident_canonical,
            EWLStrategy(0.0, 0.0, "corner"),
            EWLStrategy(0.0, math.pi / 2.0, "corner"),
            EWLStrategy(math.pi, 0.0, "corner"),
            EWLStrategy(math.pi / 2.0, 0.0, "midpoint"),
            EWLStrategy(math.pi / 2.0, math.pi / 2.0, "midpoint"),
        )
    )
    base_starts = list(deduplicate_ewl_strategies(base_starts))
    bounds = ((0.0, math.pi), (0.0, math.pi / 2.0))
    candidate_points: List[Tuple[float, float]] = [
        (strategy.theta, strategy.phi) for strategy in base_starts
    ]
    optimizer_runs = []

    for objective_index, epsilon in enumerate(frequencies):
        objective = _ContinuousObjective(
            k=k,
            gamma=gamma,
            resident=resident_canonical,
            epsilon=epsilon,
            tolerance=tolerance,
        )
        global_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed + objective_index,
            maxiter=differential_maxiter,
            popsize=differential_popsize,
            tol=optimizer_tolerance,
            atol=optimizer_tolerance,
            polish=False,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
        candidate_points.append((float(global_result.x[0]), float(global_result.x[1])))

        ranked_starts = sorted(
            base_starts,
            key=lambda strategy: objective((strategy.theta, strategy.phi)),
        )
        starts = [
            (float(global_result.x[0]), float(global_result.x[1]))
        ] + [
            (strategy.theta, strategy.phi)
            for strategy in ranked_starts[: max(0, local_start_limit - 1)]
        ]
        local_results = []
        for start in starts:
            local = minimize(
                objective,
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "ftol": optimizer_tolerance,
                    "gtol": optimizer_tolerance,
                    "maxiter": 500,
                    "maxfun": 5000,
                },
            )
            local_results.append(local)
            candidate_points.append((float(local.x[0]), float(local.x[1])))
        best_local = min(local_results, key=lambda result: float(result.fun))
        candidate_points.append((float(best_local.x[0]), float(best_local.x[1])))
        optimizer_runs.append(
            ContinuousOptimizerRun(
                objective_epsilon=epsilon,
                differential_evolution_success=bool(global_result.success),
                differential_evolution_message=str(global_result.message),
                differential_evolution_iterations=int(global_result.nit),
                differential_evolution_evaluations=int(global_result.nfev),
                best_theta=float(best_local.x[0]),
                best_phi=float(best_local.x[1]),
                best_objective_advantage=-float(best_local.fun),
                local_start_count=len(starts),
            )
        )

    candidate_strategies = deduplicate_ewl_strategies(
        EWLStrategy(theta=theta, phi=phi, label="continuous")
        for theta, phi in candidate_points
    )
    distinct_candidates = tuple(
        candidate
        for candidate in candidate_strategies
        if not ewl_strategies_equivalent(candidate, resident_canonical)
    )
    if not distinct_candidates:
        # Retain the strongest supplied finite-grid seed as a non-self audit
        # point if every optimizer converges exactly to the resident.
        distinct_candidates = tuple(
            candidate
            for candidate in deduplicate_ewl_strategies(seed_strategies)
            if not ewl_strategies_equivalent(candidate, resident_canonical)
        )
    if not distinct_candidates:
        raise RuntimeError("continuous search produced no distinct mutant candidates")

    game = KPlayerEWLGame(k=k, gamma=gamma)
    candidate_results = tuple(
        analyze_resident_mutant(
            game=game,
            resident=resident_canonical,
            mutant=mutant,
            epsilon_reference=epsilon_reference,
            tolerance=tolerance,
            diagnostic_frequencies=diagnostic_frequencies,
        )
        for mutant in distinct_candidates
    )
    return _summarize_continuous_search(
        k=k,
        gamma=gamma,
        resident=resident_canonical,
        seed=seed,
        objective_frequencies=frequencies,
        optimizer_runs=tuple(optimizer_runs),
        candidate_results=candidate_results,
        tolerance=tolerance,
    )


def _analyze_grid_mutant(task: Tuple[object, ...]) -> ResidentMutantAnalysis:
    (
        k,
        gamma,
        resident,
        mutant,
        epsilon_reference,
        tolerance,
        frequencies,
        positions,
    ) = task
    game = KPlayerEWLGame(k=int(k), gamma=float(gamma))
    return analyze_resident_mutant(
        game=game,
        resident=resident,  # type: ignore[arg-type]
        mutant=mutant,  # type: ignore[arg-type]
        epsilon_reference=float(epsilon_reference),
        tolerance=float(tolerance),
        diagnostic_frequencies=frequencies,  # type: ignore[arg-type]
        positions=positions,  # type: ignore[arg-type]
    )


def _summarize_grid_search(
    stage: str,
    resident: EWLStrategy,
    pair_results: Tuple[ResidentMutantAnalysis, ...],
    tolerance: float,
) -> MutantGridSearchResult:
    strongest = max(
        pair_results,
        key=lambda pair: pair.worst_reference_result.mutant_advantage_reference,
    )
    invaders = sum(
        pair.rare_mutant_classification == RARE_MUTANT_INVADES
        for pair in pair_results
    )
    neutral = sum(pair.rare_mutant_classification == NEUTRAL for pair in pair_results)
    indeterminate = sum(
        pair.rare_mutant_classification == INDETERMINATE for pair in pair_results
    )
    status = _finite_search_status(invaders, neutral, indeterminate)
    worst = strongest.worst_reference_result
    return MutantGridSearchResult(
        stage=stage,
        k=strongest.k,
        gamma=strongest.gamma,
        resident=resident,
        mutant_strategy_count=len(pair_results),
        pair_results=pair_results,
        resident_status=status,
        maximum_mutant_advantage_at_epsilon_reference=(
            worst.mutant_advantage_reference
        ),
        strongest_mutant=strongest.mutant,
        strongest_mutant_position=worst.mutant_position,
        number_of_robust_invaders=invaders,
        number_of_neutral_mutants=neutral,
        number_of_indeterminate_mutants=indeterminate,
        maximum_permutation_error=max(
            pair.permutation_error for pair in pair_results
        ),
        tolerance=tolerance,
    )


def _summarize_continuous_search(
    k: int,
    gamma: float,
    resident: EWLStrategy,
    seed: int,
    objective_frequencies: Tuple[float, ...],
    optimizer_runs: Tuple[ContinuousOptimizerRun, ...],
    candidate_results: Tuple[ResidentMutantAnalysis, ...],
    tolerance: float,
) -> ContinuousMutantSearchResult:
    strongest = max(
        candidate_results,
        key=lambda pair: pair.worst_reference_result.mutant_advantage_reference,
    )
    invaders = sum(
        pair.rare_mutant_classification == RARE_MUTANT_INVADES
        for pair in candidate_results
    )
    neutral = sum(
        pair.rare_mutant_classification == NEUTRAL for pair in candidate_results
    )
    indeterminate = sum(
        pair.rare_mutant_classification == INDETERMINATE
        for pair in candidate_results
    )
    all_converged = all(run.differential_evolution_success for run in optimizer_runs)
    if invaders:
        status = NOT_ESS
    elif indeterminate or not all_converged:
        status = INDETERMINATE
    elif neutral:
        status = NEUTRAL_OR_WEAK_CANDIDATE
    else:
        status = NUMERICALLY_SUPPORTED_ESS
    worst = strongest.worst_reference_result
    return ContinuousMutantSearchResult(
        k=k,
        gamma=gamma,
        resident=resident,
        seed=seed,
        objective_frequencies=objective_frequencies,
        optimizer_runs=optimizer_runs,
        candidate_results=candidate_results,
        resident_status=status,
        maximum_mutant_advantage_at_epsilon_reference=(
            worst.mutant_advantage_reference
        ),
        strongest_mutant=strongest.mutant,
        strongest_mutant_position=worst.mutant_position,
        number_of_robust_invaders=invaders,
        number_of_neutral_mutants=neutral,
        maximum_permutation_error=max(
            pair.permutation_error for pair in candidate_results
        ),
        all_global_runs_converged=all_converged,
        tolerance=tolerance,
    )


def _finite_search_status(invaders: int, neutral: int, indeterminate: int) -> str:
    if invaders:
        return NOT_ESS
    if indeterminate:
        return INDETERMINATE
    if neutral:
        return NEUTRAL_OR_WEAK_CANDIDATE
    return GRID_ESS_CANDIDATE
