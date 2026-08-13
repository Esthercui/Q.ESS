"""Strategy-grid search tools for K-player EWL games."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import permutations, product
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .ewl import EWLStrategy
from .k_player_ewl import KPlayerEWLGame, SUPPORTED_K_VALUES

SparseCountProfile = Tuple[Tuple[int, int], ...]
BestResponseTable = Dict[SparseCountProfile, Tuple[float, Tuple[int, ...]]]

_WORKER_K: Optional[int] = None
_WORKER_GAMMA: Optional[float] = None
_WORKER_STRATEGIES: Optional[Tuple[EWLStrategy, ...]] = None
_WORKER_TOLERANCE: float = 1e-8


@dataclass(frozen=True)
class EWLStrategyGrid:
    """Discrete grid of EWL strategies U(theta, phi)."""

    strategies: Tuple[EWLStrategy, ...]
    theta_values: Tuple[float, ...]
    phi_values: Tuple[float, ...]

    def __len__(self) -> int:
        return len(self.strategies)

    def __getitem__(self, index: int) -> EWLStrategy:
        return self.strategies[index]

    def index_angles(self, index: int) -> Tuple[float, float]:
        strategy = self.strategies[index]
        return (strategy.theta, strategy.phi)


@dataclass(frozen=True)
class KPlayerNashEquilibrium:
    """One ordered pure-strategy Nash equilibrium on a finite EWL grid."""

    strategy_indices: Tuple[int, ...]
    strategy_counts: SparseCountProfile
    strategies: Tuple[EWLStrategy, ...]
    payoffs: Tuple[float, ...]
    best_response_indices: Tuple[Tuple[int, ...], ...]
    unilateral_gains: Tuple[float, ...]
    probability_sum: float
    most_likely_outcome: Tuple[int, ...]
    most_likely_probability: float

    @property
    def is_symmetric(self) -> bool:
        return len(self.strategy_counts) == 1

    @property
    def is_nash(self) -> bool:
        return True

    @property
    def average_pairwise_payoffs(self) -> Tuple[float, ...]:
        """Return each total payoff divided by the number of opponents."""
        opponent_count = len(self.strategies) - 1
        return tuple(payoff / opponent_count for payoff in self.payoffs)

    def angles(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((strategy.theta, strategy.phi) for strategy in self.strategies)


@dataclass(frozen=True)
class KPlayerProfileEvaluation:
    """Nash-deviation audit for one ordered K-player strategy profile."""

    strategy_indices: Tuple[int, ...]
    strategies: Tuple[EWLStrategy, ...]
    payoffs: Tuple[float, ...]
    best_response_indices: Tuple[Tuple[int, ...], ...]
    unilateral_gains: Tuple[float, ...]
    is_nash: bool
    probability_sum: float
    most_likely_outcome: Tuple[int, ...]
    most_likely_probability: float

    @property
    def average_pairwise_payoffs(self) -> Tuple[float, ...]:
        """Return each total payoff divided by the number of opponents."""
        opponent_count = len(self.strategies) - 1
        return tuple(payoff / opponent_count for payoff in self.payoffs)

    def angles(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((strategy.theta, strategy.phi) for strategy in self.strategies)


@dataclass(frozen=True)
class KPlayerNashSearchResult:
    """Result of an exact finite-grid pure Nash search."""

    k: int
    gamma: float
    strategy_count: int
    ordered_profile_count: int
    opponent_context_count: int
    resident_count_profile_count: int
    equilibria: Tuple[KPlayerNashEquilibrium, ...]
    tolerance: float
    truncated: bool


def ewl_strategy_grid(
    theta_count: int = 11,
    phi_count: int = 6,
    theta_min: float = 0.0,
    theta_max: float = math.pi,
    phi_min: float = 0.0,
    phi_max: float = math.pi / 2.0,
    label_prefix: str = "G",
) -> EWLStrategyGrid:
    """Return a rectangular grid over the EWL strategy space.

    The default grid matches the earlier notebook scale: 11 theta values and 6
    phi values, giving 66 pure quantum strategies per player.
    """
    _validate_grid_count(theta_count, "theta_count")
    _validate_grid_count(phi_count, "phi_count")
    _validate_finite_range(theta_min, theta_max, "theta")
    _validate_finite_range(phi_min, phi_max, "phi")

    theta_values = _linspace(theta_min, theta_max, theta_count)
    phi_values = _linspace(phi_min, phi_max, phi_count)
    strategies = tuple(
        EWLStrategy(theta=theta, phi=phi, label=f"{label_prefix}{index}")
        for index, (theta, phi) in enumerate(product(theta_values, phi_values))
    )
    return EWLStrategyGrid(
        strategies=strategies,
        theta_values=theta_values,
        phi_values=phi_values,
    )


def ordered_profile_count(strategy_count: int, k: int) -> int:
    """Return the number N^K of ordered profiles in the finite game."""
    _validate_strategy_count(strategy_count)
    _validate_k(k)
    return strategy_count ** k


def iter_ordered_profile_indices(strategy_count: int, k: int) -> Iterator[Tuple[int, ...]]:
    """Yield every ordered K-player grid profile as strategy indices."""
    _validate_strategy_count(strategy_count)
    _validate_k(k)
    yield from product(range(strategy_count), repeat=k)


def evaluate_profile_on_grid(
    k: int,
    gamma: float,
    strategies: Sequence[EWLStrategy],
    strategy_indices: Sequence[int],
    tolerance: float = 1e-8,
) -> KPlayerProfileEvaluation:
    """Evaluate one ordered profile and every unilateral grid deviation.

    A profile is Nash exactly when no player can improve its own expected
    payoff by more than ``tolerance`` while all other players stay fixed.
    """
    _validate_k(k)
    _validate_search_inputs(gamma, strategies, tolerance, workers=1, chunksize=1)
    strategy_tuple = tuple(strategies)
    indices = tuple(strategy_indices)
    if len(indices) != k:
        raise ValueError(f"expected {k} strategy indices, got {len(indices)}")
    if any(index < 0 or index >= len(strategy_tuple) for index in indices):
        raise ValueError(f"strategy index outside grid: {indices!r}")

    game = KPlayerEWLGame(k=k, gamma=gamma)
    profile = tuple(strategy_tuple[index] for index in indices)
    result = game.run(profile)
    best_response_rows: List[Tuple[int, ...]] = []
    gains: List[float] = []

    for player in range(k):
        best_value = -math.inf
        best_indices: List[int] = []
        for candidate_index, candidate in enumerate(strategy_tuple):
            deviated_profile = list(profile)
            deviated_profile[player] = candidate
            candidate_payoff = game.expected_payoffs(deviated_profile)[player]
            if candidate_payoff > best_value + tolerance:
                best_value = candidate_payoff
                best_indices = [candidate_index]
            elif abs(candidate_payoff - best_value) <= tolerance:
                best_indices.append(candidate_index)
        best_response_rows.append(tuple(best_indices))
        gains.append(float(best_value - result.expected_payoffs[player]))

    outcome, probability = result.most_likely_outcome()
    return KPlayerProfileEvaluation(
        strategy_indices=indices,
        strategies=profile,
        payoffs=result.expected_payoffs,
        best_response_indices=tuple(best_response_rows),
        unilateral_gains=tuple(gains),
        is_nash=all(gain <= tolerance for gain in gains),
        probability_sum=result.probability_sum,
        most_likely_outcome=outcome,
        most_likely_probability=probability,
    )


def iter_sparse_count_profiles(strategy_count: int, total: int) -> Iterator[SparseCountProfile]:
    """Yield sparse strategy-count profiles whose counts sum to total."""
    _validate_strategy_count(strategy_count)
    if total < 0:
        raise ValueError(f"total must be nonnegative; got {total!r}")
    yield from _iter_sparse_count_profiles_from(0, strategy_count, total)


def sparse_count_profile_count(strategy_count: int, total: int) -> int:
    """Return the number of sparse count profiles for a grid and total players."""
    _validate_strategy_count(strategy_count)
    if total < 0:
        raise ValueError(f"total must be nonnegative; got {total!r}")
    return math.comb(strategy_count + total - 1, total)


def sparse_counts_to_indices(counts: SparseCountProfile) -> Tuple[int, ...]:
    """Expand sparse counts into sorted strategy indices."""
    indices: List[int] = []
    previous = -1
    for index, count in counts:
        if index <= previous:
            raise ValueError(f"sparse count indices must be increasing: {counts!r}")
        if count <= 0:
            raise ValueError(f"sparse counts must be positive: {counts!r}")
        indices.extend([index] * count)
        previous = index
    return tuple(indices)


def subtract_one_from_sparse_counts(
    counts: SparseCountProfile,
    strategy_index: int,
) -> SparseCountProfile:
    """Return a count profile with one copy of strategy_index removed."""
    updated = []
    removed = False
    for index, count in counts:
        if index == strategy_index:
            if count <= 0:
                raise ValueError(f"cannot subtract from nonpositive count: {counts!r}")
            removed = True
            if count > 1:
                updated.append((index, count - 1))
        else:
            updated.append((index, count))
    if not removed:
        raise ValueError(f"strategy {strategy_index!r} is not present in {counts!r}")
    return tuple(updated)


def build_symmetric_best_response_table(
    k: int,
    gamma: float,
    strategies: Sequence[EWLStrategy],
    tolerance: float = 1e-8,
    workers: int = 1,
    chunksize: int = 128,
) -> BestResponseTable:
    """Build best responses to every opponent count profile.

    This assumes the K-player game is symmetric, which is true for the default
    K-player EWL Prisoner's Dilemma. Opponents are represented as counts rather
    than ordered tuples, reducing the K=5, 66-grid opponent cases from 66^4 to
    C(69, 4) = 864,501.
    """
    _validate_k(k)
    _validate_search_inputs(gamma, strategies, tolerance, workers, chunksize)

    strategy_tuple = tuple(strategies)
    opponent_contexts = iter_sparse_count_profiles(len(strategy_tuple), k - 1)
    table: BestResponseTable = {}

    if workers <= 1:
        game = KPlayerEWLGame(k=k, gamma=gamma)
        for opponent_counts in opponent_contexts:
            table[opponent_counts] = _best_response_for_opponents(
                game=game,
                strategies=strategy_tuple,
                opponent_counts=opponent_counts,
                tolerance=tolerance,
            )
        return table

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_best_response_worker,
        initargs=(k, gamma, strategy_tuple, tolerance),
    ) as executor:
        for opponent_counts, best_value, best_indices in executor.map(
            _parallel_best_response_for_opponents,
            opponent_contexts,
            chunksize=chunksize,
        ):
            table[opponent_counts] = (best_value, best_indices)
    return table


def find_pure_nash_equilibria_on_grid(
    k: int,
    gamma: float,
    strategies: Sequence[EWLStrategy],
    tolerance: float = 1e-8,
    workers: int = 1,
    chunksize: int = 128,
    max_results: Optional[int] = None,
) -> KPlayerNashSearchResult:
    """Find every ordered pure Nash equilibrium on a finite EWL strategy grid.

    The finite game contains ``len(strategies) ** k`` ordered profiles. The
    default K-player EWL Prisoner's Dilemma is symmetric, so the search can
    compute best responses once per opponent strategy multiset without changing
    the game or omitting any possible deviation. Every ordered permutation of a
    confirmed equilibrium is returned separately.
    """
    _validate_k(k)
    _validate_search_inputs(gamma, strategies, tolerance, workers, chunksize)
    if max_results is not None and max_results <= 0:
        raise ValueError(f"max_results must be positive or None; got {max_results!r}")

    strategy_tuple = tuple(strategies)
    best_response_table = build_symmetric_best_response_table(
        k=k,
        gamma=gamma,
        strategies=strategy_tuple,
        tolerance=tolerance,
        workers=workers,
        chunksize=chunksize,
    )

    equilibria: List[KPlayerNashEquilibrium] = []
    truncated = False
    for counts in iter_sparse_count_profiles(len(strategy_tuple), k):
        if _count_profile_is_nash(counts, best_response_table):
            canonical_indices = sparse_counts_to_indices(counts)
            for indices in _unique_ordered_permutations(canonical_indices):
                audit = evaluate_profile_on_grid(
                    k=k,
                    gamma=gamma,
                    strategies=strategy_tuple,
                    strategy_indices=indices,
                    tolerance=tolerance,
                )
                if not audit.is_nash:
                    raise RuntimeError(
                        "symmetry-reduced search disagreed with direct deviation audit "
                        f"for profile {indices!r}"
                    )
                equilibria.append(
                    KPlayerNashEquilibrium(
                        strategy_indices=indices,
                        strategy_counts=counts,
                        strategies=audit.strategies,
                        payoffs=audit.payoffs,
                        best_response_indices=audit.best_response_indices,
                        unilateral_gains=audit.unilateral_gains,
                        probability_sum=audit.probability_sum,
                        most_likely_outcome=audit.most_likely_outcome,
                        most_likely_probability=audit.most_likely_probability,
                    )
                )
                if max_results is not None and len(equilibria) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

    return KPlayerNashSearchResult(
        k=k,
        gamma=gamma,
        strategy_count=len(strategy_tuple),
        ordered_profile_count=ordered_profile_count(len(strategy_tuple), k),
        opponent_context_count=sparse_count_profile_count(len(strategy_tuple), k - 1),
        resident_count_profile_count=sparse_count_profile_count(len(strategy_tuple), k),
        equilibria=tuple(equilibria),
        tolerance=tolerance,
        truncated=truncated,
    )


def _unique_ordered_permutations(
    indices: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], ...]:
    """Return each distinct player ordering of a canonical strategy multiset."""
    return tuple(sorted(set(permutations(indices))))


def _best_response_for_opponents(
    game: KPlayerEWLGame,
    strategies: Tuple[EWLStrategy, ...],
    opponent_counts: SparseCountProfile,
    tolerance: float,
) -> Tuple[float, Tuple[int, ...]]:
    opponent_indices = sparse_counts_to_indices(opponent_counts)
    opponents = tuple(strategies[index] for index in opponent_indices)
    best_value = -math.inf
    best_indices: List[int] = []

    for candidate_index, candidate in enumerate(strategies):
        payoff = game.expected_payoffs((candidate,) + opponents)[0]
        if payoff > best_value + tolerance:
            best_value = payoff
            best_indices = [candidate_index]
        elif abs(payoff - best_value) <= tolerance:
            best_indices.append(candidate_index)

    return (float(best_value), tuple(best_indices))


def _parallel_best_response_for_opponents(
    opponent_counts: SparseCountProfile,
) -> Tuple[SparseCountProfile, float, Tuple[int, ...]]:
    if _WORKER_K is None or _WORKER_GAMMA is None or _WORKER_STRATEGIES is None:
        raise RuntimeError("best-response worker is not initialized")
    game = KPlayerEWLGame(k=_WORKER_K, gamma=_WORKER_GAMMA)
    best_value, best_indices = _best_response_for_opponents(
        game=game,
        strategies=_WORKER_STRATEGIES,
        opponent_counts=opponent_counts,
        tolerance=_WORKER_TOLERANCE,
    )
    return (opponent_counts, best_value, best_indices)


def _count_profile_is_nash(
    counts: SparseCountProfile,
    best_response_table: BestResponseTable,
) -> bool:
    for strategy_index, _count in counts:
        opponent_counts = subtract_one_from_sparse_counts(counts, strategy_index)
        _best_value, best_indices = best_response_table[opponent_counts]
        if strategy_index not in best_indices:
            return False
    return True


def _init_best_response_worker(
    k: int,
    gamma: float,
    strategies: Tuple[EWLStrategy, ...],
    tolerance: float,
) -> None:
    global _WORKER_K, _WORKER_GAMMA, _WORKER_STRATEGIES, _WORKER_TOLERANCE
    _WORKER_K = k
    _WORKER_GAMMA = gamma
    _WORKER_STRATEGIES = strategies
    _WORKER_TOLERANCE = tolerance


def _iter_sparse_count_profiles_from(
    start_index: int,
    strategy_count: int,
    remaining: int,
) -> Iterator[SparseCountProfile]:
    if remaining == 0:
        yield ()
        return
    for index in range(start_index, strategy_count):
        for count in range(1, remaining + 1):
            for tail in _iter_sparse_count_profiles_from(index + 1, strategy_count, remaining - count):
                yield ((index, count),) + tail


def _linspace(start: float, stop: float, count: int) -> Tuple[float, ...]:
    if count == 1:
        return (float(start),)
    step = (stop - start) / (count - 1)
    return tuple(float(start + step * index) for index in range(count))


def _validate_grid_count(count: int, name: str) -> None:
    if count < 1:
        raise ValueError(f"{name} must be at least 1; got {count!r}")


def _validate_finite_range(start: float, stop: float, name: str) -> None:
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError(f"{name} range must be finite")
    if stop < start:
        raise ValueError(f"{name}_max must be greater than or equal to {name}_min")


def _validate_strategy_count(strategy_count: int) -> None:
    if strategy_count < 1:
        raise ValueError(f"strategy_count must be at least 1; got {strategy_count!r}")


def _validate_k(k: int) -> None:
    if k not in SUPPORTED_K_VALUES:
        raise ValueError(f"K must be one of {SUPPORTED_K_VALUES}; got {k!r}")


def _validate_search_inputs(
    gamma: float,
    strategies: Sequence[EWLStrategy],
    tolerance: float,
    workers: int,
    chunksize: int,
) -> None:
    if not math.isfinite(gamma):
        raise ValueError(f"gamma must be finite; got {gamma!r}")
    if not strategies:
        raise ValueError("at least one strategy is required")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite nonnegative number; got {tolerance!r}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1; got {workers!r}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be at least 1; got {chunksize!r}")
