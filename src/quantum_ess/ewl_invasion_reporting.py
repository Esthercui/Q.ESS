"""Structured CSV and JSON reporting for Step 5 EWL invasion analyses."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .ewl_invasion import INDETERMINATE, ResidentMutantAnalysis
from .ewl_invasion_search import (
    GRID_ESS_CANDIDATE,
    NEUTRAL_OR_WEAK_CANDIDATE,
    NOT_ESS,
    NUMERICALLY_SUPPORTED_ESS,
    ContinuousMutantSearchResult,
    MutantGridSearchResult,
    SymmetricNashCandidate,
)


@dataclass(frozen=True)
class ResidentSummaryRow:
    """Concise evidence summary for one resident strategy."""

    k: int
    gamma: float
    resident_theta: float
    resident_phi: float
    resident_status: str
    maximum_mutant_advantage_at_epsilon_reference: float
    strongest_mutant_theta: float
    strongest_mutant_phi: float
    strongest_mutant_position: int
    number_of_robust_invaders: int
    number_of_neutral_mutants: int
    coarse_grid_result: str
    fine_grid_result: str
    continuous_optimizer_result: str
    permutation_error: float
    final_interpretation: str


DETAILED_FIELDNAMES = (
    "search_stage",
    "k",
    "gamma",
    "resident_theta",
    "resident_phi",
    "mutant_theta",
    "mutant_phi",
    "mutant_position",
    "epsilon_reference",
    "Pi_R_epsilon_reference",
    "Pi_M_epsilon_reference",
    "DeltaPi_epsilon_reference",
    "leading_nonzero_order",
    "leading_coefficient",
    "uncertainty_or_confidence_interval",
    "rare_mutant_classification",
    "invades_at_epsilon_reference",
    "tolerance",
    "permutation_error",
    "probability_error",
    "polynomial_reconstruction_error",
    "bernstein_advantages",
    "power_coefficients",
    "composition_payoffs",
    "diagnostic_frequencies",
)


def detailed_rows_from_grid(
    result: MutantGridSearchResult,
) -> Tuple[Mapping[str, Any], ...]:
    """Return one required pair-level row per tested focal position."""
    rows = []
    for pair in result.pair_results:
        rows.extend(_pair_rows(result.stage, pair))
    return tuple(rows)


def detailed_rows_from_continuous(
    result: ContinuousMutantSearchResult,
) -> Tuple[Mapping[str, Any], ...]:
    """Return position-resolved rows for every continuous optimizer candidate."""
    rows = []
    for pair in result.candidate_results:
        rows.extend(_pair_rows("continuous", pair))
    return tuple(rows)


def build_resident_summary(
    candidate: SymmetricNashCandidate,
    coarse: MutantGridSearchResult,
    fine: Optional[MutantGridSearchResult],
    continuous: Optional[ContinuousMutantSearchResult],
) -> ResidentSummaryRow:
    """Combine staged evidence without promoting grid evidence to proof."""
    selected: Union[MutantGridSearchResult, ContinuousMutantSearchResult] = coarse
    status = coarse.resident_status
    if coarse.resident_status != NOT_ESS and fine is not None:
        selected = fine
        status = fine.resident_status
    if status != NOT_ESS and continuous is not None:
        selected = continuous
        status = continuous.resident_status

    maximum_advantage = float(
        selected.maximum_mutant_advantage_at_epsilon_reference
    )
    strongest = selected.strongest_mutant
    strongest_position = int(selected.strongest_mutant_position)
    robust_invaders = int(selected.number_of_robust_invaders)
    neutral_mutants = int(selected.number_of_neutral_mutants)
    permutation_errors = [coarse.maximum_permutation_error]
    if fine is not None:
        permutation_errors.append(fine.maximum_permutation_error)
    if continuous is not None:
        permutation_errors.append(continuous.maximum_permutation_error)

    return ResidentSummaryRow(
        k=candidate.k,
        gamma=candidate.gamma,
        resident_theta=candidate.resident.theta,
        resident_phi=candidate.resident.phi,
        resident_status=status,
        maximum_mutant_advantage_at_epsilon_reference=maximum_advantage,
        strongest_mutant_theta=strongest.theta,
        strongest_mutant_phi=strongest.phi,
        strongest_mutant_position=strongest_position,
        number_of_robust_invaders=robust_invaders,
        number_of_neutral_mutants=neutral_mutants,
        coarse_grid_result=_grid_result_text(coarse),
        fine_grid_result=(
            _grid_result_text(fine)
            if fine is not None
            else "not_run_not_apparent_stable"
        ),
        continuous_optimizer_result=(
            _continuous_result_text(continuous)
            if continuous is not None
            else "not_run_not_fine_grid_stable"
        ),
        permutation_error=max(permutation_errors),
        final_interpretation=_interpretation(status, continuous is not None),
    )


def write_detailed_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write complete position-resolved pair results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=DETAILED_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: Sequence[ResidentSummaryRow]) -> None:
    """Write concise resident evidence rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(asdict(rows[0]).keys()) if rows else tuple(
        field.name for field in ResidentSummaryRow.__dataclass_fields__.values()
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def dataclass_json(value: Any) -> Any:
    """Return a JSON-compatible representation of nested dataclasses."""
    return asdict(value)


def _open_text(path: Path, newline: Optional[str] = None):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline=newline)
    return path.open("w", encoding="utf-8", newline=newline)


def _pair_rows(
    stage: str,
    pair: ResidentMutantAnalysis,
) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    for result in pair.position_results:
        rows.append(
            {
                "search_stage": stage,
                "k": result.k,
                "gamma": result.gamma,
                "resident_theta": result.resident.theta,
                "resident_phi": result.resident.phi,
                "mutant_theta": result.mutant.theta,
                "mutant_phi": result.mutant.phi,
                "mutant_position": result.mutant_position,
                "epsilon_reference": result.epsilon_reference,
                "Pi_R_epsilon_reference": result.resident_payoff_reference,
                "Pi_M_epsilon_reference": result.mutant_payoff_reference,
                "DeltaPi_epsilon_reference": result.mutant_advantage_reference,
                "leading_nonzero_order": (
                    ""
                    if result.leading_nonzero_order is None
                    else result.leading_nonzero_order
                ),
                "leading_coefficient": result.leading_coefficient,
                "uncertainty_or_confidence_interval": (
                    result.uncertainty_or_confidence_interval
                ),
                "rare_mutant_classification": result.rare_mutant_classification,
                "invades_at_epsilon_reference": result.invades_at_epsilon_reference,
                "tolerance": result.tolerance,
                "permutation_error": result.permutation_error,
                "probability_error": result.probability_error,
                "polynomial_reconstruction_error": (
                    result.polynomial_reconstruction_error
                ),
                "bernstein_advantages": json.dumps(result.bernstein_advantages),
                "power_coefficients": json.dumps(result.power_coefficients),
                "composition_payoffs": json.dumps(
                    [asdict(row) for row in result.composition_payoffs]
                ),
                "diagnostic_frequencies": json.dumps(
                    [asdict(row) for row in result.diagnostic_frequencies]
                ),
            }
        )
    return tuple(rows)


def _grid_result_text(result: MutantGridSearchResult) -> str:
    return (
        f"{result.resident_status}; mutants={result.mutant_strategy_count}; "
        "max_delta_reference="
        f"{result.maximum_mutant_advantage_at_epsilon_reference:.17g}"
    )


def _continuous_result_text(result: ContinuousMutantSearchResult) -> str:
    return (
        f"{result.resident_status}; candidates={len(result.candidate_results)}; "
        f"all_global_runs_converged={result.all_global_runs_converged}; "
        "max_delta_reference="
        f"{result.maximum_mutant_advantage_at_epsilon_reference:.17g}"
    )


def _interpretation(status: str, continuous_ran: bool) -> str:
    if status == NOT_ESS:
        return (
            "At least one tested mutant has a robust positive rare-limit "
            "coefficient."
        )
    if status == GRID_ESS_CANDIDATE:
        return (
            "No invader was found on the current finite grids; continuous validation "
            "is still required."
        )
    if status == NUMERICALLY_SUPPORTED_ESS:
        return (
            "No invader was found after grid refinement and bounded continuous search; "
            "this is numerical support, not an analytical proof."
        )
    if status == NEUTRAL_OR_WEAK_CANDIDATE:
        return (
            "No positive invader was found, but at least one non-self mutant "
            "is neutral within tolerance; strict evolutionary stability is "
            "not established."
        )
    if status == INDETERMINATE:
        suffix = " after continuous search" if continuous_ran else ""
        return (
            "Numerical reliability or optimizer convergence is "
            f"indeterminate{suffix}."
        )
    return f"Unrecognized computational status: {status}."
