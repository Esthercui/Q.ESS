"""Independently validate the saved Step 5 analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

TOLERANCE = 1e-8
ANGLE_ENDPOINT_TOLERANCE = 16.0 * math.ulp(math.pi)

REQUIRED_DETAIL_COLUMNS = {
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
    "permutation_error",
    "probability_error",
    "polynomial_reconstruction_error",
}

REQUIRED_SUMMARY_COLUMNS = {
    "k",
    "gamma",
    "resident_theta",
    "resident_phi",
    "resident_status",
    "maximum_mutant_advantage_at_epsilon_reference",
    "strongest_mutant_theta",
    "strongest_mutant_phi",
    "strongest_mutant_position",
    "number_of_robust_invaders",
    "number_of_neutral_mutants",
    "coarse_grid_result",
    "fine_grid_result",
    "continuous_optimizer_result",
    "permutation_error",
    "final_interpretation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/step5"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    errors: List[str] = []

    details, detail_columns = _read_csv(
        results_dir / "resident_mutant_pair_results.csv.gz"
    )
    summaries, summary_columns = _read_csv(results_dir / "resident_summary.csv")
    optimizers, optimizer_columns = _read_csv(
        results_dir / "continuous_optimizer_runs.csv"
    )
    candidates, _candidate_columns = _read_csv(
        results_dir / "symmetric_nash_candidates.csv"
    )
    metadata = _read_json(results_dir / "run_metadata.json")

    _require_columns(detail_columns, REQUIRED_DETAIL_COLUMNS, "detail", errors)
    _require_columns(summary_columns, REQUIRED_SUMMARY_COLUMNS, "summary", errors)
    _require_columns(
        optimizer_columns,
        {"seed", "differential_evolution_success", "objective_epsilon"},
        "optimizer",
        errors,
    )

    if {int(row["k"]) for row in candidates} != {2, 3, 4, 5}:
        errors.append("symmetric candidate output does not cover K=2,3,4,5")
    if len(summaries) != len(candidates):
        errors.append("resident summary count does not match candidate count")
    if metadata.get("authoritative_baseline_commit") != (
        "5827d9aaf631f67a02713eb8643609a7730707d2"
    ):
        errors.append("run metadata does not name the authoritative baseline")

    finite_columns = (
        "gamma",
        "resident_theta",
        "resident_phi",
        "mutant_theta",
        "mutant_phi",
        "epsilon_reference",
        "Pi_R_epsilon_reference",
        "Pi_M_epsilon_reference",
        "DeltaPi_epsilon_reference",
        "leading_coefficient",
        "permutation_error",
        "probability_error",
        "polynomial_reconstruction_error",
    )
    for row_number, row in enumerate(details, start=2):
        for column in finite_columns:
            try:
                value = float(row[column])
            except (KeyError, ValueError):
                errors.append(f"detail row {row_number} has invalid {column}")
                continue
            if not math.isfinite(value):
                errors.append(f"detail row {row_number} has non-finite {column}")

        theta_r = float(row["resident_theta"])
        phi_r = float(row["resident_phi"])
        theta_m = float(row["mutant_theta"])
        phi_m = float(row["mutant_phi"])
        if not (0.0 <= theta_m <= math.pi and 0.0 <= phi_m <= math.pi / 2.0):
            errors.append(f"detail row {row_number} leaves the strategy domain")
        if abs(theta_m - math.pi) <= ANGLE_ENDPOINT_TOLERANCE and phi_m != 0.0:
            errors.append(f"detail row {row_number} contains an uncanonicalized D edge")
        if _equivalent(theta_r, phi_r, theta_m, phi_m):
            errors.append(f"detail row {row_number} contains an exact self-mutant")

    max_probability_error = max(float(row["probability_error"]) for row in details)
    max_polynomial_error = max(
        float(row["polynomial_reconstruction_error"]) for row in details
    )
    max_permutation_error = max(float(row["permutation_error"]) for row in details)
    if max_probability_error > TOLERANCE:
        errors.append("probability normalization error exceeds tolerance")
    if max_polynomial_error > TOLERANCE:
        errors.append("polynomial reconstruction error exceeds tolerance")
    if max_permutation_error > TOLERANCE:
        errors.append("permutation error exceeds tolerance")

    pair_key = Tuple[str, int, float, float, float, float]
    grouped: Dict[pair_key, List[Mapping[str, str]]] = defaultdict(list)
    for row in details:
        key = (
            row["search_stage"],
            int(row["k"]),
            float(row["resident_theta"]),
            float(row["resident_phi"]),
            float(row["mutant_theta"]),
            float(row["mutant_phi"]),
        )
        grouped[key].append(row)
    for key, rows in grouped.items():
        k = key[1]
        positions = {int(row["mutant_position"]) for row in rows}
        classifications = {row["rare_mutant_classification"] for row in rows}
        if positions != set(range(k)):
            errors.append(f"{key} does not contain all K focal positions")
        if len(classifications) != 1:
            errors.append(f"{key} changes classification across positions")

    for summary in summaries:
        stage = _selected_stage(summary)
        resident_key = (
            int(summary["k"]),
            float(summary["resident_theta"]),
            float(summary["resident_phi"]),
        )
        selected = [
            row
            for row in details
            if row["search_stage"] == stage
            and int(row["k"]) == resident_key[0]
            and float(row["resident_theta"]) == resident_key[1]
            and float(row["resident_phi"]) == resident_key[2]
        ]
        if not selected:
            errors.append(f"summary {resident_key} has no {stage} detail rows")
            continue
        maximum = max(float(row["DeltaPi_epsilon_reference"]) for row in selected)
        reported = float(
            summary["maximum_mutant_advantage_at_epsilon_reference"]
        )
        if abs(maximum - reported) > TOLERANCE:
            errors.append(f"summary {resident_key} maximum does not match details")

        pair_classes = {
            (float(row["mutant_theta"]), float(row["mutant_phi"])): row[
                "rare_mutant_classification"
            ]
            for row in selected
        }
        invaders = sum(
            value == "rare_mutant_invades" for value in pair_classes.values()
        )
        neutrals = sum(value == "neutral" for value in pair_classes.values())
        if invaders != int(summary["number_of_robust_invaders"]):
            errors.append(
                f"summary {resident_key} invader count does not match details"
            )
        if neutrals != int(summary["number_of_neutral_mutants"]):
            errors.append(
                f"summary {resident_key} neutral count does not match details"
            )

        if stage == "continuous":
            coarse_max = _max_for_stage(details, resident_key, "coarse")
            fine_max = _max_for_stage(details, resident_key, "fine")
            if maximum < coarse_max - TOLERANCE or maximum < fine_max - TOLERANCE:
                errors.append(
                    f"continuous search for {resident_key} underperforms a finite grid"
                )

    if not optimizers:
        errors.append("no continuous optimizer runs were saved")
    if any(row["differential_evolution_success"] != "True" for row in optimizers):
        errors.append("at least one global optimizer run did not converge")

    for k in (2, 3, 4, 5):
        figure = results_dir / "figures" / f"k{k}_mutant_invasion_landscape.png"
        if not figure.is_file() or figure.stat().st_size < 10_000:
            errors.append(f"K={k} landscape figure is missing or unexpectedly small")

    report = {
        "candidate_count": len(candidates),
        "detail_row_count": len(details),
        "global_optimizer_run_count": len(optimizers),
        "maximum_permutation_error": max_permutation_error,
        "maximum_polynomial_reconstruction_error": max_polynomial_error,
        "maximum_probability_error": max_probability_error,
        "resident_status_counts": dict(
            Counter(row["resident_status"] for row in summaries)
        ),
        "validation": "passed" if not errors else "failed",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


def _read_csv(path: Path) -> Tuple[List[Mapping[str, str]], Sequence[str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), tuple(reader.fieldnames or ())
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), tuple(reader.fieldnames or ())


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_columns(
    actual: Iterable[str],
    required: Iterable[str],
    label: str,
    errors: List[str],
) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        errors.append(f"{label} table is missing columns: {missing}")


def _canonical(theta: float, phi: float) -> Tuple[float, float]:
    if abs(theta - math.pi) <= ANGLE_ENDPOINT_TOLERANCE:
        return (math.pi, 0.0)
    return (theta, phi)


def _equivalent(
    theta_a: float,
    phi_a: float,
    theta_b: float,
    phi_b: float,
) -> bool:
    first = _canonical(theta_a, phi_a)
    second = _canonical(theta_b, phi_b)
    return (
        abs(first[0] - second[0]) <= ANGLE_ENDPOINT_TOLERANCE
        and abs(first[1] - second[1]) <= ANGLE_ENDPOINT_TOLERANCE
    )


def _selected_stage(summary: Mapping[str, str]) -> str:
    if not summary["continuous_optimizer_result"].startswith("not_run"):
        return "continuous"
    if not summary["fine_grid_result"].startswith("not_run"):
        return "fine"
    return "coarse"


def _max_for_stage(
    details: Sequence[Mapping[str, str]],
    resident_key: Tuple[int, float, float],
    stage: str,
) -> float:
    values = [
        float(row["DeltaPi_epsilon_reference"])
        for row in details
        if row["search_stage"] == stage
        and int(row["k"]) == resident_key[0]
        and float(row["resident_theta"]) == resident_key[1]
        and float(row["resident_phi"]) == resident_key[2]
    ]
    if not values:
        raise ValueError(f"no {stage} rows for resident {resident_key}")
    return max(values)


if __name__ == "__main__":
    main()
