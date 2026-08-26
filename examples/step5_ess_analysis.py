"""Run the reproducible Step 5 resident-mutant EWL invasion analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from quantum_ess import (
    DEFAULT_DIAGNOSTIC_FREQUENCIES,
    NOT_ESS,
    EWLStrategy,
    KPlayerEWLGame,
    build_resident_summary,
    continuous_mutant_search,
    dataclass_json,
    deduplicate_ewl_strategies,
    detailed_rows_from_continuous,
    detailed_rows_from_grid,
    ewl_strategy_grid,
    refined_mutant_grid,
    search_mutants_on_grid,
    symmetric_nash_candidates_on_grid,
    write_detailed_csv,
    write_json,
    write_summary_csv,
)

AUTHORITATIVE_BASELINE_COMMIT = "5827d9aaf631f67a02713eb8643609a7730707d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/step5"))
    parser.add_argument("--k-values", type=int, nargs="+", default=(2, 3, 4, 5))
    parser.add_argument("--gamma", type=float, default=math.pi / 2.0)
    parser.add_argument("--epsilon-reference", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--coarse-theta-count", type=int, default=11)
    parser.add_argument("--coarse-phi-count", type=int, default=6)
    parser.add_argument("--fine-theta-count", type=int, default=21)
    parser.add_argument("--fine-phi-count", type=int, default=11)
    parser.add_argument("--landscape-theta-count", type=int, default=41)
    parser.add_argument("--landscape-phi-count", type=int, default=21)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--continuous-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--differential-maxiter", type=int, default=200)
    parser.add_argument("--differential-popsize", type=int, default=10)
    parser.add_argument("--local-start-limit", type=int, default=8)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-continuous", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    coarse_grid = deduplicate_ewl_strategies(
        ewl_strategy_grid(
            theta_count=args.coarse_theta_count,
            phi_count=args.coarse_phi_count,
            label_prefix="C",
        ).strategies
    )
    candidate_rows = []
    detailed_rows = []
    summaries = []
    optimizer_rows = []
    complete_results: Dict[str, object] = {}
    important_residents: Dict[int, EWLStrategy] = {}

    for k in args.k_values:
        candidates = symmetric_nash_candidates_on_grid(
            k=k,
            gamma=args.gamma,
            strategies=coarse_grid,
            tolerance=args.tolerance,
        )
        if not candidates:
            raise RuntimeError(
                f"no symmetric finite-grid Nash candidate found for K={k}"
            )
        important_residents[k] = max(
            candidates,
            key=lambda candidate: candidate.resident_payoff,
        ).resident
        k_results = []

        print(f"K={k}: {len(candidates)} canonical symmetric Nash candidate(s)")
        for candidate_index, candidate in enumerate(candidates):
            candidate_rows.append(
                {
                    "k": candidate.k,
                    "gamma": candidate.gamma,
                    "resident_theta": candidate.resident.theta,
                    "resident_phi": candidate.resident.phi,
                    "resident_payoff": candidate.resident_payoff,
                    "maximum_unilateral_gain": candidate.maximum_unilateral_gain,
                    "grid_strategy_count": candidate.grid_strategy_count,
                    "best_response_mutants": json.dumps(
                        [
                            [strategy.theta, strategy.phi]
                            for strategy in candidate.best_response_mutants
                        ]
                    ),
                }
            )
            print(
                "  resident"
                f" ({candidate.resident.theta:.9g}, {candidate.resident.phi:.9g})"
                f" payoff={candidate.resident_payoff:.9g}"
            )
            coarse = search_mutants_on_grid(
                k=k,
                gamma=args.gamma,
                resident=candidate.resident,
                mutant_strategies=coarse_grid,
                epsilon_reference=args.epsilon_reference,
                stage="coarse",
                tolerance=args.tolerance,
                diagnostic_frequencies=DEFAULT_DIAGNOSTIC_FREQUENCIES,
                workers=args.workers,
                chunksize=args.chunksize,
            )
            detailed_rows.extend(detailed_rows_from_grid(coarse))
            print(
                f"    coarse: {coarse.resident_status}, "
                f"max Delta={coarse.maximum_mutant_advantage_at_epsilon_reference:.6g}"
            )

            fine = None
            continuous = None
            if coarse.resident_status != NOT_ESS:
                fine_grid = refined_mutant_grid(
                    theta_count=args.fine_theta_count,
                    phi_count=args.fine_phi_count,
                    centers=(candidate.resident, coarse.strongest_mutant),
                    local_levels=2,
                )
                fine = search_mutants_on_grid(
                    k=k,
                    gamma=args.gamma,
                    resident=candidate.resident,
                    mutant_strategies=fine_grid,
                    epsilon_reference=args.epsilon_reference,
                    stage="fine",
                    tolerance=args.tolerance,
                    diagnostic_frequencies=DEFAULT_DIAGNOSTIC_FREQUENCIES,
                    workers=args.workers,
                    chunksize=args.chunksize,
                )
                detailed_rows.extend(detailed_rows_from_grid(fine))
                print(
                    f"    fine: {fine.resident_status}, "
                    "max Delta="
                    f"{fine.maximum_mutant_advantage_at_epsilon_reference:.6g}"
                )

            if (
                fine is not None
                and fine.resident_status != NOT_ESS
                and not args.skip_continuous
            ):
                continuous = continuous_mutant_search(
                    k=k,
                    gamma=args.gamma,
                    resident=candidate.resident,
                    epsilon_reference=args.epsilon_reference,
                    seed_strategies=(coarse.strongest_mutant, fine.strongest_mutant),
                    tolerance=args.tolerance,
                    diagnostic_frequencies=DEFAULT_DIAGNOSTIC_FREQUENCIES,
                    objective_frequencies=(0.0, 1e-6, 1e-4, args.epsilon_reference),
                    seed=args.seed + 1000 * k + candidate_index,
                    differential_maxiter=args.differential_maxiter,
                    differential_popsize=args.differential_popsize,
                    local_start_limit=args.local_start_limit,
                    optimizer_tolerance=args.optimizer_tolerance,
                    workers=args.continuous_workers,
                )
                detailed_rows.extend(detailed_rows_from_continuous(continuous))
                optimizer_rows.extend(
                    {
                        "k": k,
                        "gamma": args.gamma,
                        "resident_theta": candidate.resident.theta,
                        "resident_phi": candidate.resident.phi,
                        "seed": continuous.seed,
                        **asdict(run),
                    }
                    for run in continuous.optimizer_runs
                )
                print(
                    f"    continuous: {continuous.resident_status}, "
                    "max Delta="
                    f"{continuous.maximum_mutant_advantage_at_epsilon_reference:.6g}"
                )

            summary = build_resident_summary(candidate, coarse, fine, continuous)
            summaries.append(summary)
            k_results.append(
                {
                    "candidate": dataclass_json(candidate),
                    "coarse": dataclass_json(coarse),
                    "fine": None if fine is None else dataclass_json(fine),
                    "continuous": (
                        None if continuous is None else dataclass_json(continuous)
                    ),
                    "summary": asdict(summary),
                }
            )
        complete_results[str(k)] = k_results

    write_detailed_csv(
        output_dir / "resident_mutant_pair_results.csv.gz",
        detailed_rows,
    )
    write_summary_csv(output_dir / "resident_summary.csv", summaries)
    _write_mapping_csv(output_dir / "symmetric_nash_candidates.csv", candidate_rows)
    _write_mapping_csv(output_dir / "continuous_optimizer_runs.csv", optimizer_rows)
    write_json(output_dir / "complete_results.json.gz", complete_results)

    landscape_rows = []
    for k, resident in important_residents.items():
        rows = _landscape_rows(
            k=k,
            gamma=args.gamma,
            resident=resident,
            epsilon_reference=args.epsilon_reference,
            tolerance=args.tolerance,
            theta_count=args.landscape_theta_count,
            phi_count=args.landscape_phi_count,
        )
        landscape_rows.extend(rows)
        _plot_landscape(
            rows=rows,
            k=k,
            resident=resident,
            epsilon_reference=args.epsilon_reference,
            output_path=figures_dir / f"k{k}_mutant_invasion_landscape.png",
        )
    _write_mapping_csv(output_dir / "landscape_points.csv", landscape_rows)

    metadata = {
        "authoritative_baseline_commit": AUTHORITATIVE_BASELINE_COMMIT,
        "working_tree_head_at_run": _git_commit(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": vars(args) | {"output_dir": str(output_dir)},
        "diagnostic_frequencies": DEFAULT_DIAGNOSTIC_FREQUENCIES,
        "coarse_canonical_strategy_count": len(coarse_grid),
        "result_files": sorted(
            {
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            | {"run_metadata.json"}
        ),
    }
    write_json(output_dir / "run_metadata.json", metadata)
    print(f"Results written to {output_dir}")


def _landscape_rows(
    k: int,
    gamma: float,
    resident: EWLStrategy,
    epsilon_reference: float,
    tolerance: float,
    theta_count: int,
    phi_count: int,
) -> List[dict]:
    game = KPlayerEWLGame(k=k, gamma=gamma)
    grid = ewl_strategy_grid(
        theta_count=theta_count,
        phi_count=phi_count,
        label_prefix="L",
    )
    rows = []
    from quantum_ess import analyze_resident_mutant

    for mutant in grid.strategies:
        pair = analyze_resident_mutant(
            game=game,
            resident=resident,
            mutant=mutant,
            epsilon_reference=epsilon_reference,
            tolerance=tolerance,
            diagnostic_frequencies=(epsilon_reference,),
            positions=(0,),
        )
        result = pair.worst_reference_result
        rows.append(
            {
                "k": k,
                "gamma": gamma,
                "resident_theta": resident.theta,
                "resident_phi": resident.phi,
                "mutant_theta": mutant.theta,
                "mutant_phi": mutant.phi,
                "DeltaPi_epsilon_reference": result.mutant_advantage_reference,
                "leading_nonzero_order": (
                    ""
                    if result.leading_nonzero_order is None
                    else result.leading_nonzero_order
                ),
                "leading_coefficient": result.leading_coefficient,
                "rare_mutant_classification": result.rare_mutant_classification,
            }
        )
    return rows


def _plot_landscape(
    rows: Sequence[dict],
    k: int,
    resident: EWLStrategy,
    epsilon_reference: float,
    output_path: Path,
) -> None:
    theta_values = sorted({float(row["mutant_theta"]) for row in rows})
    phi_values = sorted({float(row["mutant_phi"]) for row in rows})
    lookup = {
        (float(row["mutant_theta"]), float(row["mutant_phi"])): float(
            row["DeltaPi_epsilon_reference"]
        )
        for row in rows
    }
    values = [
        [lookup[(theta, phi)] for theta in theta_values]
        for phi in phi_values
    ]
    flat = [value for row in values for value in row]
    minimum = min(flat)
    maximum = max(flat)
    span = max(abs(minimum), abs(maximum), 1e-12)
    vmin = min(minimum, -0.02 * span)
    vmax = max(maximum, 0.02 * span)
    cmap = LinearSegmentedColormap.from_list(
        "quantum_ess_invasion",
        ("#2F6B9A", "#F4F3EF", "#D9822B"),
    )

    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    mesh = axis.pcolormesh(
        theta_values,
        phi_values,
        values,
        shading="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )
    if minimum < 0.0 < maximum:
        axis.contour(
            theta_values,
            phi_values,
            values,
            levels=(0.0,),
            colors="#252525",
            linewidths=1.2,
        )
    axis.scatter(
        [resident.theta],
        [resident.phi],
        marker="*",
        s=115,
        c="#202020",
        edgecolors="white",
        linewidths=0.8,
    )
    axis.set_title(f"K={k} mutant invasion landscape")
    axis.set_xlabel(r"Mutant $\theta_M$")
    axis.set_ylabel(r"Mutant $\phi_M$")
    axis.set_xticks((0.0, math.pi / 2.0, math.pi), ("0", r"$\pi/2$", r"$\pi$"))
    axis.set_yticks(
        (0.0, math.pi / 4.0, math.pi / 2.0),
        ("0", r"$\pi/4$", r"$\pi/2$"),
    )
    axis.annotate(
        "resident",
        xy=(resident.theta, resident.phi),
        xytext=(10, -14),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#202020",
    )
    colorbar = fig.colorbar(mesh, ax=axis)
    colorbar.set_label(rf"$\Delta\Pi(\epsilon={epsilon_reference:g})$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_mapping_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
