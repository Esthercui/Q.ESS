"""Build and execute the reader-facing Step 5 analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "step5_resident_mutant_invasion.ipynb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = _build_notebook()
    nbformat.write(notebook, output)

    if not args.no_execute:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, output)
    print(output)


def _build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# Step 5: Resident-Mutant Invasion in K-Player EWL Games

## TL;DR

This notebook reports the reproducible `gamma=pi/2` baseline analysis for
`K=2,3,4,5`. It reads the complete saved search artifacts; it does not replace
the rare-limit ESS calculation with a single-frequency plot.

- Six canonical symmetric finite-grid Nash residents were screened.
- The `K=3` resident `(pi/2,pi/2)` and the lower-payoff `K=5` resident
  `(pi/2,pi/2)` have robust higher-order invaders and are `not_ESS`.
- Four other residents had no positive invader after grid refinement and
  continuous search, but distinct near-resident mutants were neutral within
  `1e-8`; they remain `neutral_or_weak_candidate`, not proven ESSs.
- All saved optimizer runs converged, and probability, polynomial, and
  permutation residuals remained far below the declared tolerance.
"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## Context and method

The authoritative Step 1-4 baseline is commit
`5827d9aaf631f67a02713eb8643609a7730707d2`. Strategies use the restricted EWL
family on `[0,pi] x [0,pi/2]`; the duplicate `theta=pi` boundary is represented
once. The project uses exact noiseless statevectors, pairwise-summed
Prisoner's Dilemma payoffs `(R,S,T,P)=(3,0,5,1)`, and its global
parity-adjusted K-body entangler.

For a focal type `X`, let `u_X(j;R,M)` be its payoff when `j` of its `K-1`
co-players are mutants. Independent random sampling in an infinite well-mixed
population gives

$$
\Pi_X(\epsilon)=\sum_{j=0}^{K-1}\binom{K-1}{j}
\epsilon^j(1-\epsilon)^{K-1-j}u_X(j;R,M).
$$

The invasion advantage is
`DeltaPi(epsilon)=Pi_M(epsilon)-Pi_R(epsilon)`. It is a polynomial of degree at
most `K-1`; the sign of its first coefficient above `1e-8` determines the rare
mutant classification. `DeltaPi(0.01)` below is a diagnostic, not the ESS
definition. Full equations and limitations are in `docs/step5_methods.md`.
"""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from quantum_ess import (
    EWL_C,
    EWL_D,
    KPlayerEWLGame,
    analyze_resident_mutant,
    exactly_one_mutant_group_diagnostic,
)

ROOT = Path.cwd()
RESULTS = ROOT / "results" / "step5"

summary = pd.read_csv(RESULTS / "resident_summary.csv")
candidates = pd.read_csv(RESULTS / "symmetric_nash_candidates.csv")
details = pd.read_csv(RESULTS / "resident_mutant_pair_results.csv.gz")
optimizers = pd.read_csv(RESULTS / "continuous_optimizer_runs.csv")
landscape = pd.read_csv(RESULTS / "landscape_points.csv")
metadata = json.loads((RESULTS / "run_metadata.json").read_text())

print(f"Loaded {len(details):,} position-resolved pair rows.")
print(f"Loaded {len(summary)} resident summaries and {len(optimizers)} optimizer runs.")
"""
        ),
        nbformat.v4.new_markdown_cell("## Data integrity checks"),
        nbformat.v4.new_code_cell(
            """required_k = {2, 3, 4, 5}
assert set(candidates["k"]) == required_k
assert len(summary) == len(candidates) == 6
assert details["k"].isin(required_k).all()
assert details["probability_error"].max() < 1e-8
assert details["polynomial_reconstruction_error"].max() < 1e-8
assert details["permutation_error"].max() < 1e-8
assert optimizers["differential_evolution_success"].all()
assert metadata["authoritative_baseline_commit"].startswith("5827d9a")

qa = pd.Series({
    "candidate_count": len(candidates),
    "pair_position_rows": len(details),
    "optimizer_runs": len(optimizers),
    "all_global_runs_converged": bool(
        optimizers["differential_evolution_success"].all()
    ),
    "maximum_probability_error": f'{details["probability_error"].max():.6e}',
    "maximum_polynomial_error": (
        f'{details["polynomial_reconstruction_error"].max():.6e}'
    ),
    "maximum_permutation_error": f'{details["permutation_error"].max():.6e}',
})
display(qa.to_frame("value"))
"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Resident screening and final classifications"
        ),
        nbformat.v4.new_code_cell(
            """resident_results = summary.merge(
    candidates[[
        "k", "gamma", "resident_theta", "resident_phi", "resident_payoff",
        "maximum_unilateral_gain",
    ]],
    on=["k", "gamma", "resident_theta", "resident_phi"],
    how="left",
    validate="one_to_one",
)

display_columns = [
    "k", "resident_theta", "resident_phi", "resident_payoff",
    "maximum_unilateral_gain", "resident_status",
    "maximum_mutant_advantage_at_epsilon_reference",
    "strongest_mutant_theta", "strongest_mutant_phi",
    "number_of_robust_invaders", "number_of_neutral_mutants",
]
resident_display = resident_results[display_columns].copy()
for column in (
    "resident_theta", "resident_phi", "resident_payoff",
    "maximum_unilateral_gain",
    "maximum_mutant_advantage_at_epsilon_reference",
    "strongest_mutant_theta", "strongest_mutant_phi",
):
    resident_display[column] = resident_display[column].map(
        lambda value: f"{value:.10g}"
    )
display(resident_display)
"""
        ),
        nbformat.v4.new_markdown_cell(
            """The two `not_ESS` results are finite-grid Nash residents whose
order-zero mutant advantage is tied. They fail only when higher-order terms
capture encounters among mutants. This is the direct computational distinction
between Nash equilibrium and evolutionary stability."""
        ),
        nbformat.v4.new_code_cell(
            """invaders = (
    details.loc[
        (details["search_stage"] == "coarse")
        & (details["rare_mutant_classification"] == "rare_mutant_invades")
    ]
    .sort_values("DeltaPi_epsilon_reference", ascending=False)
    .drop_duplicates([
        "k", "resident_theta", "resident_phi", "mutant_theta", "mutant_phi"
    ])
)

strongest_invaders = (
    invaders.groupby(["k", "resident_theta", "resident_phi"], as_index=False)
    .head(1)[[
        "k", "resident_theta", "resident_phi", "mutant_theta", "mutant_phi",
        "leading_nonzero_order", "leading_coefficient",
        "Pi_R_epsilon_reference", "Pi_M_epsilon_reference",
        "DeltaPi_epsilon_reference",
    ]]
)
display(strongest_invaders.round(10))
"""
        ),
        nbformat.v4.new_markdown_cell(
            "## Fixed-group diagnostic versus population ESS"
        ),
        nbformat.v4.new_code_cell(
            """game = KPlayerEWLGame(k=5, gamma=math.pi / 2)
fixed_group = exactly_one_mutant_group_diagnostic(game, EWL_C, EWL_D)
population = analyze_resident_mutant(
    game=game,
    resident=EWL_C,
    mutant=EWL_D,
    epsilon_reference=0.01,
)
rare = population.position_results[0]

comparison = pd.Series({
    "fixed_group_mutant_payoff": fixed_group.mutant_payoff,
    "fixed_group_resident_payoff": fixed_group.resident_average_payoff,
    "fixed_group_difference": fixed_group.mutant_minus_group_resident,
    "population_leading_order": rare.leading_nonzero_order,
    "population_leading_advantage": rare.leading_coefficient,
    "population_classification": rare.rare_mutant_classification,
})
display(comparison.to_frame("value"))
"""
        ),
        nbformat.v4.new_markdown_cell(
            """The fixed group returns `20-9=11`. The rare-population leading
term is `20-12=8`, because a resident focal individual in an almost entirely
resident population normally meets four cooperators and earns 12. Step 5 uses
the latter population definition."""
        ),
        nbformat.v4.new_markdown_cell("## Mutant invasion landscapes"),
        nbformat.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for axis, k in zip(axes.flat, (2, 3, 4, 5)):
    image = plt.imread(RESULTS / "figures" / f"k{k}_mutant_invasion_landscape.png")
    axis.imshow(image)
    axis.axis("off")
plt.show()
"""
        ),
        nbformat.v4.new_markdown_cell(
            """Blue indicates negative mutant advantage at `epsilon=0.01`, white
is near zero, and orange is positive. The star marks the highest-payoff
symmetric Nash resident selected for each `K`. The `K=3` landscape visibly
contains an invading region. The color value is diagnostic; rare-limit labels
still come from polynomial coefficients."""
        ),
        nbformat.v4.new_markdown_cell("## Coarse-to-fine and continuous evidence"),
        nbformat.v4.new_code_cell(
            """evidence = resident_results[[
    "k", "resident_theta", "resident_phi", "coarse_grid_result",
    "fine_grid_result", "continuous_optimizer_result", "permutation_error",
]].copy()
display(evidence)

optimizer_summary = optimizers.groupby(
    ["k", "resident_theta", "resident_phi"], as_index=False
).agg(
    objectives=("objective_epsilon", "count"),
    all_global_runs_converged=("differential_evolution_success", "all"),
    total_global_evaluations=("differential_evolution_evaluations", "sum"),
    maximum_objective_advantage=("best_objective_advantage", "max"),
)
display(optimizer_summary.round(10))
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Takeaways and limitations

1. A symmetric finite-grid Nash equilibrium can fail the multiplayer
   rare-mutant test at order 1 or even order 3. Nash is necessary for a pure
   ESS candidate but is not sufficient.
2. The `K=5` resident `(0,2pi/5)` remains the strongest Step 5 candidate: all
   residents earn 12 and no positive invader was found. Its current status is
   still conservative because numerical near-self neutrality prevents a strict
   claim.
3. These are pure-strategy numerical results in the restricted EWL family and
   this project's global K-body entangler. They do not establish an analytical
   result over full `SU(2)`, mixed strategies, other multiplayer entanglers, or
   noise.
4. Exact statevectors remove shot uncertainty, but optimizer coverage and the
   `1e-8` classification tolerance remain numerical limitations.
5. The next defensible extension is to preserve these baseline definitions
   while sweeping entanglement; noise should be introduced only in the later
   robustness model and reported with its own uncertainty assumptions.
"""
        ),
        nbformat.v4.new_markdown_cell("## Reproducibility metadata"),
        nbformat.v4.new_code_cell(
            """print(json.dumps(metadata["parameters"], indent=2, sort_keys=True))
print("Authoritative baseline:", metadata["authoritative_baseline_commit"])
print("Working-tree HEAD at run:", metadata["working_tree_head_at_run"])
print("Command:", " ".join(metadata["command"]))
print("Python:", metadata["python"].splitlines()[0])
print("Platform:", metadata["platform"])
"""
        ),
    ]
    return notebook


if __name__ == "__main__":
    main()
