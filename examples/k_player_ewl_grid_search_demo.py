import argparse
import json
import math
from pathlib import Path

from quantum_ess import ewl_strategy_grid, find_pure_nash_equilibria_on_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a finite-grid pure Nash search for the K-player EWL game."
    )
    parser.add_argument("--k", type=int, default=5, help="number of players")
    parser.add_argument(
        "--gamma",
        type=float,
        default=math.pi / 2.0,
        help="entanglement value in radians",
    )
    parser.add_argument("--theta-count", type=int, default=11)
    parser.add_argument("--phi-count", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="optional output limit; omit it to return every ordered NE profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON file for the complete result",
    )
    return parser.parse_args()


def equilibrium_record(number, equilibrium):
    players = []
    for player, strategy in enumerate(equilibrium.strategies, start=1):
        players.append(
            {
                "player": player,
                "strategy_index": equilibrium.strategy_indices[player - 1],
                "theta": strategy.theta,
                "phi": strategy.phi,
                "total_expected_payoff": equilibrium.payoffs[player - 1],
                "average_payoff_per_opponent": (
                    equilibrium.average_pairwise_payoffs[player - 1]
                ),
                "best_response_indices": equilibrium.best_response_indices[player - 1],
                "maximum_unilateral_gain": equilibrium.unilateral_gains[player - 1],
            }
        )
    return {
        "number": number,
        "is_nash_equilibrium": equilibrium.is_nash,
        "is_symmetric": equilibrium.is_symmetric,
        "strategy_profile": equilibrium.strategy_indices,
        "players": players,
        "probability_sum": equilibrium.probability_sum,
        "most_likely_outcome": equilibrium.most_likely_outcome,
        "most_likely_probability": equilibrium.most_likely_probability,
    }


def main():
    args = parse_args()
    grid = ewl_strategy_grid(theta_count=args.theta_count, phi_count=args.phi_count)
    print("K-player EWL grid search")
    print("  K:", args.k)
    print("  gamma:", args.gamma)
    print("  theta values:", len(grid.theta_values))
    print("  phi values:", len(grid.phi_values))
    print("  pure strategies available to each player:", len(grid))
    print("  workers:", args.workers)

    result = find_pure_nash_equilibria_on_grid(
        k=args.k,
        gamma=args.gamma,
        strategies=grid.strategies,
        workers=args.workers,
        chunksize=args.chunksize,
        max_results=args.max_results,
    )

    print("Search size")
    print("  ordered profiles in finite game:", result.ordered_profile_count)
    print("  opponent contexts checked:", result.opponent_context_count)
    print("  symmetry-reduced profile classes checked:", result.resident_count_profile_count)
    print("Ordered Nash equilibrium profiles found:", len(result.equilibria))
    print("Output truncated:", result.truncated)

    for index, equilibrium in enumerate(result.equilibria, start=1):
        print(f"NE #{index}")
        print("  Nash equilibrium:", equilibrium.is_nash)
        print("  ordered strategy profile:", equilibrium.strategy_indices)
        print("  symmetric profile:", equilibrium.is_symmetric)
        for player, strategy in enumerate(equilibrium.strategies, start=1):
            print(f"  Player {player}")
            print("    strategy index:", equilibrium.strategy_indices[player - 1])
            print("    theta:", round(strategy.theta, 9))
            print("    phi:", round(strategy.phi, 9))
            print(
                "    total expected payoff:",
                round(equilibrium.payoffs[player - 1], 9),
            )
            print(
                "    average payoff per opponent:",
                round(equilibrium.average_pairwise_payoffs[player - 1], 9),
            )
            print(
                "    best-response indices:",
                equilibrium.best_response_indices[player - 1],
            )
            print(
                "    maximum unilateral gain:",
                round(equilibrium.unilateral_gains[player - 1], 12),
            )
        print("  probability sum:", round(equilibrium.probability_sum, 12))
        print(
            "  most likely outcome:",
            equilibrium.most_likely_outcome,
            round(equilibrium.most_likely_probability, 9),
        )

    if args.output is not None:
        payload = {
            "k": result.k,
            "gamma": result.gamma,
            "theta_values": grid.theta_values,
            "phi_values": grid.phi_values,
            "strategy_count_per_player": result.strategy_count,
            "ordered_profile_count": result.ordered_profile_count,
            "opponent_context_count": result.opponent_context_count,
            "symmetry_reduced_profile_count": result.resident_count_profile_count,
            "tolerance": result.tolerance,
            "truncated": result.truncated,
            "equilibria": [
                equilibrium_record(index, equilibrium)
                for index, equilibrium in enumerate(result.equilibria, start=1)
            ],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON result:", args.output)


if __name__ == "__main__":
    main()
