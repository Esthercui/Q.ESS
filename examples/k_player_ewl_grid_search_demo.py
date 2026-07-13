import argparse
import math

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
    parser.add_argument("--theta-count", type=int, default=3)
    parser.add_argument("--phi-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--max-results", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    grid = ewl_strategy_grid(theta_count=args.theta_count, phi_count=args.phi_count)
    print("K-player EWL grid search")
    print("  K:", args.k)
    print("  gamma:", args.gamma)
    print("  strategies:", len(grid))
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
    print("  opponent contexts:", result.opponent_context_count)
    print("  resident count profiles:", result.resident_count_profile_count)
    print("Equilibria found:", len(result.equilibria))
    print("Truncated:", result.truncated)

    for index, equilibrium in enumerate(result.equilibria, start=1):
        rounded_angles = tuple(
            (round(theta, 6), round(phi, 6))
            for theta, phi in equilibrium.angles()
        )
        rounded_payoffs = tuple(round(value, 6) for value in equilibrium.payoffs)
        print(f"NE #{index}")
        print("  strategy indices:", equilibrium.strategy_indices)
        print("  strategy counts:", equilibrium.strategy_counts)
        print("  angles:", rounded_angles)
        print("  payoffs:", rounded_payoffs)


if __name__ == "__main__":
    main()
