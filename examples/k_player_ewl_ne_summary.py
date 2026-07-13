import argparse
import math

from quantum_ess import ewl_strategy_grid, find_pure_nash_equilibria_on_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print finite-grid K-player EWL Nash equilibrium result snippets."
    )
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
    parser.add_argument(
        "--max-results-per-k",
        type=int,
        default=20,
        help="maximum NE rows printed for each K",
    )
    return parser.parse_args()


def format_angles(equilibrium):
    return tuple(
        (round(strategy.theta, 6), round(strategy.phi, 6))
        for strategy in equilibrium.strategies
    )


def format_payoffs(equilibrium):
    return tuple(round(value, 6) for value in equilibrium.payoffs)


def print_k_summary(k, result):
    print(f"K = {k}")
    print("  strategy grid size:", result.strategy_count)
    print("  opponent contexts checked:", result.opponent_context_count)
    print("  resident count profiles checked:", result.resident_count_profile_count)
    print("  NE found:", len(result.equilibria))
    print("  truncated:", result.truncated)

    if not result.equilibria:
        print("  no finite-grid pure NE found")
        print()
        return

    for index, equilibrium in enumerate(result.equilibria, start=1):
        print(f"  NE #{index}")
        print("    strategy indices:", equilibrium.strategy_indices)
        print("    strategy counts:", equilibrium.strategy_counts)
        print("    symmetric:", equilibrium.is_symmetric)
        print("    angles theta_phi:", format_angles(equilibrium))
        print("    payoffs:", format_payoffs(equilibrium))
    print()


def main():
    args = parse_args()
    grid = ewl_strategy_grid(
        theta_count=args.theta_count,
        phi_count=args.phi_count,
    )

    print("K-player EWL finite-grid NE summary")
    print("gamma:", args.gamma)
    print("theta_count:", args.theta_count)
    print("phi_count:", args.phi_count)
    print("strategies per player:", len(grid))
    print("workers:", args.workers)
    print()

    for k in (2, 3, 4, 5):
        result = find_pure_nash_equilibria_on_grid(
            k=k,
            gamma=args.gamma,
            strategies=grid.strategies,
            workers=args.workers,
            chunksize=args.chunksize,
            max_results=args.max_results_per_k,
        )
        print_k_summary(k, result)


if __name__ == "__main__":
    main()
