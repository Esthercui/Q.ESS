from quantum_ess import classical_two_player_baseline


def main():
    baseline = classical_two_player_baseline()

    print("Payoff matrix")
    for profile, payoff in sorted(baseline.payoff_matrix.items()):
        print(f"  {profile}: {payoff}")

    print("Pure Nash equilibria:", baseline.nash_equilibria)

    print("ESS results")
    for result in baseline.ess_results:
        status = "ESS" if result.is_ess else "not ESS"
        print(f"  {result.resident}: {status}")
        for comparison in result.comparisons:
            print(
                "    mutant"
                f" {comparison.mutant}: advantage={comparison.invasion_strength:.6g},"
                f" blocks_ess={comparison.blocks_ess}"
            )


if __name__ == "__main__":
    main()
