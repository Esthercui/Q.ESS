from quantum_ess import classical_k_player_baseline


def main():
    baseline = classical_k_player_baseline(k=5)

    print(f"K-player classical baseline, K={baseline.k}")
    print("Payoff by total cooperators")
    print("  cooperators | cooperator payoff | defector payoff")
    for row in baseline.payoff_table:
        print(
            f"  {row.cooperators:11d} | "
            f"{str(row.cooperator_payoff):17s} | "
            f"{str(row.defector_payoff):14s}"
        )

    print("Resident-mutant invasion checks")
    for label, result in (
        ("one D mutant in C residents", baseline.one_defector_in_cooperators),
        ("one C mutant in D residents", baseline.one_cooperator_in_defectors),
    ):
        print(f"  {label}")
        print(f"    resident payoff: {result.resident_payoff}")
        print(f"    mutant payoff:   {result.mutant_payoff}")
        print(f"    strength:        {result.invasion_strength}")
        print(f"    invades:         {result.invades}")


if __name__ == "__main__":
    main()
