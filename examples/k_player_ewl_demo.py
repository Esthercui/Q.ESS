import math

from quantum_ess import (
    EWL_C,
    EWL_D,
    EWL_Q,
    EWLStrategy,
    KPlayerEWLGame,
    k_player_resident_mutant_profile,
)


def print_result(label, result):
    outcome, probability = result.most_likely_outcome()
    print(label)
    print("  payoffs:", tuple(round(value, 6) for value in result.expected_payoffs))
    print("  most likely outcome:", outcome, round(probability, 6))
    print("  probability sum:", round(result.probability_sum, 12))


def main():
    print("Payoff examples only; this script does not search for Nash equilibria.")
    game = KPlayerEWLGame(k=5, gamma=math.pi / 4.0)

    profiles = {
        "all C": [EWL_C] * 5,
        "all D": [EWL_D] * 5,
        "all Q": [EWL_Q] * 5,
        "one D mutant in C residents": k_player_resident_mutant_profile(
            k=5,
            resident=EWL_C,
            mutant=EWL_D,
            mutant_indices=(0,),
        ),
        "two quantum mutants in C residents": k_player_resident_mutant_profile(
            k=5,
            resident=EWL_C,
            mutant=EWLStrategy(theta=0.7, phi=0.2, label="M"),
            mutant_indices=(1, 3),
        ),
    }

    for label, profile in profiles.items():
        print_result(label, game.run(profile))

    mutant_payoff, resident_payoff, _ = game.resident_mutant_payoffs(
        resident=EWL_C,
        mutant=EWL_D,
        mutant_indices=(0,),
    )
    print("Average resident-mutant payoffs")
    print("  mutant average:", round(mutant_payoff, 6))
    print("  resident average:", round(resident_payoff, 6))


if __name__ == "__main__":
    main()
