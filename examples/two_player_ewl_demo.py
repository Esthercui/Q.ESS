import math

from quantum_ess import EWL_C, EWL_D, EWL_Q, EWLStrategy, EWLTwoPlayerGame


def print_result(label, result):
    outcome, probability = result.most_likely_outcome()
    print(label)
    print("  payoffs:", tuple(round(value, 6) for value in result.expected_payoffs))
    print("  most likely outcome:", outcome, round(probability, 6))
    print("  probability sum:", round(result.probability_sum, 12))


def main():
    classical_limit = EWLTwoPlayerGame(gamma=0.0)
    entangled_game = EWLTwoPlayerGame(gamma=math.pi / 2.0)

    print("Classical C/D profiles at gamma = 0")
    for label, strategies in (
        ("CC", (EWL_C, EWL_C)),
        ("CD", (EWL_C, EWL_D)),
        ("DC", (EWL_D, EWL_C)),
        ("DD", (EWL_D, EWL_D)),
    ):
        print_result(label, classical_limit.run(*strategies))

    print("Entangled quantum examples at gamma = pi/2")
    print_result("QQ", entangled_game.run(EWL_Q, EWL_Q))
    print_result(
        "U(0.7, 0.2) vs U(1.1, 0.4)",
        entangled_game.run(
            EWLStrategy(theta=0.7, phi=0.2),
            EWLStrategy(theta=1.1, phi=0.4),
        ),
    )


if __name__ == "__main__":
    main()
