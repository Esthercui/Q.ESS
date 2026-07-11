import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_ess import (
    EWL_C,
    EWL_D,
    EWL_Q,
    EWLStrategy,
    EWLTwoPlayerGame,
    classical_action_to_ewl_strategy,
)


class TestTwoPlayerEWLGame(unittest.TestCase):
    def assertComplexAlmostEqual(self, actual, expected, places=12):
        self.assertAlmostEqual(actual.real, expected.real, places=places)
        self.assertAlmostEqual(actual.imag, expected.imag, places=places)

    def assertProbabilityDistribution(self, result):
        self.assertAlmostEqual(result.probability_sum, 1.0, places=12)
        for probability in result.probabilities.values():
            self.assertGreaterEqual(probability, -1e-12)
            self.assertLessEqual(probability, 1.0 + 1e-12)

    def test_classical_c_and_d_are_special_quantum_operations(self):
        c_matrix = EWL_C.matrix()
        d_matrix = EWL_D.matrix()

        self.assertComplexAlmostEqual(c_matrix[0][0], 1.0 + 0j)
        self.assertComplexAlmostEqual(c_matrix[0][1], 0.0 + 0j)
        self.assertComplexAlmostEqual(c_matrix[1][0], 0.0 + 0j)
        self.assertComplexAlmostEqual(c_matrix[1][1], 1.0 + 0j)

        self.assertComplexAlmostEqual(d_matrix[0][0], 0.0 + 0j)
        self.assertComplexAlmostEqual(d_matrix[0][1], 1.0 + 0j)
        self.assertComplexAlmostEqual(d_matrix[1][0], -1.0 + 0j)
        self.assertComplexAlmostEqual(d_matrix[1][1], 0.0 + 0j)

        self.assertIs(classical_action_to_ewl_strategy("C"), EWL_C)
        self.assertIs(classical_action_to_ewl_strategy("D"), EWL_D)

    def test_gamma_zero_matches_classical_behavior(self):
        game = EWLTwoPlayerGame(gamma=0.0)
        cases = {
            (EWL_C, EWL_C): ((0, 0), (3.0, 3.0)),
            (EWL_C, EWL_D): ((0, 1), (0.0, 5.0)),
            (EWL_D, EWL_C): ((1, 0), (5.0, 0.0)),
            (EWL_D, EWL_D): ((1, 1), (1.0, 1.0)),
        }

        for strategies, (expected_outcome, expected_payoffs) in cases.items():
            result = game.run(*strategies)
            self.assertProbabilityDistribution(result)
            self.assertAlmostEqual(result.probabilities[expected_outcome], 1.0)
            self.assertEqual(result.expected_payoffs, expected_payoffs)

    def test_classical_profiles_recover_classical_payoffs_under_entanglement(self):
        cases = {
            (EWL_C, EWL_C): ((0, 0), (3.0, 3.0)),
            (EWL_C, EWL_D): ((0, 1), (0.0, 5.0)),
            (EWL_D, EWL_C): ((1, 0), (5.0, 0.0)),
            (EWL_D, EWL_D): ((1, 1), (1.0, 1.0)),
        }

        for gamma in (math.pi / 8.0, math.pi / 4.0, math.pi / 2.0):
            game = EWLTwoPlayerGame(gamma=gamma)
            for strategies, (expected_outcome, expected_payoffs) in cases.items():
                result = game.run(*strategies)
                self.assertProbabilityDistribution(result)
                self.assertAlmostEqual(result.probabilities[expected_outcome], 1.0)
                self.assertEqual(result.expected_payoffs, expected_payoffs)

    def test_probabilities_sum_to_one_for_quantum_strategies(self):
        game = EWLTwoPlayerGame(gamma=math.pi / 2.0)
        result = game.run(
            EWLStrategy(theta=0.7, phi=0.2),
            EWLStrategy(theta=1.1, phi=0.4),
        )

        self.assertProbabilityDistribution(result)
        for payoff in result.expected_payoffs:
            self.assertIsInstance(payoff, float)

    def test_symmetric_quantum_profile_has_symmetric_payoffs(self):
        game = EWLTwoPlayerGame(gamma=math.pi / 2.0)
        result = game.run(EWL_Q, EWL_Q)

        self.assertProbabilityDistribution(result)
        self.assertAlmostEqual(result.expected_payoffs[0], result.expected_payoffs[1])

    def test_invalid_payoff_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            EWLTwoPlayerGame(payoff_matrix={(0, 0): (1.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
