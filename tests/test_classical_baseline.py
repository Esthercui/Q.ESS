import unittest

from quantum_ess import classical_two_player_baseline, prisoners_dilemma_game


class TestClassicalTwoPlayerBaseline(unittest.TestCase):
    def test_default_pd_payoff_matrix(self):
        baseline = classical_two_player_baseline()

        self.assertEqual(baseline.payoff_matrix[("C", "C")], (3.0, 3.0))
        self.assertEqual(baseline.payoff_matrix[("C", "D")], (0.0, 5.0))
        self.assertEqual(baseline.payoff_matrix[("D", "C")], (5.0, 0.0))
        self.assertEqual(baseline.payoff_matrix[("D", "D")], (1.0, 1.0))

    def test_default_pd_has_defection_nash_equilibrium(self):
        baseline = classical_two_player_baseline()

        self.assertEqual(baseline.nash_equilibria, (("D", "D"),))

    def test_default_pd_ess_result(self):
        baseline = classical_two_player_baseline()

        cooperation = baseline.ess_for("C")
        defection = baseline.ess_for("D")

        self.assertFalse(cooperation.is_ess)
        self.assertEqual(cooperation.max_mutant_advantage, 2.0)
        self.assertEqual(cooperation.comparisons[0].mutant, "D")
        self.assertTrue(cooperation.comparisons[0].blocks_ess)

        self.assertTrue(defection.is_ess)
        self.assertEqual(defection.max_mutant_advantage, -1.0)
        self.assertEqual(defection.comparisons[0].mutant, "C")
        self.assertFalse(defection.comparisons[0].blocks_ess)

    def test_custom_pd_payoffs_keep_expected_defection_result(self):
        game = prisoners_dilemma_game(
            reward=4.0,
            sucker=-1.0,
            temptation=6.0,
            punishment=0.5,
        )
        baseline = game.baseline()

        self.assertEqual(baseline.nash_equilibria, (("D", "D"),))
        self.assertFalse(baseline.ess_for("C").is_ess)
        self.assertTrue(baseline.ess_for("D").is_ess)

    def test_invalid_resident_is_rejected(self):
        game = prisoners_dilemma_game()

        with self.assertRaises(ValueError):
            game.ess_result("X")


if __name__ == "__main__":
    unittest.main()
