import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_ess import ClassicalKPlayerGame, classical_k_player_baseline


class TestClassicalKPlayerBaseline(unittest.TestCase):
    def test_k_equals_two_recovers_step_one_payoff_matrix(self):
        game = ClassicalKPlayerGame(k=2)

        self.assertEqual(game.profile_payoffs(("C", "C")), (3.0, 3.0))
        self.assertEqual(game.profile_payoffs(("C", "D")), (0.0, 5.0))
        self.assertEqual(game.profile_payoffs(("D", "C")), (5.0, 0.0))
        self.assertEqual(game.profile_payoffs(("D", "D")), (1.0, 1.0))

    def test_payoff_table_is_function_of_total_cooperators(self):
        baseline = classical_k_player_baseline(k=5)
        rows = {row.cooperators: row for row in baseline.payoff_table}

        self.assertIsNone(rows[0].cooperator_payoff)
        self.assertEqual(rows[0].defector_payoff, 4.0)

        self.assertEqual(rows[4].cooperator_payoff, 9.0)
        self.assertEqual(rows[4].defector_payoff, 20.0)

        self.assertEqual(rows[5].cooperator_payoff, 12.0)
        self.assertIsNone(rows[5].defector_payoff)

    def test_one_defector_invades_cooperators(self):
        baseline = classical_k_player_baseline(k=5)
        invasion = baseline.one_defector_in_cooperators

        self.assertEqual(invasion.resident, "C")
        self.assertEqual(invasion.mutant, "D")
        self.assertEqual(invasion.cooperators, 4)
        self.assertEqual(invasion.resident_payoff, 9.0)
        self.assertEqual(invasion.mutant_payoff, 20.0)
        self.assertEqual(invasion.invasion_strength, 11.0)
        self.assertTrue(invasion.invades)

    def test_one_cooperator_does_not_invade_defectors(self):
        baseline = classical_k_player_baseline(k=5)
        invasion = baseline.one_cooperator_in_defectors

        self.assertEqual(invasion.resident, "D")
        self.assertEqual(invasion.mutant, "C")
        self.assertEqual(invasion.cooperators, 1)
        self.assertEqual(invasion.resident_payoff, 8.0)
        self.assertEqual(invasion.mutant_payoff, 0.0)
        self.assertEqual(invasion.invasion_strength, -8.0)
        self.assertFalse(invasion.invades)

    def test_same_resident_and_mutant_have_equal_payoff(self):
        game = ClassicalKPlayerGame(k=5)
        invasion = game.resident_mutant_invasion(resident="D", mutant="D")

        self.assertEqual(invasion.resident_payoff, invasion.mutant_payoff)
        self.assertEqual(invasion.invasion_strength, 0.0)
        self.assertFalse(invasion.invades)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ClassicalKPlayerGame(k=1)

        game = ClassicalKPlayerGame(k=3)
        with self.assertRaises(ValueError):
            game.payoff_for_action("C", 0)
        with self.assertRaises(ValueError):
            game.payoff_for_action("D", 3)
        with self.assertRaises(ValueError):
            game.resident_mutant_invasion("C", "D", mutant_count=3)


if __name__ == "__main__":
    unittest.main()
