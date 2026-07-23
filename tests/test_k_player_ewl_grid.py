import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_ess import (
    build_symmetric_best_response_table,
    evaluate_profile_on_grid,
    ewl_strategy_grid,
    find_pure_nash_equilibria_on_grid,
    iter_ordered_profile_indices,
    iter_sparse_count_profiles,
    ordered_profile_count,
    sparse_count_profile_count,
    sparse_counts_to_indices,
    subtract_one_from_sparse_counts,
)


class TestKPlayerEWLGrid(unittest.TestCase):
    def test_default_grid_matches_notebook_size(self):
        grid = ewl_strategy_grid(theta_count=11, phi_count=6)

        self.assertEqual(len(grid), 66)
        self.assertEqual(grid.index_angles(0), (0.0, 0.0))
        self.assertAlmostEqual(grid.theta_values[-1], math.pi)
        self.assertAlmostEqual(grid.phi_values[-1], math.pi / 2.0)

    def test_sparse_count_profiles_represent_combinations_with_repetition(self):
        profiles = tuple(iter_sparse_count_profiles(strategy_count=3, total=2))

        self.assertEqual(len(profiles), 6)
        self.assertEqual(sparse_count_profile_count(strategy_count=3, total=2), 6)
        self.assertIn(((0, 2),), profiles)
        self.assertIn(((0, 1), (2, 1)), profiles)
        self.assertIn(((2, 2),), profiles)

    def test_sparse_counts_expand_and_subtract(self):
        counts = ((0, 2), (3, 1))

        self.assertEqual(sparse_counts_to_indices(counts), (0, 0, 3))
        self.assertEqual(subtract_one_from_sparse_counts(counts, 0), ((0, 1), (3, 1)))
        self.assertEqual(subtract_one_from_sparse_counts(counts, 3), ((0, 2),))

    def test_ordered_profile_iterator_keeps_full_profile_count_available(self):
        profiles = tuple(iter_ordered_profile_indices(strategy_count=2, k=3))

        self.assertEqual(len(profiles), 8)
        self.assertEqual(ordered_profile_count(strategy_count=2, k=3), 8)
        self.assertIn((0, 1, 0), profiles)

    def test_direct_deviation_audit_rejects_one_defector_profile_as_nash(self):
        grid = ewl_strategy_grid(theta_count=2, phi_count=1)
        audit = evaluate_profile_on_grid(
            k=5,
            gamma=0.0,
            strategies=grid.strategies,
            strategy_indices=(1, 0, 0, 0, 0),
        )

        self.assertFalse(audit.is_nash)
        self.assertEqual(audit.payoffs, (20.0, 9.0, 9.0, 9.0, 9.0))
        self.assertEqual(
            audit.average_pairwise_payoffs,
            (5.0, 2.25, 2.25, 2.25, 2.25),
        )
        self.assertEqual(audit.best_response_indices[0], (1,))
        self.assertEqual(audit.unilateral_gains[0], 0.0)
        for player in range(1, 5):
            self.assertEqual(audit.best_response_indices[player], (1,))
            self.assertEqual(audit.unilateral_gains[player], 7.0)

    def test_direct_deviation_audit_accepts_all_defection_as_nash(self):
        grid = ewl_strategy_grid(theta_count=2, phi_count=1)
        audit = evaluate_profile_on_grid(
            k=5,
            gamma=math.pi / 2.0,
            strategies=grid.strategies,
            strategy_indices=(1, 1, 1, 1, 1),
        )

        self.assertTrue(audit.is_nash)
        self.assertEqual(audit.payoffs, (4.0, 4.0, 4.0, 4.0, 4.0))
        self.assertAlmostEqual(audit.probability_sum, 1.0, places=12)
        for gain in audit.unilateral_gains:
            self.assertAlmostEqual(gain, 0.0, places=12)

    def test_best_response_table_finds_defection_against_cooperators(self):
        grid = ewl_strategy_grid(theta_count=2, phi_count=1)
        table = build_symmetric_best_response_table(
            k=3,
            gamma=0.0,
            strategies=grid.strategies,
        )

        best_value, best_indices = table[((0, 2),)]
        self.assertEqual(best_indices, (1,))
        self.assertEqual(best_value, 10.0)

    def test_classical_k5_grid_has_all_defection_nash_equilibrium(self):
        grid = ewl_strategy_grid(theta_count=2, phi_count=1)
        result = find_pure_nash_equilibria_on_grid(
            k=5,
            gamma=0.0,
            strategies=grid.strategies,
        )

        self.assertFalse(result.truncated)
        self.assertEqual(result.strategy_count, 2)
        self.assertEqual(result.ordered_profile_count, 32)
        self.assertEqual(result.opponent_context_count, 5)
        self.assertEqual(result.resident_count_profile_count, 6)
        self.assertEqual(len(result.equilibria), 1)
        equilibrium = result.equilibria[0]
        self.assertEqual(equilibrium.strategy_indices, (1, 1, 1, 1, 1))
        self.assertTrue(equilibrium.is_symmetric)
        self.assertTrue(equilibrium.is_nash)
        self.assertEqual(equilibrium.payoffs, (4.0, 4.0, 4.0, 4.0, 4.0))
        self.assertEqual(
            equilibrium.average_pairwise_payoffs,
            (1.0, 1.0, 1.0, 1.0, 1.0),
        )
        self.assertEqual(equilibrium.best_response_indices, ((1,),) * 5)

    def test_max_results_can_truncate_large_result_sets(self):
        grid = ewl_strategy_grid(theta_count=1, phi_count=1)
        result = find_pure_nash_equilibria_on_grid(
            k=5,
            gamma=0.0,
            strategies=grid.strategies,
            max_results=1,
        )

        self.assertTrue(result.truncated)
        self.assertEqual(len(result.equilibria), 1)


if __name__ == "__main__":
    unittest.main()
