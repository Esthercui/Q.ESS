import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_ess import (
    EWL_Q,
    GRID_ESS_CANDIDATE,
    NOT_ESS,
    deduplicate_ewl_strategies,
    ewl_strategy_grid,
    refined_mutant_grid,
    search_mutants_on_grid,
    symmetric_nash_candidates_on_grid,
)


class TestEWLInvasionSearch(unittest.TestCase):
    def setUp(self):
        self.coarse_grid = deduplicate_ewl_strategies(
            ewl_strategy_grid(theta_count=11, phi_count=6).strategies
        )

    def test_symmetric_nash_screen_matches_validated_step4_candidates(self):
        expected_counts = {2: 1, 3: 1, 4: 2, 5: 2}
        candidates = {
            k: symmetric_nash_candidates_on_grid(
                k=k,
                gamma=math.pi / 2.0,
                strategies=self.coarse_grid,
            )
            for k in (2, 3, 4, 5)
        }

        self.assertEqual(
            {k: len(rows) for k, rows in candidates.items()},
            expected_counts,
        )
        self.assertEqual(
            (candidates[5][0].resident.theta, candidates[5][0].resident.phi),
            (0.0, 2.0 * math.pi / 5.0),
        )
        self.assertAlmostEqual(candidates[5][0].resident_payoff, 12.0)

    def test_coarse_to_fine_grid_exposes_closer_mutants_without_invasion(self):
        coarse = search_mutants_on_grid(
            k=2,
            gamma=math.pi / 2.0,
            resident=EWL_Q,
            mutant_strategies=self.coarse_grid,
            epsilon_reference=0.01,
            stage="coarse",
            positions=(0,),
        )
        fine_strategies = refined_mutant_grid(
            theta_count=21,
            phi_count=11,
            centers=(EWL_Q, coarse.strongest_mutant),
        )
        fine = search_mutants_on_grid(
            k=2,
            gamma=math.pi / 2.0,
            resident=EWL_Q,
            mutant_strategies=fine_strategies,
            epsilon_reference=0.01,
            stage="fine",
            positions=(0,),
        )

        self.assertEqual(coarse.resident_status, GRID_ESS_CANDIDATE)
        self.assertEqual(fine.resident_status, GRID_ESS_CANDIDATE)
        self.assertGreater(fine.mutant_strategy_count, coarse.mutant_strategy_count)
        self.assertGreaterEqual(
            fine.maximum_mutant_advantage_at_epsilon_reference,
            coarse.maximum_mutant_advantage_at_epsilon_reference - 1e-8,
        )

    def test_higher_order_invader_is_detected_for_k3_candidate(self):
        resident = symmetric_nash_candidates_on_grid(
            k=3,
            gamma=math.pi / 2.0,
            strategies=self.coarse_grid,
        )[0].resident
        result = search_mutants_on_grid(
            k=3,
            gamma=math.pi / 2.0,
            resident=resident,
            mutant_strategies=self.coarse_grid,
            epsilon_reference=0.01,
            stage="coarse",
            positions=(0,),
        )

        self.assertEqual(result.resident_status, NOT_ESS)
        self.assertGreater(result.number_of_robust_invaders, 0)
        self.assertTrue(
            any(
                pair.worst_reference_result.leading_nonzero_order == 1
                for pair in result.pair_results
                if pair.rare_mutant_classification == "rare_mutant_invades"
            )
        )

    def test_parallel_grid_search_matches_serial_result(self):
        small_grid = deduplicate_ewl_strategies(
            ewl_strategy_grid(theta_count=2, phi_count=1).strategies
        )
        serial = search_mutants_on_grid(
            k=3,
            gamma=0.0,
            resident=small_grid[0],
            mutant_strategies=small_grid,
            epsilon_reference=0.01,
            stage="serial",
            positions=(0,),
        )
        parallel = search_mutants_on_grid(
            k=3,
            gamma=0.0,
            resident=small_grid[0],
            mutant_strategies=small_grid,
            epsilon_reference=0.01,
            stage="parallel",
            positions=(0,),
            workers=2,
            chunksize=1,
        )

        self.assertEqual(serial.resident_status, parallel.resident_status)
        self.assertAlmostEqual(
            serial.maximum_mutant_advantage_at_epsilon_reference,
            parallel.maximum_mutant_advantage_at_epsilon_reference,
            places=12,
        )

    def test_continuous_optimizer_is_not_worse_than_coarse_grid(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("optional research dependencies are not installed")

        from quantum_ess import continuous_mutant_search

        coarse = search_mutants_on_grid(
            k=2,
            gamma=math.pi / 2.0,
            resident=EWL_Q,
            mutant_strategies=self.coarse_grid,
            epsilon_reference=0.01,
            stage="coarse",
            positions=(0,),
        )
        continuous = continuous_mutant_search(
            k=2,
            gamma=math.pi / 2.0,
            resident=EWL_Q,
            epsilon_reference=0.01,
            seed_strategies=(coarse.strongest_mutant,),
            objective_frequencies=(0.01,),
            differential_maxiter=20,
            differential_popsize=6,
            local_start_limit=4,
            seed=17,
        )

        self.assertGreaterEqual(
            continuous.maximum_mutant_advantage_at_epsilon_reference,
            coarse.maximum_mutant_advantage_at_epsilon_reference - 1e-8,
        )
        self.assertLessEqual(continuous.maximum_permutation_error, 1e-8)


if __name__ == "__main__":
    unittest.main()
