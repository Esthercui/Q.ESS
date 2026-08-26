import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_ess import (
    EWL_C,
    EWL_D,
    EWL_Q,
    NEUTRAL,
    RARE_MUTANT_INVADES,
    RARE_MUTANT_REJECTED,
    EWLStrategy,
    KPlayerEWLGame,
    analyze_resident_mutant,
    bernstein_to_power_coefficients,
    canonicalize_ewl_strategy,
    deduplicate_ewl_strategies,
    ewl_strategies_equivalent,
    ewl_strategy_grid,
    exactly_one_mutant_group_diagnostic,
    k_player_ewl_pd_payoff,
    population_fitness,
)


class TestEWLPopulationInvasion(unittest.TestCase):
    def test_same_resident_and_mutant_is_neutral_at_every_frequency(self):
        game = KPlayerEWLGame(k=5, gamma=math.pi / 2.0)
        strategy = EWLStrategy(theta=0.4, phi=0.3)
        analysis = analyze_resident_mutant(
            game,
            resident=strategy,
            mutant=strategy,
            epsilon_reference=0.01,
        )

        self.assertEqual(analysis.rare_mutant_classification, NEUTRAL)
        for result in analysis.position_results:
            self.assertEqual(result.rare_mutant_classification, NEUTRAL)
            self.assertTrue(
                all(abs(value) <= 1e-12 for value in result.power_coefficients)
            )
            self.assertAlmostEqual(result.mutant_advantage_reference, 0.0, places=12)
            for diagnostic in result.diagnostic_frequencies:
                self.assertAlmostEqual(diagnostic.mutant_advantage, 0.0, places=12)

    def test_classical_cooperation_is_invaded_and_defection_rejects_cooperation(self):
        game = KPlayerEWLGame(k=2, gamma=0.0)
        cooperation = analyze_resident_mutant(
            game, EWL_C, EWL_D, epsilon_reference=0.01
        ).position_results[0]
        defection = analyze_resident_mutant(
            game, EWL_D, EWL_C, epsilon_reference=0.01
        ).position_results[0]

        self.assertEqual(cooperation.rare_mutant_classification, RARE_MUTANT_INVADES)
        self.assertEqual(cooperation.leading_nonzero_order, 0)
        self.assertAlmostEqual(cooperation.leading_coefficient, 2.0)
        self.assertEqual(defection.rare_mutant_classification, RARE_MUTANT_REJECTED)
        self.assertEqual(defection.leading_nonzero_order, 0)
        self.assertAlmostEqual(defection.leading_coefficient, -1.0)

    def test_k2_equal_first_condition_uses_second_ess_condition(self):
        stable_payoffs = {
            (0, 0): (2.0, 2.0),
            (0, 1): (3.0, 2.0),
            (1, 0): (2.0, 3.0),
            (1, 1): (1.0, 1.0),
        }
        unstable_payoffs = {
            (0, 0): (2.0, 2.0),
            (0, 1): (1.0, 2.0),
            (1, 0): (2.0, 1.0),
            (1, 1): (3.0, 3.0),
        }

        def payoff_function(matrix):
            return lambda outcome: matrix[outcome]

        stable = analyze_resident_mutant(
            KPlayerEWLGame(k=2, gamma=0.0, payoff_fn=payoff_function(stable_payoffs)),
            EWL_C,
            EWL_D,
            epsilon_reference=0.01,
        ).position_results[0]
        unstable = analyze_resident_mutant(
            KPlayerEWLGame(k=2, gamma=0.0, payoff_fn=payoff_function(unstable_payoffs)),
            EWL_C,
            EWL_D,
            epsilon_reference=0.01,
        ).position_results[0]

        self.assertEqual(stable.power_coefficients, (0.0, -2.0))
        self.assertEqual(stable.leading_nonzero_order, 1)
        self.assertEqual(stable.rare_mutant_classification, RARE_MUTANT_REJECTED)
        self.assertEqual(unstable.power_coefficients, (0.0, 2.0))
        self.assertEqual(unstable.leading_nonzero_order, 1)
        self.assertEqual(unstable.rare_mutant_classification, RARE_MUTANT_INVADES)

    def test_population_fitness_uses_binomial_random_group_weights(self):
        game = KPlayerEWLGame(k=3, gamma=0.0)
        result = analyze_resident_mutant(
            game, EWL_C, EWL_D, epsilon_reference=0.2, positions=(0,)
        ).position_results[0]

        resident_values = tuple(
            row.resident_payoff for row in result.composition_payoffs
        )
        mutant_values = tuple(row.mutant_payoff for row in result.composition_payoffs)
        self.assertEqual(resident_values, (6.0, 3.0, 0.0))
        self.assertEqual(mutant_values, (10.0, 6.0, 2.0))
        self.assertAlmostEqual(population_fitness(resident_values, 0.2), 4.8)
        self.assertAlmostEqual(population_fitness(mutant_values, 0.2), 8.4)
        self.assertEqual(result.power_coefficients, (4.0, -2.0, 0.0))
        self.assertAlmostEqual(result.mutant_advantage_reference, 3.6)

    def test_bernstein_conversion_recovers_known_polynomial(self):
        coefficients = bernstein_to_power_coefficients((4.0, 3.0, 2.0))
        self.assertEqual(coefficients, (4.0, -2.0, 0.0))

    def test_fixed_group_diagnostic_is_not_population_ess_fitness(self):
        game = KPlayerEWLGame(k=5, gamma=0.0)
        group = exactly_one_mutant_group_diagnostic(game, EWL_C, EWL_D)
        population = analyze_resident_mutant(
            game, EWL_C, EWL_D, epsilon_reference=0.0, positions=(0,)
        ).position_results[0]

        self.assertEqual(group.mutant_payoff, 20.0)
        self.assertEqual(group.resident_average_payoff, 9.0)
        self.assertEqual(group.mutant_minus_group_resident, 11.0)
        self.assertEqual(population.mutant_payoff_reference, 20.0)
        self.assertEqual(population.resident_payoff_reference, 12.0)
        self.assertEqual(population.mutant_advantage_reference, 8.0)

    def test_theta_pi_phi_duplicates_are_canonicalized(self):
        duplicate_d = EWLStrategy(theta=math.pi, phi=math.pi / 2.0)
        canonical = canonicalize_ewl_strategy(duplicate_d)
        grid = ewl_strategy_grid(theta_count=11, phi_count=6)
        unique = deduplicate_ewl_strategies(grid.strategies)

        self.assertEqual((canonical.theta, canonical.phi), (math.pi, 0.0))
        self.assertTrue(ewl_strategies_equivalent(EWL_D, duplicate_d))
        self.assertEqual(len(grid.strategies), 66)
        self.assertEqual(len(unique), 61)

    def test_nearby_theta_pi_mutant_is_not_deduplicated(self):
        nearby = EWLStrategy(theta=math.pi - 1e-12, phi=0.0)
        self.assertFalse(ewl_strategies_equivalent(EWL_D, nearby))
        self.assertEqual(len(deduplicate_ewl_strategies((EWL_D, nearby))), 2)

    def test_arbitrary_profile_payoffs_match_direct_probability_enumeration(self):
        game = KPlayerEWLGame(k=3, gamma=math.pi / 2.0)
        profile = (
            EWLStrategy(theta=0.3, phi=0.1),
            EWLStrategy(theta=1.2, phi=0.4),
            EWL_Q,
        )
        result = game.run(profile)
        payoff_fn = k_player_ewl_pd_payoff(3)
        direct = tuple(
            sum(
                probability * payoff_fn(outcome)[player]
                for outcome, probability in result.probabilities.items()
            )
            for player in range(3)
        )

        self.assertAlmostEqual(result.probability_sum, 1.0, places=12)
        for actual, expected in zip(result.expected_payoffs, direct):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_population_results_are_permutation_invariant(self):
        game = KPlayerEWLGame(k=5, gamma=math.pi / 2.0)
        analysis = analyze_resident_mutant(
            game,
            EWLStrategy(theta=0.4, phi=0.2),
            EWLStrategy(theta=1.0, phi=0.6),
            epsilon_reference=0.01,
        )

        self.assertLessEqual(analysis.permutation_error, 1e-12)
        reference = analysis.position_results[0]
        for result in analysis.position_results[1:]:
            self.assertEqual(
                result.rare_mutant_classification,
                reference.rare_mutant_classification,
            )
            self.assertAlmostEqual(
                result.mutant_advantage_reference,
                reference.mutant_advantage_reference,
                places=12,
            )

    def test_existing_step4_k5_classical_result_is_unchanged(self):
        game = KPlayerEWLGame(k=5, gamma=math.pi / 2.0)
        result = game.run((EWL_D, EWL_C, EWL_C, EWL_C, EWL_C))
        self.assertEqual(result.expected_payoffs, (20.0, 9.0, 9.0, 9.0, 9.0))

    def test_restricted_domain_is_enforced_by_invasion_layer(self):
        game = KPlayerEWLGame(k=2, gamma=0.0)
        with self.assertRaises(ValueError):
            analyze_resident_mutant(
                game,
                EWL_C,
                EWLStrategy(theta=math.pi + 1e-6, phi=0.0),
                epsilon_reference=0.01,
            )


if __name__ == "__main__":
    unittest.main()
