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
    KPlayerEWLGame,
    all_k_player_classical_profiles,
    bits_to_index,
    index_to_bits,
    k_player_classical_profile,
    k_player_ewl_pd_payoff,
    k_player_resident_mutant_profile,
)


class TestKPlayerEWLGame(unittest.TestCase):
    def assertProbabilityDistribution(self, result):
        self.assertAlmostEqual(result.probability_sum, 1.0, places=12)
        for probability in result.probabilities.values():
            self.assertGreaterEqual(probability, -1e-12)
            self.assertLessEqual(probability, 1.0 + 1e-12)

    def test_index_helpers_round_trip(self):
        for k in (2, 3, 4, 5):
            for index in range(1 << k):
                self.assertEqual(bits_to_index(index_to_bits(index, k)), index)

    def test_supports_research_k_values(self):
        for k in (2, 3, 4, 5):
            game = KPlayerEWLGame(k=k, gamma=0.4)
            result = game.run([EWL_C] * k)
            self.assertProbabilityDistribution(result)
            self.assertEqual(len(result.expected_payoffs), k)
            self.assertEqual(len(result.amplitudes), 1 << k)

    def test_rejects_out_of_scope_k(self):
        with self.assertRaises(ValueError):
            KPlayerEWLGame(k=6)

    def test_k_equals_two_matches_two_player_ewl_engine(self):
        strategies = (EWLStrategy(theta=0.7, phi=0.2), EWLStrategy(theta=1.1, phi=0.4))
        gamma = math.pi / 3.0
        two_player = EWLTwoPlayerGame(gamma=gamma).run(*strategies)
        k_player = KPlayerEWLGame(k=2, gamma=gamma).run(strategies)

        self.assertProbabilityDistribution(k_player)
        self.assertEqual(set(k_player.probabilities), set(two_player.probabilities))
        for outcome in two_player.probabilities:
            self.assertAlmostEqual(
                k_player.probabilities[outcome],
                two_player.probabilities[outcome],
                places=12,
            )
        for actual, expected in zip(k_player.expected_payoffs, two_player.expected_payoffs):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_gamma_zero_classical_profiles_are_deterministic(self):
        for k in (2, 3, 4, 5):
            game = KPlayerEWLGame(k=k, gamma=0.0)
            payoff_fn = k_player_ewl_pd_payoff(k)
            for profile in all_k_player_classical_profiles(k):
                expected_outcome = tuple(0 if strategy is EWL_C else 1 for strategy in profile)
                result = game.run(profile)
                self.assertProbabilityDistribution(result)
                self.assertAlmostEqual(result.probabilities[expected_outcome], 1.0)
                self.assertEqual(result.expected_payoffs, payoff_fn(expected_outcome))

    def test_classical_profiles_recover_classical_outcomes_under_entanglement(self):
        for k in (2, 3, 4, 5):
            game = KPlayerEWLGame(k=k, gamma=math.pi / 2.0)
            payoff_fn = k_player_ewl_pd_payoff(k)
            for profile in all_k_player_classical_profiles(k):
                expected_outcome = tuple(0 if strategy is EWL_C else 1 for strategy in profile)
                result = game.run(profile)
                self.assertProbabilityDistribution(result)
                self.assertAlmostEqual(result.probabilities[expected_outcome], 1.0)
                self.assertEqual(result.expected_payoffs, payoff_fn(expected_outcome))

    def test_arbitrary_quantum_profile_returns_one_payoff_per_player(self):
        strategies = (
            EWLStrategy(theta=0.2, phi=0.1),
            EWLStrategy(theta=0.8, phi=0.3),
            EWL_Q,
            EWLStrategy(theta=1.1, phi=0.4),
            EWL_D,
        )
        game = KPlayerEWLGame(k=5, gamma=math.pi / 4.0)
        result = game.run(strategies)

        self.assertProbabilityDistribution(result)
        self.assertEqual(len(result.expected_payoffs), 5)
        for payoff in result.expected_payoffs:
            self.assertIsInstance(payoff, float)

    def test_symmetric_quantum_profile_has_symmetric_payoffs(self):
        game = KPlayerEWLGame(k=5, gamma=math.pi / 2.0)
        result = game.run([EWL_Q] * 5)

        self.assertProbabilityDistribution(result)
        first = result.expected_payoffs[0]
        for payoff in result.expected_payoffs:
            self.assertAlmostEqual(payoff, first, places=12)

    def test_resident_mutant_profile_helper_supports_arbitrary_mutant_indices(self):
        profile = k_player_resident_mutant_profile(
            k=5,
            resident=EWL_C,
            mutant=EWL_D,
            mutant_indices=(1, 3),
        )

        self.assertEqual(profile, (EWL_C, EWL_D, EWL_C, EWL_D, EWL_C))

    def test_resident_mutant_payoffs_average_groups(self):
        game = KPlayerEWLGame(k=5, gamma=0.0)
        mutant_payoff, resident_payoff, result = game.resident_mutant_payoffs(
            resident=EWL_C,
            mutant=EWL_D,
            mutant_indices=(1, 3),
        )

        self.assertProbabilityDistribution(result)
        self.assertEqual(result.expected_payoffs, (6.0, 16.0, 6.0, 16.0, 6.0))
        self.assertEqual(mutant_payoff, 16.0)
        self.assertEqual(resident_payoff, 6.0)

    def test_classical_profile_helper(self):
        self.assertEqual(k_player_classical_profile(("C", "D", "C")), (EWL_C, EWL_D, EWL_C))
        with self.assertRaises(ValueError):
            k_player_classical_profile(("C", "X"))

    def test_invalid_payoff_function_output_is_rejected(self):
        def bad_payoff(_outcome):
            return (1.0,)

        game = KPlayerEWLGame(k=3, payoff_fn=bad_payoff)
        with self.assertRaises(ValueError):
            game.run([EWL_C, EWL_C, EWL_C])


if __name__ == "__main__":
    unittest.main()
