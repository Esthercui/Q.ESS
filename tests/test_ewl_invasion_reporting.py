import csv
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from quantum_ess import (
    DETAILED_FIELDNAMES,
    EWL_Q,
    EWLStrategy,
    KPlayerEWLGame,
    analyze_resident_mutant,
    detailed_rows_from_grid,
    search_mutants_on_grid,
    write_detailed_csv,
    write_json,
)


class TestEWLInvasionReporting(unittest.TestCase):
    def test_required_detailed_fields_and_gzip_round_trip(self):
        grid_result = search_mutants_on_grid(
            k=2,
            gamma=math.pi / 2.0,
            resident=EWL_Q,
            mutant_strategies=(EWLStrategy(math.pi, 0.0),),
            epsilon_reference=0.01,
            stage="test",
        )
        rows = detailed_rows_from_grid(grid_result)
        self.assertEqual(set(rows[0]), set(DETAILED_FIELDNAMES))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv.gz"
            write_detailed_csv(path, rows)
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                restored = list(csv.DictReader(handle))

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0]["search_stage"], "test")
        self.assertEqual(restored[0]["k"], "2")

    def test_gzip_json_round_trip(self):
        value = {"baseline": "5827d9a", "values": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json.gz"
            write_json(path, value)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                restored = json.load(handle)
        self.assertEqual(restored, value)

    def test_uncertainty_field_declares_exact_statevector(self):
        analysis = analyze_resident_mutant(
            game=KPlayerEWLGame(k=2, gamma=math.pi / 2.0),
            resident=EWL_Q,
            mutant=EWLStrategy(theta=math.pi / 2.0, phi=0.0),
            epsilon_reference=0.01,
        )
        for result in analysis.position_results:
            self.assertIn(
                "exact_statevector",
                result.uncertainty_or_confidence_interval,
            )
            self.assertIn(
                "sampling_confidence_interval=not_applicable",
                result.uncertainty_or_confidence_interval,
            )


if __name__ == "__main__":
    unittest.main()
