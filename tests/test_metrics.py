import csv
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from sweettv.metrics import ap_at_k, audit_exported_ap, evaluate


class AveragePrecisionTests(unittest.TestCase):
    def test_assignment_hand_computed_examples(self):
        self.assertAlmostEqual(ap_at_k({"A", "B", "C"}, ["A", "B", "D", "C", "E"]), 11 / 12)
        self.assertAlmostEqual(ap_at_k({"A", "B", "C"}, ["D", "A", "E", "B", "C"]), 8 / 15)
        result = evaluate(
            {"001": [1, 2, 4, 3, 5], "002": [4, 1, 5, 2, 3]},
            {"001": {1, 2, 3}, "002": {1, 2, 3}},
            ["001", "002"],
        )
        self.assertAlmostEqual(result["map_at_5"], 0.725)
        self.assertEqual(list(result["per_user_ap"]), ["001", "002"])

    def test_predictions_truncate_at_five(self):
        self.assertEqual(ap_at_k({6}, [1, 2, 3, 4, 5, 6]), 0)
        self.assertEqual(ap_at_k({1}, [1, 2], k=1), 1)

    def test_duplicate_prediction_keeps_rank_but_gets_no_repeat_credit(self):
        self.assertAlmostEqual(ap_at_k({1, 2}, [1, 1, 2, 2, 3]), 5 / 6)
        self.assertEqual(ap_at_k({1}, [1, 1, 1, 1, 1]), 1)
        self.assertAlmostEqual(ap_at_k({1, 2}, [3, 3, 1, 2, 4]), (1 / 3 + 2 / 4) / 2)

    def test_labels_are_unique(self):
        self.assertEqual(ap_at_k([1, 1, 1], [1, 2, 3, 4, 5]), 1)

    def test_empty_labels_no_hits_or_empty_predictions(self):
        self.assertEqual(ap_at_k(set(), [1, 2, 3]), 0)
        self.assertEqual(ap_at_k({1}, [2, 3, 4]), 0)
        self.assertEqual(ap_at_k({1}, []), 0)
        self.assertEqual(ap_at_k({1}, [1], k=0), 0)

    def test_more_than_five_labels_uses_five_denominator(self):
        self.assertEqual(ap_at_k(set(range(1, 9)), [1, 2, 3, 4, 5]), 1)

    def test_fixed_cohort_includes_zero_history_and_explicit_empty_targets(self):
        result = evaluate(
            {"warm": [1], "cold": [2], "empty": [1], "outside": [999]},
            {"warm": {1}, "cold": {2}, "empty": set(), "outside": {999}},
            ["cold", "empty", "warm"],
        )
        self.assertEqual(result["cohort_size"], 3)
        self.assertEqual(result["empty_target_users"], 1)
        self.assertAlmostEqual(result["map_at_5"], 2 / 3)
        self.assertEqual(list(result["per_user_ap"]), ["cold", "empty", "warm"])

    def test_invalid_cohort_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({}, {}, [])
        with self.assertRaises(ValueError):
            evaluate({"1": []}, {"1": set()}, ["1", "1"])
        with self.assertRaises(ValueError):
            evaluate({}, {"1": set()}, ["1"])
        with self.assertRaises(ValueError):
            evaluate({"1": []}, {}, ["1"])


class ExportedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "per_user.csv"
        self.rows = [
            {"method": method, "user_id": user, "relevant_ids": "[1,2,3]", "predictions": json.dumps(predictions), "ap_at_5": ap}
            for method in ["popularity", "content"]
            for user, predictions, ap in [("001", [1, 2, 4, 3, 5], 11 / 12), ("002", [4, 1, 5, 2, 3], 8 / 15)]
        ]

    def write(self):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "user_id", "relevant_ids", "predictions", "ap_at_5"])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_exact_fraction_and_saved_summary(self):
        self.write()
        summary = self.directory / "summary.json"
        summary.write_text(json.dumps({"methods": {name: {"map_at_5": 0.725} for name in ["popularity", "content"]}}))
        result = audit_exported_ap(self.path, summary)
        self.assertTrue(result["passed"])
        for values in result["methods"].values():
            self.assertEqual(Fraction(values["exact_numerator"], values["exact_denominator"]), Fraction(29, 40))
            self.assertEqual(values["cohort_size"], 2)

    def test_saved_user_ap_mismatch_fails(self):
        self.rows[0]["ap_at_5"] = 0.9
        self.write()
        with self.assertRaisesRegex(ValueError, "AP audit mismatch"):
            audit_exported_ap(self.path)

    def test_saved_macro_mismatch_fails(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "Macro audit mismatch"):
            audit_exported_ap(self.path, expected_map={"popularity": 0.7, "content": 0.725})

    def test_cohort_change_or_duplicate_fails(self):
        self.rows[-1]["user_id"] = "003"
        self.write()
        with self.assertRaisesRegex(ValueError, "same ordered"):
            audit_exported_ap(self.path)
        self.rows[-1]["user_id"] = "001"
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_exported_ap(self.path)

    def test_audit_handles_duplicates_empty_labels_and_truncation(self):
        self.rows = [
            {"method": "model", "user_id": "a", "relevant_ids": "[1,1,2]", "predictions": "[1,1,2,2,3,4]", "ap_at_5": 5 / 6},
            {"method": "model", "user_id": "b", "relevant_ids": "[]", "predictions": "[1,2,3,4,5]", "ap_at_5": 0},
            {"method": "model", "user_id": "c", "relevant_ids": "[6]", "predictions": "[1,2,3,4,5,6]", "ap_at_5": 0},
        ]
        self.write()
        values = audit_exported_ap(self.path, expected_map={"model": 5 / 18})["methods"]["model"]
        self.assertEqual(values["exact_numerator"], 5)
        self.assertEqual(values["exact_denominator"], 18)
        self.assertEqual(values["empty_target_users"], 1)


if __name__ == "__main__":
    unittest.main()
