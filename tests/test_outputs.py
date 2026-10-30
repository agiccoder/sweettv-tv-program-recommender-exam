import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from sweettv.outputs import validate_submission, write_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.template = self.directory / "template.csv"
        self.output = self.directory / "final.csv"
        self.users = ["00012345678901234567890", "9007199254740993"]
        self.header = ["user_id", "tv_show_id"]
        self.candidates = set(range(1, 9))
        self.recommendations = {self.users[0]: [1, 2, 3, 4, 5], self.users[1]: [8, 7, 6, 5, 4]}
        self.save(self.template, [[user, "0 0 0 0 0"] for user in self.users])

    def save(self, path, rows, header=None, delimiter=","):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(self.header if header is None else header)
            writer.writerows(rows)

    def validate(self):
        return validate_submission(self.template, self.output, self.candidates, expected_users=2)

    def test_writer_preserves_raw_user_ids_order_headers_and_template(self):
        original = self.template.read_bytes()
        result = write_submission(self.template, self.output, self.recommendations, self.candidates, expected_users=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(result["prediction_column"], "tv_show_id")
        self.assertEqual(self.template.read_bytes(), original)
        with self.output.open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], self.header)
        self.assertEqual([row[0] for row in rows[1:]], self.users)
        self.assertEqual(rows[1][1], "1 2 3 4 5")

    def test_prediction_column_can_precede_user_column(self):
        self.save(self.template, [["0 0 0 0 0", user] for user in self.users], header=["programmes", "user_id"], delimiter=";")
        result = write_submission(self.template, self.output, self.recommendations, self.candidates, expected_users=2)
        self.assertEqual(result["headers"], ["programmes", "user_id"])
        self.assertEqual(result["prediction_column"], "programmes")

    def test_bom_is_preserved(self):
        self.template.write_bytes(b"\xef\xbb\xbf" + self.template.read_bytes())
        write_submission(self.template, self.output, self.recommendations, self.candidates, expected_users=2)
        self.assertTrue(self.output.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_invalid_prediction_formats(self):
        for value in ["0 0 0 0 0", "1 2 3 4", "1 2 3 4 5 6", "1  2 3 4 5", " 1 2 3 4 5", "1 2 3 4 5 ", "1\t2 3 4 5", "1 2 3 4 4", "1 2 3 4 9", "1 2 3 4 5.0", "1 2 3 4 -5", "1 2 3 4 +5", "1 2 3 4 0", "", "NaN"]:
            with self.subTest(value=value):
                self.save(self.output, [[self.users[0], value], [self.users[1], "1 2 3 4 5"]])
                with self.assertRaises(ValueError):
                    self.validate()

    def test_user_id_order_changes_and_numeric_conversion_fail(self):
        for users in [self.users[::-1], [self.users[0].lstrip("0"), self.users[1]], [self.users[0], "9007199254740992"], [self.users[0], self.users[0]], [self.users[0], "null"]]:
            with self.subTest(users=users):
                self.save(self.output, [[user, "1 2 3 4 5"] for user in users])
                with self.assertRaises(ValueError):
                    self.validate()

    def test_header_and_row_count_failures(self):
        self.save(self.output, [[user, "1 2 3 4 5"] for user in self.users], header=["user_id", "prediction"])
        with self.assertRaisesRegex(ValueError, "headers"):
            self.validate()
        self.save(self.output, [[self.users[0], "1 2 3 4 5"]])
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            self.validate()

    def test_ambiguous_template_and_duplicate_users_fail(self):
        self.save(self.template, [[user, "0", "0"] for user in self.users], header=["user_id", "x", "y"])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            write_submission(self.template, self.output, self.recommendations, self.candidates, expected_users=2)
        self.save(self.template, [[self.users[0], "0"], [self.users[0], "0"]])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            write_submission(self.template, self.output, self.recommendations, self.candidates, expected_users=2)

    def test_writer_requires_complete_valid_recommendations(self):
        for recommendations in [{}, {**self.recommendations, self.users[0]: [1, 2, 3, 4, 4]}, {**self.recommendations, self.users[0]: [1, 2, 3, 4, 9]}, {**self.recommendations, self.users[0]: [1, 2, 3, 4, 5.0]}, {**self.recommendations, self.users[0]: [1, 2, 3, 4]}]:
            with self.subTest(recommendations=recommendations):
                with self.assertRaises(ValueError):
                    write_submission(self.template, self.output, recommendations, self.candidates, expected_users=2)

    def test_writer_cannot_overwrite_original_template(self):
        with self.assertRaisesRegex(ValueError, "must not be overwritten"):
            write_submission(self.template, self.template, self.recommendations, self.candidates, expected_users=2)


if __name__ == "__main__":
    unittest.main()
