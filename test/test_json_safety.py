import math
import unittest

from json_safety import sanitize_json_value, validate_json_payload


class TestJsonSafety(unittest.TestCase):
    def test_sanitizes_non_finite_numbers_recursively(self):
        payload = {
            "finite": 1.25,
            "nan": math.nan,
            "positive_infinity": math.inf,
            "negative_infinity": -math.inf,
            "nested": [math.nan, {"value": math.inf}],
        }

        sanitized = sanitize_json_value(payload)

        self.assertEqual(sanitized["finite"], 1.25)
        self.assertIsNone(sanitized["nan"])
        self.assertIsNone(sanitized["positive_infinity"])
        self.assertIsNone(sanitized["negative_infinity"])
        self.assertEqual(sanitized["nested"], [None, {"value": None}])
        validate_json_payload(sanitized)

    def test_rejects_values_that_are_not_json_serializable(self):
        with self.assertRaises(ValueError):
            validate_json_payload({"unsupported": object()})


if __name__ == "__main__":
    unittest.main()
