"""Hand-checked cost and evaluation metrics."""

import unittest

import support
from src.linear_regression import (
    cost,
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)

_ = support


class MetricTests(unittest.TestCase):
    def test_one_row_untrained_cost_from_learning_guide(self) -> None:
        # prediction 0, actual 100, error -100, J = 10000 / 2
        self.assertEqual(cost([10.0], [100.0], 0.0, 0.0), 5000.0)
        self.assertEqual(mean_squared_error([10.0], [100.0], 0.0, 0.0), 10000.0)

    def test_two_row_untrained_metrics(self) -> None:
        mileages = [0.0, 1.0]
        prices = [10.0, 12.0]
        self.assertEqual(mean_absolute_error(mileages, prices, 0.0, 0.0), 11.0)
        self.assertEqual(mean_squared_error(mileages, prices, 0.0, 0.0), 122.0)
        self.assertEqual(cost(mileages, prices, 0.0, 0.0), 61.0)
        self.assertAlmostEqual(root_mean_squared_error(mileages, prices, 0.0, 0.0), 122.0**0.5)
        self.assertEqual(r_squared(mileages, prices, 0.0, 0.0), -121.0)

    def test_perfect_line_has_zero_error_and_r_squared_one(self) -> None:
        mileages = [0.0, 1.0]
        prices = [10.0, 12.0]
        self.assertEqual(mean_absolute_error(mileages, prices, 10.0, 2.0), 0.0)
        self.assertEqual(mean_squared_error(mileages, prices, 10.0, 2.0), 0.0)
        self.assertEqual(cost(mileages, prices, 10.0, 2.0), 0.0)
        self.assertEqual(r_squared(mileages, prices, 10.0, 2.0), 1.0)

    def test_rejects_empty_or_mismatched_lists(self) -> None:
        with self.assertRaises(ValueError):
            cost([], [], 0.0, 0.0)
        with self.assertRaises(ValueError):
            cost([1.0], [1.0, 2.0], 0.0, 0.0)

    def test_r_squared_undefined_when_prices_are_constant_and_wrong(self) -> None:
        with self.assertRaises(ValueError):
            r_squared([0.0, 1.0], [5.0, 5.0], 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
