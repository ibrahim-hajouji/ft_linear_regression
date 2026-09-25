"""Hand-checked predictions for the required hypothesis."""

import unittest

import support
from src.linear_regression import estimate_price

_ = support


class EstimatePriceTests(unittest.TestCase):
    def test_untrained_model_always_predicts_zero(self) -> None:
        self.assertEqual(estimate_price(84000, 0.0, 0.0), 0.0)
        self.assertEqual(estimate_price(0.0, 0.0, 0.0), 0.0)

    def test_worked_example_from_learning_guide(self) -> None:
        self.assertEqual(estimate_price(100000, 9000.0, -0.020), 7000.0)

    def test_one_row_training_example_starts_at_zero(self) -> None:
        self.assertEqual(estimate_price(10, 0.0, 0.0), 0.0)

    def test_after_one_manual_gradient_step(self) -> None:
        self.assertEqual(estimate_price(10, 1.0, 10.0), 101.0)


if __name__ == "__main__":
    unittest.main()
