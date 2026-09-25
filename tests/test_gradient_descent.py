"""Hand-checked gradient steps and a tiny training run."""

import unittest

import support
from src.linear_regression import cost, estimate_price, gradient_step, train

_ = support


class GradientStepTests(unittest.TestCase):
    def test_one_row_example_from_learning_guide(self) -> None:
        theta0, theta1 = gradient_step([10.0], [100.0], 0.0, 0.0, 0.01)
        self.assertEqual((theta0, theta1), (1.0, 10.0))

    def test_two_row_example_from_learning_guide(self) -> None:
        theta0, theta1 = gradient_step([0.0, 1.0], [10.0, 12.0], 0.0, 0.0, 0.1)
        self.assertAlmostEqual(theta0, 1.1)
        self.assertAlmostEqual(theta1, 0.6)

    def test_updates_are_simultaneous(self) -> None:
        # Using the new theta0 to recompute theta1 would leave theta1 at 1.
        theta0, theta1 = gradient_step([2.0], [0.0], 1.0, 1.0, 1.0)
        self.assertEqual((theta0, theta1), (-2.0, -5.0))


class TrainTests(unittest.TestCase):
    def test_recovers_a_perfect_two_point_line(self) -> None:
        theta0, theta1 = train([0.0, 1.0], [10.0, 12.0], learning_rate=0.1, iterations=5000)
        self.assertAlmostEqual(theta0, 10.0, places=2)
        self.assertAlmostEqual(theta1, 2.0, places=2)
        self.assertAlmostEqual(estimate_price(0.0, theta0, theta1), 10.0, places=2)
        self.assertAlmostEqual(estimate_price(1.0, theta0, theta1), 12.0, places=2)

    def test_lowers_cost_on_the_subject_shape(self) -> None:
        mileages = [0.0, 100.0, 200.0]
        prices = [20.0, 10.0, 0.0]
        before = cost(mileages, prices, 0.0, 0.0)
        theta0, theta1 = train(mileages, prices, learning_rate=0.5, iterations=200)
        after = cost(mileages, prices, theta0, theta1)
        self.assertLess(after, before)

    def test_rejects_bad_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            train([0.0, 1.0], [10.0, 12.0], learning_rate=0.0)
        with self.assertRaises(ValueError):
            train([0.0, 1.0], [10.0, 12.0], iterations=0)


if __name__ == "__main__":
    unittest.main()