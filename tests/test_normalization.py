"""Min-max scaling and conversion of normalized thetas to raw mileage units."""

import unittest

import support
from src.linear_regression import denormalize, estimate_price, normalize, thetas_to_original

_ = support

KM_MIN = 22899.0
KM_MAX = 240000.0
KM_SPAN = KM_MAX - KM_MIN


class NormalizationTests(unittest.TestCase):
    def test_dataset_bounds_map_to_zero_and_one(self) -> None:
        self.assertEqual(normalize(KM_MIN, KM_MIN, KM_MAX), 0.0)
        self.assertEqual(normalize(KM_MAX, KM_MIN, KM_MAX), 1.0)

    def test_learning_guide_example(self) -> None:
        self.assertAlmostEqual(normalize(84000.0, KM_MIN, KM_MAX), (84000.0 - KM_MIN) / KM_SPAN)

    def test_round_trip(self) -> None:
        original = 84000.0
        restored = denormalize(normalize(original, KM_MIN, KM_MAX), KM_MIN, KM_MAX)
        self.assertAlmostEqual(restored, original)

    def test_constant_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize(100.0, 50.0, 50.0)
        with self.assertRaises(ValueError):
            denormalize(0.5, 50.0, 50.0)
        with self.assertRaises(ValueError):
            thetas_to_original(1.0, 2.0, 50.0, 50.0)


class ThetaConversionTests(unittest.TestCase):
    def test_converted_thetas_match_normalized_predictions(self) -> None:
        theta0_n = 10.0
        theta1_n = 20.0
        x_min = 100.0
        x_max = 200.0
        mileage = 150.0

        theta0, theta1 = thetas_to_original(theta0_n, theta1_n, x_min, x_max)
        self.assertEqual((theta0, theta1), (-10.0, 0.2))

        from_raw = estimate_price(mileage, theta0, theta1)
        from_norm = estimate_price(normalize(mileage, x_min, x_max), theta0_n, theta1_n)
        self.assertEqual(from_raw, from_norm)


if __name__ == "__main__":
    unittest.main()
