"""Save / load of the JSON model artifact."""

import json
import tempfile
import unittest
from pathlib import Path

import support
from src.linear_regression import estimate_price
from src.model import Model, ModelError, load_model, save_model, untrained_model

_ = support


def _temp_path() -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    handle.close()
    return Path(handle.name)


class ModelPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _temp_path()

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_round_trip(self) -> None:
        original = Model(
            theta0=8000.0,
            theta1=-0.02,
            x_min=22899.0,
            x_max=240000.0,
            learning_rate=0.1,
            iterations=1000,
        )
        save_model(original, self.path)
        loaded = load_model(self.path)
        self.assertEqual(loaded, original)
        self.assertEqual(loaded.estimate(100000.0), estimate_price(100000.0, 8000.0, -0.02))

    def test_untrained_model_predicts_zero(self) -> None:
        model = untrained_model()
        self.assertEqual(model.estimate(84000.0), 0.0)

    def test_missing_file(self) -> None:
        with self.assertRaises(ModelError):
            load_model("/tmp/does-not-exist-ft-linear-regression.json")

    def test_invalid_json(self) -> None:
        self.path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(ModelError):
            load_model(self.path)

    def test_missing_field(self) -> None:
        self.path.write_text(json.dumps({"theta0": 1.0}), encoding="utf-8")
        with self.assertRaises(ModelError):
            load_model(self.path)

    def test_unsupported_version(self) -> None:
        payload = {
            "theta0": 1.0,
            "theta1": 2.0,
            "x_min": 0.0,
            "x_max": 1.0,
            "learning_rate": 0.1,
            "iterations": 10,
            "version": 99,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ModelError):
            load_model(self.path)


if __name__ == "__main__":
    unittest.main()
