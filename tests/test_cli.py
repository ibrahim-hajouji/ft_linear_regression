"""Trainer and predictor programs."""

import tempfile
import unittest
from pathlib import Path

import support
from predict import parse_mileage, run as run_predict
from src.model import load_model
from train import run as run_train

_ = support

SUBJECT_CSV = Path(__file__).resolve().parents[1] / "data" / "data.csv"


class ParseMileageTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(parse_mileage(" 84000 "), 84000.0)

    def test_rejects_empty_non_numeric_inf_and_negative(self) -> None:
        for raw in ("", "abc", "inf", "-1"):
            with self.assertRaises(ValueError):
                parse_mileage(raw)


class TrainPredictTests(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handle.close()
        self.model_path = Path(handle.name)
        self.model_path.unlink()

    def tearDown(self) -> None:
        self.model_path.unlink(missing_ok=True)

    def test_untrained_predict_is_zero(self) -> None:
        code = run_predict(
            ["--model", str(self.model_path)],
            input_fn=lambda _: "84000",
        )
        self.assertEqual(code, 0)

    def test_train_then_predict_uses_saved_thetas(self) -> None:
        data = Path(tempfile.mkstemp(suffix=".csv")[1])
        data.write_text("km,price\n0,10\n1,12\n", encoding="utf-8")
        try:
            code = run_train(
                [
                    "--data",
                    str(data),
                    "--model",
                    str(self.model_path),
                    "--learning-rate",
                    "0.1",
                    "--iterations",
                    "5000",
                ]
            )
            self.assertEqual(code, 0)
            model = load_model(self.model_path)
            self.assertAlmostEqual(model.theta0, 10.0, places=2)
            self.assertAlmostEqual(model.theta1, 2.0, places=2)
            self.assertAlmostEqual(model.estimate(1.0), 12.0, places=2)
        finally:
            data.unlink(missing_ok=True)

    def test_train_subject_dataset(self) -> None:
        code = run_train(
            [
                "--data",
                str(SUBJECT_CSV),
                "--model",
                str(self.model_path),
                "--iterations",
                "1000",
            ]
        )
        self.assertEqual(code, 0)
        model = load_model(self.model_path)
        self.assertLess(model.theta1, 0.0)

    def test_invalid_mileage_exits_one(self) -> None:
        code = run_predict(
            ["--model", str(self.model_path)],
            input_fn=lambda _: "nope",
        )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
