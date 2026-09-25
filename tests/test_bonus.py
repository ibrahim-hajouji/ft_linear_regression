"""Bonus precision report and plot helpers."""

import tempfile
import unittest
from pathlib import Path

import support
from plot import line_endpoints, run as run_plot
from precision import precision_report, run as run_precision
from train import run as run_train

_ = support


class PrecisionTests(unittest.TestCase):
    def test_perfect_line(self) -> None:
        report = precision_report([0.0, 1.0], [10.0, 12.0], 10.0, 2.0)
        self.assertEqual(report["mae"], 0.0)
        self.assertEqual(report["mse"], 0.0)
        self.assertEqual(report["rmse"], 0.0)
        self.assertEqual(report["cost"], 0.0)
        self.assertEqual(report["r2"], 1.0)

    def test_cli_needs_a_saved_model(self) -> None:
        code = run_precision(["--model", "/tmp/does-not-exist-ft-lr.json"])
        self.assertEqual(code, 1)


class PlotHelperTests(unittest.TestCase):
    def test_line_endpoints(self) -> None:
        xs, ys = line_endpoints(10.0, 2.0, 0.0, 1.0)
        self.assertEqual(xs, [0.0, 1.0])
        self.assertEqual(ys, [10.0, 12.0])


class PlotProgramTests(unittest.TestCase):
    def test_precision_writes_charts(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is optional")

        out_dir = Path(tempfile.mkdtemp())
        data = Path(tempfile.mkstemp(suffix=".csv")[1])
        model_path = Path(tempfile.mkstemp(suffix=".json")[1])
        data.write_text("km,price\n0,10\n1,12\n", encoding="utf-8")
        try:
            self.assertEqual(
                run_train(
                    ["--data", str(data), "--model", str(model_path), "--iterations", "2000"]
                ),
                0,
            )
            code = run_precision(
                ["--data", str(data), "--model", str(model_path), "--output-dir", str(out_dir)]
            )
            self.assertEqual(code, 0)
            self.assertGreater((out_dir / "precision_metrics.png").stat().st_size, 0)
            self.assertGreater((out_dir / "residuals.png").stat().st_size, 0)
            self.assertTrue((out_dir / "precision.txt").is_file())
        finally:
            data.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            for child in out_dir.iterdir():
                child.unlink()
            out_dir.rmdir()

    def test_writes_png_when_matplotlib_is_installed(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is optional")

        data = Path(tempfile.mkstemp(suffix=".csv")[1])
        model_path = Path(tempfile.mkstemp(suffix=".json")[1])
        png = Path(tempfile.mkstemp(suffix=".png")[1])
        data.write_text("km,price\n0,10\n1,12\n", encoding="utf-8")
        try:
            self.assertEqual(
                run_train(
                    [
                        "--data",
                        str(data),
                        "--model",
                        str(model_path),
                        "--iterations",
                        "2000",
                    ]
                ),
                0,
            )
            code = run_plot(
                ["--data", str(data), "--model", str(model_path), "--output", str(png)]
            )
            self.assertEqual(code, 0)
            self.assertGreater(png.stat().st_size, 0)
        finally:
            data.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            png.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
