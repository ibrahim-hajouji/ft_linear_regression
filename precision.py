#!/usr/bin/env python3
"""Print how close the saved model is to the training prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import DatasetError, load_dataset
from src.linear_regression import (
    cost,
    estimate_price,
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)
from src.model import ModelError, load_model

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data.csv"
DEFAULT_MODEL = ROOT / "model.json"
DEFAULT_OUTPUT_DIR = ROOT / "bonus_results"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a saved model on a km,price CSV.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def precision_report(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> dict[str, float]:
    return {
        "cost": cost(mileages, prices, theta0, theta1),
        "mae": mean_absolute_error(mileages, prices, theta0, theta1),
        "mse": mean_squared_error(mileages, prices, theta0, theta1),
        "rmse": root_mean_squared_error(mileages, prices, theta0, theta1),
        "r2": r_squared(mileages, prices, theta0, theta1),
    }


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        mileages, prices = load_dataset(args.data)
        model = load_model(args.model)
    except (DatasetError, ModelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report = precision_report(mileages, prices, model.theta0, model.theta1)
    print(f"theta0 = {model.theta0}")
    print(f"theta1 = {model.theta1}")
    print(f"cost   = {report['cost']}")
    print(f"MAE    = {report['mae']}")
    print(f"MSE    = {report['mse']}")
    print(f"RMSE   = {report['rmse']}")
    print(f"R2     = {report['r2']}")

    try:
        saved = write_precision_charts(
            report, mileages, prices, model.theta0, model.theta1, args.output_dir
        )
    except ImportError:
        print(
            "error: matplotlib is required for precision charts (pip install -r requirements.txt)",
            file=sys.stderr,
        )
        return 1
    for path in saved:
        print(f"saved {path}")
    return 0


def write_precision_charts(
    report: dict[str, float],
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = [estimate_price(mileage, theta0, theta1) for mileage in mileages]
    residuals = [prediction - price for prediction, price in zip(predictions, prices)]

    metrics_path = output_dir / "precision_metrics.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["MAE", "RMSE"], [report["mae"], report["rmse"]])
    axes[0].set_title("Error in price units")
    axes[0].set_ylabel("Price")
    axes[1].bar(["R²"], [report["r2"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Coefficient of determination")
    figure.tight_layout()
    figure.savefig(metrics_path)
    plt.close(figure)

    residuals_path = output_dir / "residuals.png"
    figure, axes = plt.subplots()
    axes.scatter(mileages, residuals)
    axes.axhline(0, color="red")
    axes.set_title("Residuals (prediction - actual price)")
    axes.set_xlabel("Mileage (km)")
    axes.set_ylabel("Residual")
    figure.tight_layout()
    figure.savefig(residuals_path)
    plt.close(figure)

    report_path = output_dir / "precision.txt"
    report_path.write_text(
        "\n".join(
            [
                f"theta0 = {theta0}",
                f"theta1 = {theta1}",
                f"cost   = {report['cost']}",
                f"MAE    = {report['mae']}",
                f"MSE    = {report['mse']}",
                f"RMSE   = {report['rmse']}",
                f"R2     = {report['r2']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [metrics_path, residuals_path, report_path]


if __name__ == "__main__":
    raise SystemExit(run())
