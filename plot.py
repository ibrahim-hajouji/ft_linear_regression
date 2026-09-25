#!/usr/bin/env python3
"""Bonus: scatter the dataset and draw the fitted regression line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import DatasetError, load_dataset
from src.model import ModelError, load_model

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data.csv"
DEFAULT_MODEL = ROOT / "model.json"
DEFAULT_OUTPUT = ROOT / "bonus_results" / "regression.png"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mileage vs price and the fitted line.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def line_endpoints(
    theta0: float,
    theta1: float,
    x_min: float,
    x_max: float,
) -> tuple[list[float], list[float]]:
    """Two points are enough to draw a straight line."""
    return [x_min, x_max], [theta0 + theta1 * x_min, theta0 + theta1 * x_max]


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        mileages, prices = load_dataset(args.data)
        model = load_model(args.model)
    except (DatasetError, ModelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("error: matplotlib is required for plot.py (pip install -r requirements.txt)", file=sys.stderr)
        return 1

    xs_line, ys_line = line_endpoints(model.theta0, model.theta1, min(mileages), max(mileages))
    figure, axes = plt.subplots()
    axes.scatter(mileages, prices, label="training cars")
    axes.plot(xs_line, ys_line, color="red", label="fitted line")
    axes.set_title("Car price vs mileage")
    axes.set_xlabel("Mileage (km)")
    axes.set_ylabel("Price")
    axes.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
