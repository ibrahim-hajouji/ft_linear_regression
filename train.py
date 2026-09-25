#!/usr/bin/env python3
"""Train theta0 and theta1 from a km,price CSV and save model.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import DatasetError, load_dataset
from src.linear_regression import (
    cost,
    mean_absolute_error,
    r_squared,
    train,
)
from src.model import Model, save_model

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data.csv"
DEFAULT_MODEL = ROOT / "model.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-feature linear regression.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="km,price CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="output JSON path")
    parser.add_argument("--learning-rate", type=float, default=0.1, dest="learning_rate")
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        mileages, prices = load_dataset(args.data)
        theta0, theta1 = train(
            mileages,
            prices,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
        )
    except (DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    model = Model(
        theta0=theta0,
        theta1=theta1,
        x_min=min(mileages),
        x_max=max(mileages),
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )
    saved = save_model(model, args.model)

    print(f"saved {saved}")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"cost   = {cost(mileages, prices, theta0, theta1)}")
    print(f"MAE    = {mean_absolute_error(mileages, prices, theta0, theta1)}")
    print(f"R2     = {r_squared(mileages, prices, theta0, theta1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
