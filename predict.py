#!/usr/bin/env python3
"""Estimate a car price from mileage using the saved model, or zeros if none."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from src.model import Model, ModelError, load_model, untrained_model

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a car price from mileage.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained JSON path")
    return parser.parse_args(argv)


def parse_mileage(raw: str) -> float:
    text = raw.strip()
    if text == "":
        raise ValueError("mileage cannot be empty")
    try:
        mileage = float(text)
    except ValueError as error:
        raise ValueError(f"mileage is not a number: {text!r}") from error
    if not math.isfinite(mileage):
        raise ValueError("mileage must be a finite number")
    if mileage < 0:
        raise ValueError("mileage cannot be negative")
    return mileage


def load_predictor(path: Path) -> tuple[Model, bool]:
    """Return (model, trained). Missing file → untrained zeros. Corrupt file → error."""
    if not path.is_file():
        return untrained_model(), False
    return load_model(path), True


def run(
    argv: list[str] | None = None,
    input_fn=input,
) -> int:
    args = parse_args(argv)
    try:
        model, trained = load_predictor(args.model)
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not trained:
        print("no trained model found; using theta0=0, theta1=0", file=sys.stderr)

    try:
        mileage = parse_mileage(input_fn("mileage: "))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except EOFError:
        print("error: no mileage provided", file=sys.stderr)
        return 1

    print(f"estimated price: {model.estimate(mileage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
