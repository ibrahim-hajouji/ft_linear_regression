"""Load and validate the km,price training CSV."""

from __future__ import annotations

import csv
import math
from pathlib import Path


class DatasetError(ValueError):
    """Raised when a dataset file is missing, empty, or malformed."""


def load_dataset(path: str | Path) -> tuple[list[float], list[float]]:
    """Return (mileages, prices) from a CSV with a km,price header."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DatasetError(f"dataset not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        _require_header(reader.fieldnames)
        mileages, prices = _read_rows(reader, csv_path)

    if len(mileages) < 2:
        raise DatasetError(f"{csv_path}: need at least 2 valid rows, got {len(mileages)}")
    if min(mileages) == max(mileages):
        raise DatasetError(f"{csv_path}: all mileage values are identical; a slope cannot be learned")
    return mileages, prices


def _require_header(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise DatasetError("dataset is empty or has no header")
    columns = {name.strip() for name in fieldnames if name is not None}
    if "km" not in columns or "price" not in columns:
        raise DatasetError("dataset header must contain the columns km and price")


def _read_rows(
    reader: csv.DictReader,
    csv_path: Path,
) -> tuple[list[float], list[float]]:
    mileages: list[float] = []
    prices: list[float] = []
    for line_number, row in enumerate(reader, start=2):
        raw_km = (row.get("km") or "").strip()
        raw_price = (row.get("price") or "").strip()
        if raw_km == "" and raw_price == "":
            continue
        mileage = _parse_number(raw_km, csv_path, line_number, "km")
        price = _parse_number(raw_price, csv_path, line_number, "price")
        if mileage < 0:
            raise DatasetError(f"{csv_path}:{line_number}: km cannot be negative")
        if price < 0:
            raise DatasetError(f"{csv_path}:{line_number}: price cannot be negative")
        mileages.append(mileage)
        prices.append(price)
    return mileages, prices


def _parse_number(raw: str, csv_path: Path, line_number: int, column: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise DatasetError(
            f"{csv_path}:{line_number}: {column} is not a number: {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise DatasetError(f"{csv_path}:{line_number}: {column} must be a finite number")
    return value
