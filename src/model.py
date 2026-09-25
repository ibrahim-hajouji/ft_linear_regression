"""Save and load the trained theta values as JSON."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path("model.json")


class ModelError(ValueError):
    """Raised when a model file is missing or invalid."""


@dataclass(frozen=True)
class Model:
    """Parameters needed to run the subject hypothesis on raw mileage."""

    theta0: float
    theta1: float
    x_min: float
    x_max: float
    learning_rate: float
    iterations: int
    version: int = MODEL_VERSION

    def estimate(self, mileage: float) -> float:
        return self.theta0 + self.theta1 * mileage


def untrained_model() -> Model:
    """Subject rule: before training, both thetas are 0."""
    return Model(
        theta0=0.0,
        theta1=0.0,
        x_min=0.0,
        x_max=1.0,
        learning_rate=0.0,
        iterations=0,
        version=MODEL_VERSION,
    )


def save_model(model: Model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Write the model as JSON. Overwrites an existing file."""
    model_path = Path(path)
    payload = asdict(model)
    model_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return model_path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Model:
    """Read a previously saved model. Does not invent default thetas."""
    model_path = Path(path)
    if not model_path.is_file():
        raise ModelError(f"model not found: {model_path}")
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelError(f"{model_path}: file is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ModelError(f"{model_path}: model file must be a JSON object")
    return _model_from_payload(payload, model_path)


def _model_from_payload(payload: dict, model_path: Path) -> Model:
    required = ("theta0", "theta1", "x_min", "x_max", "learning_rate", "iterations", "version")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ModelError(f"{model_path}: missing fields: {', '.join(missing)}")

    version = _as_int(payload["version"], "version", model_path)
    if version != MODEL_VERSION:
        raise ModelError(f"{model_path}: unsupported model version {version}")

    theta0 = _as_finite_float(payload["theta0"], "theta0", model_path)
    theta1 = _as_finite_float(payload["theta1"], "theta1", model_path)
    x_min = _as_finite_float(payload["x_min"], "x_min", model_path)
    x_max = _as_finite_float(payload["x_max"], "x_max", model_path)
    learning_rate = _as_finite_float(payload["learning_rate"], "learning_rate", model_path)
    iterations = _as_int(payload["iterations"], "iterations", model_path)

    if x_max == x_min:
        raise ModelError(f"{model_path}: x_max must differ from x_min")
    if iterations < 0:
        raise ModelError(f"{model_path}: iterations cannot be negative")
    if learning_rate < 0:
        raise ModelError(f"{model_path}: learning_rate cannot be negative")

    return Model(
        theta0=theta0,
        theta1=theta1,
        x_min=x_min,
        x_max=x_max,
        learning_rate=learning_rate,
        iterations=iterations,
        version=version,
    )


def _as_finite_float(value: object, field: str, model_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{model_path}: {field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ModelError(f"{model_path}: {field} must be a finite number")
    return number


def _as_int(value: object, field: str, model_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"{model_path}: {field} must be an integer")
    return value
