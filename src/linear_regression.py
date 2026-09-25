"""Core hypothesis, scaling, and metrics for single-feature linear regression."""

from math import sqrt


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Return h_theta(mileage) = theta0 + theta1 * mileage."""
    return theta0 + theta1 * mileage


def normalize(value: float, x_min: float, x_max: float) -> float:
    """Scale a feature into [0, 1] using training-set min and max."""
    span = x_max - x_min
    if span == 0:
        raise ValueError("cannot normalize a constant feature: x_max equals x_min")
    return (value - x_min) / span


def denormalize(value: float, x_min: float, x_max: float) -> float:
    """Invert min-max normalization back to the original feature scale."""
    span = x_max - x_min
    if span == 0:
        raise ValueError("cannot denormalize a constant feature: x_max equals x_min")
    return value * span + x_min


def thetas_to_original(
    theta0_normalized: float,
    theta1_normalized: float,
    x_min: float,
    x_max: float,
) -> tuple[float, float]:
    """Convert thetas trained on normalized x into thetas for raw mileage.

    If training used x_n = (x - x_min) / (x_max - x_min), then:

        h = theta0_n + theta1_n * x_n
          = (theta0_n - theta1_n * x_min / span) + (theta1_n / span) * x
    """
    span = x_max - x_min
    if span == 0:
        raise ValueError("cannot convert thetas for a constant feature")
    theta1 = theta1_normalized / span
    theta0 = theta0_normalized - theta1_normalized * x_min / span
    return theta0, theta1


def _errors(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> list[float]:
    if len(mileages) != len(prices):
        raise ValueError("mileages and prices must have the same length")
    if not mileages:
        raise ValueError("cannot score an empty dataset")
    return [
        estimate_price(mileage, theta0, theta1) - price
        for mileage, price in zip(mileages, prices)
    ]


def cost(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    """Return J(theta) = (1 / (2m)) * sum(error^2)."""
    errors = _errors(mileages, prices, theta0, theta1)
    squared = sum(error * error for error in errors)
    return squared / (2 * len(errors))


def mean_squared_error(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    errors = _errors(mileages, prices, theta0, theta1)
    return sum(error * error for error in errors) / len(errors)


def root_mean_squared_error(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    return sqrt(mean_squared_error(mileages, prices, theta0, theta1))


def mean_absolute_error(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    errors = _errors(mileages, prices, theta0, theta1)
    return sum(abs(error) for error in errors) / len(errors)


def r_squared(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    """Return 1 - SSE / SST. SST is the variance of the actual prices."""
    errors = _errors(mileages, prices, theta0, theta1)
    sse = sum(error * error for error in errors)
    mean_price = sum(prices) / len(prices)
    sst = sum((price - mean_price) ** 2 for price in prices)
    if sst == 0:
        if sse == 0:
            return 1.0
        raise ValueError("R-squared is undefined when all actual prices are identical")
    return 1.0 - sse / sst


def gradient_step(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    learning_rate: float,
) -> tuple[float, float]:
    """One simultaneous batch update of both parameters.

    Matches the subject:

        tmp_theta0 = alpha * (1/m) * sum(error)
        tmp_theta1 = alpha * (1/m) * sum(error * x)
        theta0, theta1 = theta0 - tmp_theta0, theta1 - tmp_theta1
    """
    errors = _errors(mileages, prices, theta0, theta1)
    sample_count = len(errors)
    tmp_theta0 = learning_rate * sum(errors) / sample_count
    tmp_theta1 = (
        learning_rate
        * sum(error * mileage for error, mileage in zip(errors, mileages))
        / sample_count
    )
    return theta0 - tmp_theta0, theta1 - tmp_theta1


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> tuple[float, float]:
    """Learn theta0 and theta1 for raw mileage, starting from (0, 0)."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    x_min = min(mileages)
    x_max = max(mileages)
    normalized = [normalize(mileage, x_min, x_max) for mileage in mileages]

    theta0 = 0.0
    theta1 = 0.0
    for _ in range(iterations):
        theta0, theta1 = gradient_step(
            normalized, prices, theta0, theta1, learning_rate
        )
    return thetas_to_original(theta0, theta1, x_min, x_max)
