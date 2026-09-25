# ft_linear_regression

Predict a car price from mileage with a single-feature linear regression trained by batch gradient descent.

## Constraints

- Hypothesis: `estimated_price(mileage) = theta0 + theta1 * mileage`
- Before training, `theta0 = 0` and `theta1 = 0`
- Both thetas are updated simultaneously from the subject formulas
- No library that performs the regression for us (`numpy.polyfit`, scikit-learn, …)
- Python 3.12+ standard library only for the mandatory programs

## Usage

```bash
make test
make train
make predict
```

Or without Make:

```bash
python3 -m unittest discover -s tests -b
python3 train.py
python3 predict.py
```

`train.py` reads `data/data.csv` (`km,price`), learns thetas, and writes `model.json`.

`predict.py` asks for a mileage and prints `theta0 + theta1 * mileage`. If `model.json` is missing, both thetas are `0` and the estimate is `0`.

Optional flags:

```bash
python3 train.py --data data/data.csv --model model.json --learning-rate 0.1 --iterations 1000
python3 predict.py --model model.json
```

## Tests

```bash
python3 -m unittest discover -s tests -b
```

## Bonus

```bash
make install    # once, for matplotlib
make bonus      # trains if needed, then writes charts into bonus_results/
```

Open the files in `bonus_results/`:

- `regression.png` — cars + fitted line
- `precision_metrics.png` — MAE, RMSE, and R² bars
- `residuals.png` — prediction error vs mileage
- `precision.txt` — the same numbers as text

`train.py` and `predict.py` do not need matplotlib.

## Generated files

`model.json` and everything in `bonus_results/` are generated. Do not commit them. `make clean` removes them.

## Limits

Mileage is only one factor. Predictions outside the dataset range (22,899–240,000 km) are extrapolation. The printed price is the raw hypothesis and can be negative for extreme mileage.
