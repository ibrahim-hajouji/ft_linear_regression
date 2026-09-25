# ft_linear_regression — Learning Guide

Read this file the night before the evaluation. It is the theory, the code map, and the defense answers in one place. `README.md` is only how to run the programs.

```text
historical car data -> training -> learned parameters -> saved model -> new mileage -> predicted price
```

**How to re-read**

1. Sections 1–12 — ideas (line, error, gradient, alpha, normalization).
2. Section 13 — what each file actually does.
3. Section 14 — say these answers out loud, without the file.
4. Section 15 — `make` targets.
5. Section 16 — how AI was used (the subject asks for honesty).

| Section | Topic |
| --- | --- |
| 1–4 | Contract, vocabulary, dataset, supervised regression |
| 5–6 | Hypothesis, bias/weight, error, cost |
| 7–8 | Gradient descent, alpha, neural-net analogy |
| 9–12 | Normalization, metrics, plot reading, validation |
| 13 | Implementation map (keep this aligned with the code) |
| 14 | Evaluator questions with answers |
| 15 | Makefile |
| 16 | AI-use disclosure |

## 1. Project contract

The subject requires a **simple linear regression with one feature**:

- input feature: a car's mileage in kilometres, written `x` or `km`;
- target: its price, written `y` or `price`;
- model: a straight line;
- learning method: gradient descent;
- programs: one trainer and one interactive predictor.

The required hypothesis is:

```text
estimated_price(mileage) = theta0 + theta1 * mileage
```

`theta0` and `theta1` begin at `0` before training. The trainer reads the dataset, learns better values, and saves them. The predictor later loads those values and uses the same hypothesis for a mileage entered by the user.

The subject permits libraries in general, but forbids a library that does the core work for us. In particular, calling a regression helper such as `numpy.polyfit` would fail the learning objective and evaluation requirement. We will implement the prediction and gradient-descent equations ourselves with Python's standard library.

## 2. The vocabulary


| Term                                      | Meaning in this project                                           |
| ----------------------------------------- | ----------------------------------------------------------------- |
| Dataset                                   | The historical examples in `data/data.csv`.                       |
| Example / sample / row                    | One known car, such as `(240000 km, 3650 price)`.                 |
| Feature, input, independent variable      | The value used to make a prediction: mileage (`x`).               |
| Target, label, output, dependent variable | The known value we want to predict: price (`y`).                  |
| Model                                     | The mathematical rule converting mileage into estimated price.    |
| Parameter / coefficient                   | A value learned by training: `theta0` or `theta1`.                |
| Hypothesis                                | The model's proposed output for one input: `h_theta(x)`.          |
| Prediction / inference                    | Applying already learned parameters to a new mileage.             |
| Training                                  | Improving parameters from known examples.                         |
| Residual / error                          | `prediction - actual_price` for one example.                      |
| Loss                                      | A number measuring error for an example.                          |
| Cost                                      | A number measuring loss over the whole dataset.                   |
| Iteration / epoch                         | One complete gradient-descent parameter update over all rows.     |
| Learning rate                             | The size of each gradient-descent step, commonly written `alpha`. |




## 3. What the supplied data tells us

The supplied CSV has this schema:

```csv
km,price
240000,3650
139800,3800
...
```

It contains 24 cars. Mileage ranges from **22,899 km** to **240,000 km** and price ranges from **3,650** to **8,290**. The general pattern is that lower-mileage cars tend to cost more, so we expect a learned slope (`theta1`) below zero.

That is an observed association in this small dataset. It does **not** prove that mileage alone causes every price difference. Age, make, condition, engine, maintenance history, market demand, and many other variables affect price. This project intentionally uses only one feature to teach the algorithm, so its predictions are limited.

## 4. Supervised learning and regression

This is **supervised learning** because training data includes both:

```text
input (mileage) + correct answer (known price)
```

The trainer compares each prediction against the known price, measures how wrong the current model is, and uses that information to improve its parameters.

This is **regression** because price is a continuous numeric quantity. A different task such as deciding whether a car is `"cheap"` or `"expensive"` would be classification, because its output is a category.

The model is not memorizing every row. It finds one line that summarizes the general relationship. For a mileage that was not present in the CSV, it can still produce a price estimate. That ability is useful, but means a prediction is an estimate, not a guarantee.

## 5. The hypothesis: a line represented by two values

For one input `x`, write the hypothesis as:

```text
h_theta(x) = theta0 + theta1 * x
```

This is the familiar line equation `y = b + mx`:


| Regression notation | Geometry name         | Meaning here                                        |
| ------------------- | --------------------- | --------------------------------------------------- |
| `theta0`            | intercept             | Predicted price at `0 km`.                          |
| `theta1`            | slope                 | Predicted price change for an additional kilometre. |
| `x`                 | horizontal coordinate | Mileage.                                            |
| `h_theta(x)`        | vertical coordinate   | Estimated price.                                    |


In general machine-learning vocabulary, the same equation is commonly written:

```text
y_hat = b + w * x
```

The names are equivalent:

```text
y_hat   = h_theta(x) = predicted price
b       = theta0     = bias (also called intercept)
w       = theta1     = weight (also called coefficient or slope)
x       = mileage
```

The **bias** is an independent starting value. It shifts every prediction upward or downward by the same amount. The **weight** scales an input feature and determines how strongly that feature changes the prediction. This project has one feature (`mileage`), so it has one weight (`theta1`). A model with several input features would have one weight for each feature:

```text
predicted_price = bias
                + weight_mileage * mileage
                + weight_age * age
                + weight_engine * engine_size
```

For example, suppose a trained model is:

```text
theta0 = 9000
theta1 = -0.020
```

For a car with `100000 km`:

```text
h_theta(100000) = 9000 + (-0.020 * 100000)
                 = 9000 - 2000
                 = 7000
```

The slope is negative, so every extra kilometre lowers this hypothetical estimate by `0.02` price units. A slope of `-0.020` is also `-20` price units per 1,000 km.

### The initial model

Before training, the subject requires:

```text
theta0 = 0
theta1 = 0
```

So every prediction is zero:

```text
h_theta(84000) = 0 + 0 * 84000 = 0
```

This is deliberately poor, but it gives gradient descent a defined starting point. If the predictor is run before a trained model is available, it must use this initial model and therefore estimate `0`.

### An important interpretation limit

`theta0` is mathematically the predicted price at `0 km`, but that does not mean a real zero-mileage car would cost exactly that amount. The data may contain no values close to zero. Interpreting a line far beyond the data range is called **extrapolation** and is less trustworthy than predictions inside the observed range (**interpolation**).

## 6. Errors, loss, and cost

For a row `i`, with mileage `x[i]` and real price `y[i]`:

```text
prediction[i] = h_theta(x[i])
error[i]      = prediction[i] - y[i]
```

For the untrained model and the dataset row `(84000, 6200)`:

```text
prediction = 0
error      = 0 - 6200 = -6200
```

The negative sign says the prediction is too low. If a model predicted `7000` instead, its error would be `7000 - 6200 = +800`: too high.

We cannot add signed errors directly because a `+800` and a `-800` would cancel even though both predictions are wrong. A standard solution is to square each error.

```text
squared_error[i] = (prediction[i] - y[i])^2
```

The most common whole-dataset cost function is:

```text
J(theta0, theta1) = (1 / (2m)) * sum from i=0 to m-1 of (h_theta(x[i]) - y[i])^2
```

`m` is the number of examples: **24** for the supplied dataset. Dividing by `m` makes the measurement comparable across dataset sizes. The factor `1/2` is conventional because it cancels a `2` produced when differentiating the square; it does not change the location of the best line.

The corresponding mean squared error is:

```text
MSE = (1 / m) * sum(error[i]^2)
```

Therefore `J = MSE / 2`. Either can monitor convergence as long as it is named correctly.

Squaring has two effects:

1. all contributions become non-negative, so errors cannot cancel;
2. large mistakes receive much more penalty than small ones (`100^2` is 10,000, while `10^2` is 100).

It also makes the cost smooth and differentiable, which is why gradient descent can follow its slope.

## 7. Gradient descent: learning the two parameters

Imagine cost as a landscape whose position is `(theta0, theta1)` and whose height is `J`. A good model lies near a low point. The **gradient** points uphill: toward the fastest increase in cost. Gradient descent moves in the opposite direction to reduce cost.

For this project, batch gradient descent uses every example to calculate one update:

```text
gradient0 = (1 / m) * sum(error[i])
gradient1 = (1 / m) * sum(error[i] * x[i])

theta0_new = theta0_old - learning_rate * gradient0
theta1_new = theta1_old - learning_rate * gradient1
```

These are the subject formulas expressed so their roles are unambiguous. Some presentations call `learning_rate * gradient` a temporary value (`tmp_theta0`, `tmp_theta1`) and subtract it from the old parameter. The key behavior is identical: calculate both changes from the old parameters, then apply both changes.

### Where the adjustment formulas come from

The cost function is:

```text
J(theta0, theta1) = (1 / (2m)) * sum(error[i]^2)
error[i] = (theta0 + theta1 * x[i]) - y[i]
```

The two gradient formulas are the **partial derivatives** of this cost: each measures how cost changes when one parameter changes by a tiny amount while the other is temporarily held fixed.

```text
partial J / partial theta0 = (1 / m) * sum(error[i])
partial J / partial theta1 = (1 / m) * sum(error[i] * x[i])
```

For the bias, changing `theta0` by `+1` changes every prediction by `+1`. Its gradient therefore combines the errors themselves.

For the weight, changing `theta1` by `+1` changes a prediction by its input `x`. A row at `x = 10` is changed ten times more than a row at `x = 1`, so its error is multiplied by `x` in the weight gradient.

The update rule always has this form:

```text
new_parameter = old_parameter - learning_rate * gradient_for_that_parameter
```

The minus sign is essential: a positive gradient says that increasing the parameter raises cost, so we reduce it. A negative gradient says that increasing the parameter lowers cost, and subtracting the negative value increases the parameter.

### Why simultaneous updates are mandatory

This is correct:

```text
next_theta0 = theta0 - alpha * gradient0_using_old_theta0_and_theta1
next_theta1 = theta1 - alpha * gradient1_using_old_theta0_and_theta1

theta0 = next_theta0
theta1 = next_theta1
```

This is incorrect:

```text
theta0 = theta0 - ...
theta1 = theta1 - ... using_the_already_changed_theta0
```

The second version changes the mathematical algorithm. The two partial derivatives describe a single position on the cost surface, so both must be calculated at the same old position.

### Worked one-row update

Use a deliberately small invented dataset with one row:

```text
x = 10
y = 100
m = 1
theta0 = 0
theta1 = 0
alpha = 0.01
```

First calculate the prediction and error:

```text
prediction = 0 + 0 * 10 = 0
error = 0 - 100 = -100
```

Then calculate the gradients:

```text
gradient0 = (1 / 1) * -100      = -100
gradient1 = (1 / 1) * (-100*10) = -1000
```

Finally update both parameters:

```text
theta0_new = 0 - 0.01 * (-100)  = 1
theta1_new = 0 - 0.01 * (-1000) = 10
```

Both values increase because the estimate was too low. This particular example also shows why input magnitude matters: the slope update was ten times larger because it was multiplied by `x = 10`.

### Batch gradient descent

The trainer will use **batch** gradient descent. It calculates both gradients across all 24 rows, then performs one update. This is the approach directly described by the subject's sums.

Other variants exist:

- stochastic gradient descent updates after one example;
- mini-batch gradient descent updates after a small group.

They are valid machine-learning techniques, but we will not substitute them for the subject's required batch calculation.

### Learning rate and iterations

The learning rate `alpha` controls the step size:

- too small: cost usually decreases, but training is very slow;
- too large: updates can overshoot the minimum, oscillate, or grow without bound;
- suitable: cost decreases steadily toward a stable value.

Training also needs an iteration count or a convergence rule. We will begin with a fixed, documented maximum number of iterations and inspect the cost. Later, a tolerance-based early-stop rule can be considered if it is transparent, tested, and does not obscure the required algorithm.

Convergence means the parameters and cost have become effectively stable; it does not mean the model perfectly predicts each row. A single straight line generally cannot pass through all real-world data points.

## 8. Connection to neural networks

Linear regression is the simplest form of a neural-network computation: one input, one weight, one bias, and no activation function.

```text
input x -> multiply by weight theta1 -> add bias theta0 -> output y_hat
```

Equivalently:

```text
z = w * x + b
y_hat = z
```

In a neural network, a unit (often called a neuron) normally has multiple inputs:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
activation = f(z)
```

- `x1 ... xn`: input features or outputs from previous neurons;
- `w1 ... wn`: learned weights, one for each input;
- `b`: learned bias;
- `z`: the weighted sum before activation;
- `f`: an activation function, such as ReLU or sigmoid;
- `activation`: the neuron's output, passed to the next layer or used as a prediction.

Our linear-regression model is the special case:

```text
z = theta1 * mileage + theta0
f(z) = z
```

The activation is the identity function: it returns its input unchanged. That is why the result is a straight line.

### How neural networks adjust weights and biases

The high-level learning loop is exactly the same as this project:

```text
1. Forward pass: use current weights and biases to make predictions.
2. Loss calculation: compare predictions with correct answers.
3. Gradient calculation: determine how each parameter affected the loss.
4. Update: subtract learning_rate * gradient from every parameter.
5. Repeat.
```

For linear regression, the two gradients can be written directly, as this subject requires. In a multi-layer neural network, each weight can influence the final output through many later neurons. **Backpropagation** uses the chain rule from calculus to efficiently calculate the gradient for every weight and bias, starting at the output loss and working backward through the layers.

The parameter update remains the same:

```text
w_new = w_old - alpha * (partial loss / partial w)
b_new = b_old - alpha * (partial loss / partial b)
```

So gradient descent is the optimizer, while backpropagation is the efficient method used to obtain gradients in a layered network. This project needs gradient descent, but not a neural network or backpropagation.

### Why activation functions matter

If every layer only did `w*x + b`, stacking layers would still be equivalent to one linear equation. Nonlinear activation functions let neural networks learn curved and more complex relationships.

For example, ReLU is:

```text
ReLU(z) = max(0, z)
```

It outputs zero for negative values and preserves positive values. Neural networks need such nonlinearities; this project intentionally does not because its goal is a single straight-line relationship.

## 9. Why normalization matters

The supplied mileage inputs can be as large as `240000`. In the `theta1` gradient, each error is multiplied by mileage:

```text
gradient1 = (1 / m) * sum(error[i] * mileage[i])
```

This produces much larger numbers for the slope update than for the intercept update. A learning rate safe for one parameter can be wildly unsafe for the other. This makes unscaled gradient descent difficult to tune and can cause divergence.

We will use **min-max normalization** for the input feature during training:

```text
x_normalized = (x - x_min) / (x_max - x_min)
```

For the supplied dataset:

```text
x_min = 22899
x_max = 240000
range = 217101
```

Examples:

```text
normalize(22899)  = (22899 - 22899) / 217101 = 0
normalize(240000) = (240000 - 22899) / 217101 = 1
normalize(84000)  = (84000 - 22899) / 217101 ≈ 0.28144
```

The model is trained with normalized mileage:

```text
h_theta(x_normalized) = theta0 + theta1 * x_normalized
```

The predictor must normalize an entered mileage with the **same saved** `x_min` and `x_max` before applying the trained parameters. It must never recompute them from the one value being predicted.

### What must be saved

If training uses normalization, `theta0` and `theta1` alone are insufficient to make a later prediction. The saved model also needs the transformation parameters:

```text
theta0, theta1, x_min, x_max
```

It is reasonable to save a model-format version and training configuration too, but those are implementation details to document once we build the trainer.

### Constant-feature edge case

If all mileage values were equal, then `x_max - x_min` would be zero and min-max normalization would divide by zero. The trainer must reject this dataset clearly because a slope cannot be learned from a feature with no variation.

## 10. Evaluating a trained model

Evaluation answers: “how close are predictions to known prices?” It is separate from training, even if evaluated on the same small dataset for this exercise.

### MAE

Mean absolute error:

```text
MAE = (1 / m) * sum(abs(prediction[i] - actual[i]))
```

It reports a typical error in price units and is often intuitive to explain: “predictions are off by about this many price units on average.”

### MSE and RMSE

```text
MSE  = (1 / m) * sum(error[i]^2)
RMSE = sqrt(MSE)
```

RMSE is also in price units, while MSE is in squared price units. Because errors are squared, both emphasize large errors more strongly than MAE.

### R-squared

The coefficient of determination compares the model against a simple baseline that always predicts the average price:

```text
R² = 1 - (sum(error[i]^2) / sum((actual[i] - mean_price)^2))
```

- `R² = 1`: perfect predictions on the evaluated data;
- `R² = 0`: no better than always predicting the mean price;
- `R² < 0`: worse than that mean-price baseline.

R² is useful, but never a complete proof of quality. It does not show whether the causal story is true, whether new data will behave similarly, or whether individual errors are acceptable.

## 11. Visualizing the result

The bonus plot should contain:

- a scatter plot: each observed `(mileage, price)` row;
- the fitted line: model predictions across the mileage range;
- clear axis labels, units, and a legend.

The plot is a diagnostic tool. A good line follows the centre of the cloud of points. Patterns matter:


| Pattern                                  | Possible meaning                                                 |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Randomly scattered residuals around zero | A line may be a reasonable approximation.                        |
| Curved residual pattern                  | The true relationship may not be linear.                         |
| One extreme isolated point               | A potential outlier may strongly affect squared-error training.  |
| Widely expanding scatter                 | Error size changes by mileage; the model omits relevant factors. |


The fitted line minimizes squared error under the chosen algorithm; it does not need to pass through every point.

## 12. Data validation and responsible behavior

The implementation must define and document predictable behavior for invalid input:

- missing dataset file;
- missing or incorrect `km,price` header;
- empty dataset;
- non-numeric or non-finite values such as `NaN` and `inf`;
- a dataset with fewer than enough useful examples;
- all mileage values identical;
- missing/corrupt saved model;
- blank, non-numeric, negative, or non-finite mileage supplied to the predictor.

We will treat mileage as a non-negative physical quantity. The predictor should display an actionable error rather than silently converting invalid input into a misleading estimate.

Prices predicted by an unconstrained line can theoretically become negative at large mileages. `predict.py` prints the **raw** hypothesis value so it always matches `theta0 + theta1 * mileage`. We do not clamp to zero. A negative printout is a limitation of a straight line, not a real market price.

## 13. Implementation — current status

This section grows as we code. Each subsection records a design choice and maps it to the theory above.

### 13.1 Project layout

```text
ft_linear_regression/
  train.py                      mandatory trainer
  predict.py                    mandatory predictor
  precision.py                  bonus: metrics + charts
  plot.py                       bonus: scatter + fitted line
  bonus_results/                generated charts (created by make bonus)
  requirements.txt              matplotlib for plot.py only
  data/data.csv                 subject dataset: km,price
  src/linear_regression.py      hypothesis, scaling, cost, metrics, gradient descent
  src/dataset.py                CSV loading and validation
  src/model.py                  save / load model.json
  tests/                        unit and CLI tests
  Makefile                      train / test / plot / clean
  LEARNING.md                   this study guide
  README.md                     evaluator-facing usage
  .gitignore                    local junk and generated artifacts
```

`train.py` and `predict.py` do not exist yet. They will call the same functions in `src/` so the trainer and the predictor cannot drift apart.

`.gitignore` excludes:

- `.venv/` — a local virtual environment must never be submitted;
- `__pycache__/` and `*.pyc` — Python bytecode;
- `model.json` — generated trained parameters, not source;
- `*.png` — generated bonus plots.



### 13.2 Why a `src/` module instead of putting math in the scripts

The subject requires two programs that share the same hypothesis. If we copy `theta0 + theta1 * mileage` into both files, one of them can be changed later and evaluation would fail. One function, imported by both programs, is the shared source of truth.

We use only the Python standard library for this step: no NumPy, no pandas, no scikit-learn.

### 13.3 The hypothesis in code

```python
def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage
```

This is the required formula, written as a function:

```text
h_theta(x) = theta0 + theta1 * x
```


| Function argument | Math symbol   | Role            |
| ----------------- | ------------- | --------------- |
| `mileage`         | `x`           | input feature   |
| `theta0`          | `θ₀` / bias   | intercept       |
| `theta1`          | `θ₁` / weight | slope           |
| return value      | `h_θ(x)`      | predicted price |


The function does not train anything. It does not read a file. It only applies the current line to one mileage. Training will later change the two theta values; prediction will later ask the user for a mileage and call this same function.

Type hints (`float`) are documentation for us and for the evaluator. They do not change the arithmetic.

### 13.4 Tests that lock the formula

`tests/test_hypothesis.py` uses Python's built-in `unittest`. No extra test library is required.

The cases are the hand calculations from this guide:


| Test                  | Input                                    | Expected result | Why it exists                       |
| --------------------- | ---------------------------------------- | --------------- | ----------------------------------- |
| untrained model       | `mileage=84000`, both thetas `0`         | `0`             | subject: before training, predict 0 |
| worked example        | `100000`, `theta0=9000`, `theta1=-0.020` | `7000`          | section 5 arithmetic                |
| one-row start         | `x=10`, both thetas `0`                  | `0`             | start of the section 7 example      |
| after one manual step | `x=10`, `theta0=1`, `theta1=10`          | `101`           | end of the section 7 example        |


The last case is not yet produced by a training function. It only checks that if gradient descent later reaches those thetas, the hypothesis still uses them correctly:

```text
1 + 10 * 10 = 101
```

Run the tests from the project root:

```text
python3 -m unittest discover -s tests -b
```



### 13.5 Min-max normalization in code

```python
normalize(value, x_min, x_max) = (value - x_min) / (x_max - x_min)
denormalize(value, x_min, x_max) = value * (x_max - x_min) + x_min
```

These are exact inverses of each other. Tests lock the dataset bounds from section 9:

```text
normalize(22899)  = 0
normalize(240000) = 1
normalize(84000)  ≈ 0.281440435
```

If `x_max == x_min`, both functions raise `ValueError`. That is the constant-feature case: every car has the same mileage, so a slope cannot be learned and division by zero would occur.

`x_min` and `x_max` must come from the **training dataset**, never from the single mileage being predicted later.

### 13.6 Why we convert normalized thetas back to raw kilometres

Training will use normalized mileage so gradient descent stays stable. The subject hypothesis, however, is written for raw mileage:

```text
estimatePrice(mileage) = theta0 + theta1 * mileage
```

If we leave thetas in normalized units, `predict.py` would have to normalize every user input. That is valid, but then the saved `theta1` would not mean "price change per kilometre".

Algebra of the conversion:

```text
x_n = (x - x_min) / span
span = x_max - x_min

h = theta0_n + theta1_n * x_n
  = theta0_n + theta1_n * (x - x_min) / span
  = theta0_n - (theta1_n * x_min) / span + (theta1_n / span) * x
```

Therefore:

```text
theta1 = theta1_n / span
theta0 = theta0_n - (theta1_n * x_min) / span
```

`thetas_to_original` implements that. After conversion, `estimate_price(raw_km, theta0, theta1)` equals `estimate_price(normalize(raw_km), theta0_n, theta1_n)`.

Worked test:

```text
theta0_n = 10, theta1_n = 20, x_min = 100, x_max = 200
theta1 = 20 / 100 = 0.2
theta0 = 10 - 20 * 100 / 100 = -10

For x = 150:
  normalized path: 10 + 20 * 0.5 = 20
  raw path:       -10 + 0.2 * 150 = 20
```



### 13.7 Cost and metrics in code

All scoring functions share `_errors`, which builds `prediction - actual` for every row.


| Function                  | Formula                    | Units             |
| ------------------------- | -------------------------- | ----------------- |
| `cost`                    | `(1 / (2m)) * sum(error²)` | squared price / 2 |
| `mean_squared_error`      | `(1 / m) * sum(error²)`    | squared price     |
| `root_mean_squared_error` | `sqrt(MSE)`                | price             |
| `mean_absolute_error`     | `(1 / m) * sum(|error|)`   | price             |
| `r_squared`               | `1 - SSE / SST`            | unitless          |


`cost` is `MSE / 2`. We keep both names because the theory section uses `J`, while bonus precision reports usually quote MAE / RMSE / R².

Hand-checked cases in `tests/test_metrics.py`:

```text
One row, untrained: x=10, y=100, thetas=0
  error = -100
  MSE = 10000
  J   = 5000

Two rows, untrained: (0, 10) and (1, 12)
  errors = -10, -12
  MAE = 11
  MSE = 122
  J   = 61
  R²  = 1 - 244 / 2 = -121   (worse than predicting the mean price 11)

Perfect line theta0=10, theta1=2 on those two rows:
  MAE = MSE = J = 0, R² = 1
```

If all actual prices are identical, `SST = 0` and R² is undefined unless the predictions are also perfect (`SSE = 0`), in which case we return `1`. Empty or mismatched lists raise `ValueError`.

### 13.8 CSV loading

`src/dataset.py` is separate from the math module on purpose: reading a file is not part of the hypothesis or the gradient. `train.py` will call `load_dataset`, then pass the two lists into the math functions.

```python
mileages, prices = load_dataset("data/data.csv")
```

`m` in the subject formulas is `len(mileages)`. For the supplied file that is `24`.

We use the standard-library `csv` module, not pandas. `csv.DictReader` reads the header row, then each following row as a dictionary: `{"km": "240000", "price": "3650"}`.

`DatasetError` is a `ValueError` subclass so `train.py` can catch dataset problems without catching unrelated bugs.

Validation policy:


| Situation                      | Result                                            |
| ------------------------------ | ------------------------------------------------- |
| file does not exist            | `DatasetError`                                    |
| empty file / no header         | `DatasetError`                                    |
| header missing `km` or `price` | `DatasetError`                                    |
| fewer than 2 valid rows        | `DatasetError` (a line needs two distinct points) |
| all km values identical        | `DatasetError` (normalize would divide by zero)   |
| non-numeric cell (`abc`)       | `DatasetError` with line number                   |
| `inf` or `NaN`                 | `DatasetError`                                    |
| negative km or price           | `DatasetError`                                    |
| blank line                     | skipped                                           |
| subject `data/data.csv`        | 24 mileages and 24 prices                         |


Numbers are parsed with `float(...)`. `math.isfinite` rejects `inf` and `NaN` because those would poison every later sum.

We require at least two rows because:

- one point does not define a unique line;
- min-max normalization needs `x_max > x_min`.

The loader does not train and does not call `estimate_price`. It only produces clean lists.

### 13.9 Gradient descent in code



#### Start here if `gradient_step` still feels abstract

We have a line:

```text
predicted_price = theta0 + theta1 * mileage
```

At the beginning both thetas are `0`, so the line is “every car costs 0”. That is wrong. We want to **nudge the two numbers** until the line sits closer to the real points.

`gradient_step` is **one nudge**. That is all. It does not finish training. It looks at the current mistakes, decides “move the line a little this way”, and returns the slightly better thetas.

`learning_rate` and `alpha` are the **same thing**. Two names, one number:

```text
alpha = learning_rate = how big that one nudge is
```

- `alpha = 0.01` → tiny nudge, safe, slow
- `alpha = 0.1` → medium nudge
- `alpha = 10` → huge nudge, can jump past the good line and explode

We do **not** learn alpha from the data. We pick it, like choosing a walking speed.

What we are doing technically, in order:

```text
1. Use the current thetas to predict every car.
2. error = prediction - real price     (too high = positive, too low = negative)
3. Average those errors                → “should the whole line go up or down?”
   That average is the gradient for theta0 (the bias).
4. Average (error * mileage)           → “should the slope get steeper or flatter?”
   That average is the gradient for theta1 (the weight).
5. Move each theta a little opposite to its gradient:
      new_theta = old_theta - alpha * gradient
6. Return both new thetas together.
```

“Opposite to the gradient” is the whole trick. The gradient points toward **worse** cost. Subtracting it points toward **better** cost. That is why the method is called **gradient descent**: go down the cost.

Picture a valley. Your position is `(theta0, theta1)`. Height is “how wrong the line is”. The gradient says “uphill is that way”. You take one step downhill. Alpha is the length of that step.

One nudge is almost never enough. `train` calls `gradient_step` again and again (1000 times by default). Each time the errors get a bit smaller, until the line stops moving much. That is training.

Two functions, on purpose:


| Function        | Job                                                                  |
| --------------- | -------------------------------------------------------------------- |
| `gradient_step` | one simultaneous update of both thetas                               |
| `train`         | start at `(0, 0)`, normalize x, repeat the step, convert thetas back |


`gradient_step` is one footstep. `train` is the walk: it chooses the path (normalized km), starts at the required point `(0, 0)`, takes many footsteps, then translates the result back into raw kilometres.

#### `gradient_step` arguments

```python
def gradient_step(mileages, prices, theta0, theta1, learning_rate) -> tuple[float, float]:
```


| Argument        | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| `mileages`      | the `x` values used in this step (raw or already normalized) |
| `prices`        | the known `y` values, same length as `mileages`              |
| `theta0`        | current bias                                                 |
| `theta1`        | current weight                                               |
| `learning_rate` | step size `alpha`                                            |
| return value    | the **new** `(theta0, theta1)`, not applied in place         |


It returns a new pair instead of changing the arguments, so the caller can write:

```python
theta0, theta1 = gradient_step(...)
```

That one assignment updates both parameters at the same moment.

#### `gradient_step` line by line

```python
errors = _errors(mileages, prices, theta0, theta1)
```

For every car, compute `estimate_price(x, theta0, theta1) - price`.  
All errors use the **same old** thetas. That is the batch, and it is also why the later update is simultaneous: `tmp_theta1` never sees a half-updated `theta0`.

Example, untrained, two cars `(x=0, y=10)` and `(x=1, y=12)`:

```text
errors = [0 - 10, 0 - 12] = [-10, -12]
```

```python
sample_count = len(errors)
```

This is `m` in the subject. For those two cars, `m = 2`. For the real dataset, `m = 24`.

```python
tmp_theta0 = learning_rate * sum(errors) / sample_count
```

This is the subject’s first temporary:

```text
tmp_theta0 = alpha * (1/m) * sum(error)
```

`sum(errors) / sample_count` is the average error, also called `gradient0`.  
If the average error is negative, predictions are too low, so we will **increase** `theta0` (because we subtract a negative).  
If the average error is positive, predictions are too high, so we will **decrease** `theta0`.

With `alpha = 0.1`:

```text
tmp_theta0 = 0.1 * (-10 + -12) / 2 = 0.1 * -11 = -1.1
```

```python
tmp_theta1 = learning_rate * sum(error * mileage for error, mileage in zip(errors, mileages)) / sample_count
```

This is the subject’s second temporary:

```text
tmp_theta1 = alpha * (1/m) * sum(error * x)
```

`zip(errors, mileages)` walks the two lists together: first error with first x, second with second.

```text
(-10 * 0) + (-12 * 1) = -12
tmp_theta1 = 0.1 * -12 / 2 = -0.6
```

The extra `* x` is there because `theta1` is the slope. Changing the slope moves a prediction by `x`, so an error at a larger `x` should pull the slope harder.

```python
return theta0 - tmp_theta0, theta1 - tmp_theta1
```

Subtract, do not add. A positive `tmp` means “cost grows if this theta grows”, so we go the other way.

```text
theta0 = 0 - (-1.1) = 1.1
theta1 = 0 - (-0.6) = 0.6
```

The function does not assign `theta0 = ...` and then compute `tmp_theta1`. Both new values are produced from the old pair, then returned together.

#### Why a sequential update would be wrong

Start at `theta0=1`, `theta1=1`, one point `x=2`, `y=0`, `alpha=1`:

```text
prediction = 1 + 1*2 = 3
error = 3
tmp_theta0 = 3
tmp_theta1 = 3*2 = 6
correct result = (1-3, 1-6) = (-2, -5)
```

Wrong version: change `theta0` to `-2` first, then recompute the error for `theta1`:

```text
prediction = -2 + 1*2 = 0
error = 0
tmp_theta1 = 0
wrong result = (-2, 1)
```

That is a different algorithm. The test `test_updates_are_simultaneous` locks `(-2, -5)`.

#### `gradient_step` does not normalize

It is deliberately dumb: it treats `mileages` as “the x I was given”.  
If you pass raw kilometres (`240000`), `error * x` becomes huge and a normal `alpha` explodes.  
`train` therefore passes **normalized** kilometres in `[0, 1]`.

#### `train` arguments

```python
def train(mileages, prices, learning_rate=0.1, iterations=1000) -> tuple[float, float]:
```

`learning_rate` and `iterations` have defaults so a caller can write `train(mileages, prices)`.  
They are **hyperparameters**: we choose them; the dataset does not. They are not learned.

- `learning_rate` (`alpha`): how big each footstep is.
- `iterations`: how many footsteps we take.

Too small `alpha` → slow. Too large `alpha` → cost jumps up and can become `inf`. Too few iterations → we stop before the line settles.

#### `train` line by line

```python
if learning_rate <= 0: raise ...
if iterations < 1: raise ...
```

A zero or negative step size is not a descent. Zero iterations would return the untrained model `(0, 0)` after conversion, which looks like “training ran” but did nothing.

```python
x_min = min(mileages)
x_max = max(mileages)
normalized = [normalize(mileage, x_min, x_max) for mileage in mileages]
```

Find the training-set bounds once. Map every km into `[0, 1]`.  
For `(0, 10)` and `(1, 12)`, bounds are already `0` and `1`, so normalized equals the original x.  
For the real cars, `22899 → 0` and `240000 → 1`.

The same `x_min` / `x_max` are reused at the end when we convert thetas. We must not recompute them from something else.

```python
theta0 = 0.0
theta1 = 0.0
```

Subject rule: before training, both parameters are zero. Every first prediction is `0`.

```python
for _ in range(iterations):
    theta0, theta1 = gradient_step(normalized, prices, theta0, theta1, learning_rate)
```

`range(iterations)` is `0, 1, ..., iterations-1`.  
`_` means “we do not use the loop index”; we only care that the body runs that many times.

Each pass:

1. predict every price with the **current** thetas and **normalized** x;
2. compute both `tmp` values from those errors;
3. replace both thetas.

`prices` stay raw. Only `x` is scaled. Price is the target, not a feature, so we do not normalize it.

After iteration 1 on `(0, 10)`, `(1, 12)`, `alpha=0.1` we have `(1.1, 0.6)`.  
After many iterations we approach `(10, 2)`, the perfect line.

```python
return thetas_to_original(theta0, theta1, x_min, x_max)
```

The loop’s thetas belong to normalized x:

```text
price ≈ theta0_n + theta1_n * x_normalized
```

The subject wants:

```text
price ≈ theta0 + theta1 * raw_km
```

`thetas_to_original` does that change of units. After the return, `predict.py` can call `estimate_price(user_km, theta0, theta1)` with the user’s raw mileage.

On `(0, 10)` and `(1, 12)`, conversion does almost nothing because the span is `1`. On the real dataset, `theta1_n` is “price change from lowest-km car to highest-km car”, and `theta1` becomes “price change per kilometre” (a small negative number).

#### What `train` does not do

- It does not read a CSV (`load_dataset` already did that).
- It does not save a file (next step: `model.json`).
- It does not prompt a user (that is `predict.py`).
- It does not plot.

It only returns two numbers: the learned bias and weight in raw-km units.

#### Four questions that are easy to mix up

**Is** `sum(error)` **MAE, MSE, or RMSE?**

None of them. In the gradient, `error` is only:

```text
error = predicted_price - real_price
```

That is a signed leftover. It still has `+` or `-`. MAE, MSE, and RMSE are **reports** we compute later to say “how good is the finished line?”


| Name          | What it does to the errors                                     |
| ------------- | -------------------------------------------------------------- |
| this gradient | averages the raw signed errors (and for `theta1`, `error * x`) |
| MAE           | average of `abs(error)` — always positive                      |
| MSE           | average of `error²`                                            |
| RMSE          | square root of MSE                                             |
| `cost` / `J`  | MSE / 2                                                        |


We **minimize** `J` (squared errors). After you differentiate `J` with respect to `theta0`, the `²` and the `1/2` disappear, and you are left with the average of the signed errors. So `sum(error) / m` is not a metric we publish. It is “which way should the bias move?”

**What are** `theta0` **and** `theta1` **in real life?**

Imagine the price tag the model would write:

- `theta0` = “I think a car with **0 km** costs this much.” The starting height of the line.
- `theta1` = “For **each extra kilometre**, I add this to the price.” Usually negative: more km → cheaper.

If `theta0 = 8000` and `theta1 = -0.02`:

```text
price = 8000 + (-0.02) * km

0 km     → 8000 + (-0.02)*0 = 8000     ← this is theta0, not theta1
1000 km  → 8000 + (-0.02)*1000 = 7980
```

When mileage is `0`, the `theta1 * 0` term vanishes. The prediction is `theta0`, never `theta1`. `theta1` is only the tilt (`-0.02` per km). It is not a price.

Those `8000` / `-0.02` numbers are a teaching example, not the trained model.

Also, `theta0` is **not** “the true price of a brand-new car”. It is where our straight line crosses the price axis if we extend it to `0 km`. The real dataset’s smallest mileage is **22,899 km**. We have no 0 km cars. Trusting the line that far left is **extrapolation**. In this file the highest prices are already around 8,000–8,290, so an intercept near 8,000 is not “a cheap new car”; it is just the line continuing the same trend. A real 0 km car could cost much more because of age, brand, and condition, which this model ignores.

They are not errors, not learning rates, not cars. They **are** the line.

**Why multiply by** `x` **for** `theta1`**?**

Two knobs, two effects. Start with `theta0 = 0`, `theta1 = 0` (every prediction is `0`).

```text
Car A: 0 km,   real price 10
Car B: 10 km,  real price 30
```

If I change **only** `theta0` to `1` (add 1 to everyone):

```text
Car A prediction: 1 + 0 * 0  = 1     (moved by 1)
Car B prediction: 1 + 0 * 10 = 1     (moved by 1)
```

Same move for both cars. So for `theta0` we just average the errors. No extra `x`.

If I change **only** `theta1` to `1` (tilt):

```text
Car A prediction: 0 + 1 * 0  = 0     (moved by 0)
Car B prediction: 0 + 1 * 10 = 10    (moved by 10)
```

The same tilt barely touches a 0 km car and hits a 10 km car ten times harder. So when we ask “should the slope go up or down?”, Car B’s mistake must count more. We do that by `error * x`. For Car A, `x = 0`, so `error * 0 = 0`: a 0 km car cannot teach us the slope. For Car B, `x = 10`, so its error is multiplied by 10.

**Why** `new = old - tmp`**, not** `new = tmp`**?**

`tmp` is **how far to walk**, not **where to stand**.

```text
tmp_theta0 = alpha * (average error)
new_theta0 = old_theta0 - tmp_theta0
```

`old` is the current line. `tmp` is this step’s correction. You keep the line and apply the correction.

If we set `theta0 = tmp_theta0`, we would throw the old line away and replace it with “alpha times average error”. After one step from zero, `tmp` was `-1.1`. The new bias we wanted was `1.1`, which is `0 - (-1.1)`, not `-1.1` itself. `tmp` is the step, not the destination.

**Why minus tmp, not plus tmp?**

`tmp` has the same sign as the average error. Error is `prediction - reality`.

- Predictions too **high** → error positive → `tmp` positive → we must **lower** the line → `old - tmp`
- Predictions too **low** → error negative → `tmp` negative → `old - (negative)` becomes `old + amount` → we **raise** the line

The two-car start (`errors = -10, -12`, `alpha = 0.1`):

```text
tmp = -1.1
old - tmp = 0 - (-1.1) = 1.1    correct: we were too cheap, lift the line
old + tmp = 0 + (-1.1) = -1.1   wrong: drop the line, even cheaper, worse
```

Adding `tmp` would be gradient **ascent**: climb toward more error. The subject subtracts.

### 13.10 Saving and loading `model.json`

The subject requires two programs: the trainer writes `theta0` and `theta1`, the predictor reads them later. We store them in `model.json` (standard-library `json`, not a database).

`.gitignore` lists `model.json` because it is **generated**. An evaluator trains, then predicts. We do not commit a leftover trained file as if it were source.

#### What is stored

```json
{
  "version": 1,
  "theta0": 8000.0,
  "theta1": -0.02,
  "x_min": 22899.0,
  "x_max": 240000.0,
  "learning_rate": 0.1,
  "iterations": 1000
}
```

| Field | Why it is there |
| --- | --- |
| `theta0`, `theta1` | the subject parameters, already converted to **raw kilometres** |
| `x_min`, `x_max` | the training-set bounds used during normalization |
| `learning_rate`, `iterations` | the hyperparameters we actually used (for the report / debug) |
| `version` | so a future format change can be rejected instead of silently misread |

Prediction uses only `theta0` and `theta1`:

```text
estimate_price(mileage) = theta0 + theta1 * mileage
```

We still save `x_min` and `x_max` so we can explain how training was scaled, and so we could rebuild normalized thetas if needed. The predictor does **not** re-normalize the user's mileage, because `train()` already called `thetas_to_original`.

#### Functions

| Function | Job |
| --- | --- |
| `untrained_model()` | `theta0 = 0`, `theta1 = 0` — subject rule before training |
| `save_model(model, path)` | write JSON, overwrite if present |
| `load_model(path)` | read JSON, or raise `ModelError` |
| `Model.estimate(mileage)` | same as `estimate_price` with this model's thetas |

`load_model` does **not** invent zeros. A missing or corrupt file is an error. `predict.py` catches a **missing** file and uses `untrained_model()` so a first-time run still prints `0`. A **corrupt** file is still fatal.

Rejected on load: missing file, invalid JSON, missing fields, `version != 1`, non-finite numbers, `x_max == x_min`.

`Model` is a frozen dataclass: a simple bag of numbers. It cannot change after load, which avoids accidental edits.

### 13.11 The two programs

This is the data flow the subject asked for:

```text
data/data.csv
    -> train.py
        load_dataset
        train()                # start (0,0), normalize, descend, convert thetas
        save_model -> model.json
    -> predict.py
        load_model or untrained_model
        prompt mileage
        print theta0 + theta1 * mileage
```

#### `train.py`

1. Parse flags (`--data`, `--model`, `--learning-rate`, `--iterations`).
2. `load_dataset` — any `DatasetError` is printed and the process exits `1`.
3. `train(...)` — the math we already tested.
4. Build a `Model` with the learned thetas plus `min(km)`, `max(km)`, and the hyperparameters.
5. `save_model` writes `model.json`.
6. Print thetas plus `cost`, `MAE`, and `R²` on the training data so we can see if learning worked.

Defaults: `data/data.csv`, `model.json` next to the script, `learning_rate=0.1`, `iterations=1000`. Those defaults are for **normalized** x. They are configuration, not part of the algorithm.

#### `predict.py`

1. If `--model` exists, `load_model`. If the file is corrupt, exit `1`.
2. If the file is **missing**, use `untrained_model()` (`theta0=0`, `theta1=0`) and warn on stderr. That is the subject’s “before training” behaviour.
3. Prompt `mileage: `.
4. `parse_mileage` rejects empty text, non-numbers, `inf`/`NaN`, and negative km.
5. Print `estimated price: ` plus `model.estimate(mileage)`.

The printed number is the raw line, not `max(0, price)`. Evaluators can check it by hand: `theta0 + theta1 * mileage`.

#### Commands

```text
make test
make train
make predict
```

### 13.12 Bonus programs

The subject says bonuses are graded only if the mandatory part is perfect. So `plot.py` and `precision.py` never sit inside `train.py` / `predict.py`. A missing matplotlib install cannot break prediction.

#### `make bonus`

Creates `bonus_results/` if needed, then runs precision and plot. Look there for the charts:

| File | What it shows |
| --- | --- |
| `bonus_results/regression.png` | cars + fitted line |
| `bonus_results/precision_metrics.png` | MAE, RMSE, R² bars |
| `bonus_results/residuals.png` | error vs mileage |
| `bonus_results/precision.txt` | the same numbers as text |

#### `precision.py`

Loads the **already trained** `model.json` and the CSV, then prints:

```text
cost, MAE, MSE, RMSE, R²
```

These are the same formulas as in `src/linear_regression.py`. This program does not train again. It only **scores** the saved line on the known prices.

How to read them in the defense:

- **MAE** — typical miss, in price units, easy to say out loud.
- **RMSE** — also price units, but large mistakes hurt more (squares).
- **R²** — `1` is a perfect fit on this file; `0` is no better than always guessing the mean price. On the 24-car set we saw about `0.73`: mileage explains a lot, not everything.
- These numbers are computed on the **training** cars. They are not a test-set proof that new cars will behave the same.

#### `plot.py`

Loads the same files and draws:

- blue-ish scatter: each real `(km, price)`
- red line: `theta0 + theta1 * km` from the min km to the max km

Two endpoints are enough for a straight line (`line_endpoints`). We do not use `numpy.polyfit` to invent the line; we plot the thetas we already learned.

`matplotlib` is imported **inside** `run()`, after the files loaded. If it is missing, the program exits `1` with an install hint. The `Agg` backend writes PNGs into `bonus_results/` without opening a window.

A good plot for this dataset: points slope down to the right, and the red line follows the middle of the cloud. A pretty line is not proof of causation.

```text
make train
make bonus
```

## 14. Evaluator questions and answers

Say these in your own words. Numbers below match a default `make train` on `data/data.csv` (`alpha=0.1`, `1000` iterations) and can shift slightly with more iterations.

**What are the feature, target, parameters, and hypothesis?**  
Feature = mileage (`x`). Target = price (`y`). Parameters = `theta0` (bias) and `theta1` (weight). Hypothesis = `price = theta0 + theta1 * mileage`.

**Why supervised learning and why regression?**  
Each training row has the correct price. We learn from those answers. Price is a number, not a class, so it is regression, not classification.

**What does each theta mean, and why is the slope negative?**  
`theta0` is where the line would cross `0 km` (a mathematical intercept, not a real new-car price). `theta1` is the price change per extra kilometre. The CSV slopes down: more km, lower price, so `theta1` is negative (about `-0.021` after a default train). At `x = 0` the prediction is `theta0`, not `theta1`.

**What is `m`?**  
The number of training cars. Here `m = 24`.

**Why subtract the gradient (`old - tmp`) instead of adding?**  
`tmp` has the same sign as the average error (`prediction - reality`). Subtracting walks toward *less* error. Adding would make the line worse (ascent). `tmp` is the step size, not the new theta.

**Why must updates be simultaneous?**  
Both `tmp` values are computed from the same old line. If you change `theta0` first and recompute, `theta1` uses a different model. That is not the subject algorithm.

**What if the learning rate is wrong?**  
Too small: cost falls, but slowly. Too large: cost jumps or becomes `inf`. `alpha` is a choice we make, not a learned theta. Defaults: `0.1` on *normalized* km, `1000` iterations.

**Why normalize, and why save min/max?**  
Raw km (up to `240000`) makes `error * x` huge, so `theta1` steps explode. We train on `x` in `[0, 1]`, then convert thetas back to raw km with `thetas_to_original`. Prediction uses the converted thetas and does **not** normalize the user’s input. We still store `x_min` / `x_max` so we can explain the scaling.

**What does the predictor do before training?**  
No `model.json` → `theta0 = 0`, `theta1 = 0` → every price is `0`. Corrupt JSON is an error, not silent zeros.

**Cost vs MAE / MSE / RMSE / R²?**  
The gradient uses signed `error = prediction - price`. That is not a published metric. We *minimize* `J = MSE / 2`. MAE = average `|error|`. MSE = average `error²`. RMSE = `√MSE`. R² = `1` if we fit this file perfectly, `0` if we are no better than always predicting the mean price. After a default train: MAE ≈ `556`, R² ≈ `0.73`.

**Why can a good graph still be unreliable?**  
The line only summarizes these 24 cars and one feature. Correlation is not causation. Outside 22,899–240,000 km we extrapolate. The printed price is the raw line and can go negative.

**Which libraries are forbidden?**  
Anything that *fits* the line for us: `numpy.polyfit`, scikit-learn regression, pandas `ols`, … Allowed: `csv`, `json`, `math`, `argparse`, `unittest`, and matplotlib **only** to draw points we already computed.

**Walk through one program run.**  
`make train` → `load_dataset` → `train()` starts at `(0,0)`, normalizes, loops `gradient_step`, converts thetas, writes `model.json`. `make predict` loads that file (or zeros), asks for km, prints `theta0 + theta1 * km`.

**Why a `src/` module plus thin scripts?**  
The subject wants two programs sharing one hypothesis. The math lives in `src/linear_regression.py`. `train.py` / `predict.py` only handle files, flags, and printing.

## 15. Makefile

All evaluator commands from the project root:

| Target | What it runs |
| --- | --- |
| `make` / `make help` | list targets |
| `make test` | `python3 -m unittest discover -s tests -b` |
| `make train` | `python3 train.py` |
| `make predict` | `python3 predict.py` (interactive) |
| `make precision` | write precision charts into `bonus_results/` (trains first if needed) |
| `make plot` | write `bonus_results/regression.png` |
| `make bonus` | create `bonus_results/`, then precision + plot |
| `make all` | test + bonus |
| `make install` | `pip install -r requirements.txt` (matplotlib) |
| `make clean` | delete `model.json`, `bonus_results/`, `__pycache__` |

Make dependencies:

```text
data/data.csv + train.py + src/*.py  ->  model.json
model.json + precision.py            ->  bonus_results/precision_metrics.png
                                         bonus_results/residuals.png
                                         bonus_results/precision.txt
model.json + plot.py                 ->  bonus_results/regression.png
bonus  ->  precision + plot
all    ->  test + bonus
```

`make predict` does **not** depend on `model.json`. The subject allows an untrained predictor (`theta0 = theta1 = 0`). It is also left out of `make all` because it waits for keyboard input.

`make plot` still needs matplotlib installed once (`make install`). That is a pip package, not a file Make can rebuild.

## 16. AI-use disclosure

The subject requires honesty about AI.

AI was used as a tutor and pair-programmer: explaining gradient descent, drafting tests and docs, and implementing code that was then read, questioned, and kept only when it matched the subject formulas. Design choices (normalize then convert thetas, raw printed price, matplotlib isolated in `plot.py`) were reviewed and can be defended without the tool.

AI did **not** replace understanding. If an evaluator asks about any function in `src/`, the answer must come from this file and the code, not from “the assistant wrote it.”