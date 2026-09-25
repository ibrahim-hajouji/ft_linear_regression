PYTHON ?= python3
PIP    ?= pip3

DATA      := data/data.csv
MODEL     := model.json
BONUS_DIR := bonus_results
REGRESSION := $(BONUS_DIR)/regression.png
METRICS    := $(BONUS_DIR)/precision_metrics.png
RESIDUALS  := $(BONUS_DIR)/residuals.png
REPORT     := $(BONUS_DIR)/precision.txt

TRAIN_DEPS := $(DATA) train.py src/linear_regression.py src/dataset.py src/model.py
PLOT_DEPS  := $(MODEL) $(DATA) plot.py
PREC_DEPS  := $(MODEL) $(DATA) precision.py src/linear_regression.py

.PHONY: help all install test train predict precision plot bonus clean

help:
	@echo "ft_linear_regression"
	@echo "  make test       run unit tests (mandatory, no extra packages)"
	@echo "  make train      learn thetas from data/data.csv -> model.json"
	@echo "  make predict    prompt for mileage and print a price"
	@echo "  make precision  write precision charts into bonus_results/"
	@echo "  make plot       write the fitted-line graph into bonus_results/"
	@echo "  make bonus      precision + plot (creates bonus_results/)"
	@echo "  make all        test + bonus"
	@echo "  make install    pip install matplotlib (bonus charts only)"
	@echo "  make clean      remove generated model, bonus_results, and bytecode"

all: test bonus

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -b

$(MODEL): $(TRAIN_DEPS)
	$(PYTHON) train.py

train: $(MODEL)

predict:
	$(PYTHON) predict.py

$(BONUS_DIR):
	mkdir -p $(BONUS_DIR)

$(METRICS) $(RESIDUALS) $(REPORT) &: $(PREC_DEPS) | $(BONUS_DIR)
	$(PYTHON) precision.py --output-dir $(BONUS_DIR)

$(REGRESSION): $(PLOT_DEPS) | $(BONUS_DIR)
	$(PYTHON) plot.py --output $(REGRESSION)

precision: $(METRICS) $(RESIDUALS) $(REPORT)

plot: $(REGRESSION)

bonus: precision plot

clean:
	rm -f $(MODEL) regression.png
	rm -rf $(BONUS_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
