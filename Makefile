PY ?= .venv/bin/python
export PYTHONPATH := .
N ?= 30000

.PHONY: setup calibrate data train value value-demo evaluate predict llm-eval llm-audit all test api app clean

setup:            ## create the virtualenv and install dependencies
	uv venv .venv --python 3.12 || python3 -m venv .venv
	uv pip install --python $(PY) -e ".[dev]" || $(PY) -m pip install -e ".[dev]"

calibrate:        ## fit global simulator parameters to NEED medians (writes data/processed/calibration.json)
	$(PY) -m gen.calibration

data:             ## calibrate, sample N homes, simulate labels, add noise, split
	$(PY) -m gen.build_dataset --n $(N) --seed 42 --calibrate

train:            ## train baseline, conformal, feasibility, surrogate, ranking and improvement models
	$(PY) -m ml.train --rebuild-features

value:            ## hedonic value model from Land Registry joined to EPC
	$(PY) -m ml.value_model --csv data/raw/land_registry_epc.csv

value-demo:       ## value model on synthetic sales (pipeline test only, flagged DEMO ONLY)
	$(PY) -m ml.value_model --synthetic-demo

predict:          ## held-out test predictions + improvement-potential ranking (data/predictions/)
	$(PY) -m ml.predict_holdout

llm-eval:         ## held-out question set: verifier pass rate + manual audit sheet
	$(PY) -m llm.evaluate

llm-audit:        ## score the MANUAL_* columns you filled in docs/llm_audit.csv
	$(PY) -m llm.evaluate --score-audit

evaluate:         ## docs/evaluation_report.md
	$(PY) -m ml.evaluate

all: data train value-demo predict llm-eval evaluate

test:
	$(PY) -m pytest -q

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

app:
	$(PY) -m streamlit run app/dashboard.py

clean:
	rm -rf data/processed/* data/models/* .pytest_cache
