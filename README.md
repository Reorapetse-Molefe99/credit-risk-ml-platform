# Credit Risk ML Platform

[![CI](https://github.com/Reorapetse-Molefe99/credit-risk-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Reorapetse-Molefe99/credit-risk-ml-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue) ![MLflow](https://img.shields.io/badge/MLflow-3.18-0194E2) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

End-to-end **probability-of-default (PD)** platform for consumer loan applications: from raw data to a validated,
explainable, monitored model behind an API and a business dashboard. The model lifecycle is managed in **MLflow**.

| | |
|---|---|
| **Champion model** | LightGBM pipeline · test **AUC 0.769 · Gini 0.537 · KS 0.400** · calibrated (mean PD 8.00% vs 8.07% observed) |
| **Rank ordering** | the riskiest 10% of loans contain 35% of defaults (lift 3.5x); the riskiest 20% contain 54% |
| **Data** | Home Credit Default Risk: 307,511 applications, 122 columns, 8.1% default rate |
| **Stack** | scikit-learn · LightGBM · XGBoost · Optuna · SHAP · MLflow · FastAPI · Streamlit · Docker · GitHub Actions |
| **Live demo** | Dashboard: _add link after deploying_ · API docs: _add link_ · Video: _add link_. See [docs/deployment.md](docs/deployment.md) |

## What's inside

- **Data ingestion and validation:** reproducible stratified splits, schema/range checks, synthetic data for tests
- **Feature engineering:** bureau-score aggregates, loan-structure and affordability ratios, employment sentinel
  handling, all inside the model pipeline so training and serving use the same code
- **Experiment tracking:** five algorithms (logistic regression, random forest, HGB, XGBoost, LightGBM), Optuna
  tuning and an optional calibration layer. Every run logs metrics, plots, datasets and the model to MLflow
- **Champion/challenger:** hold-out validation, promotion through registry aliases, one-step rollback
- **Scoring API:** PD → risk grade A–E → approve / refer / decline, with **reason codes** (TreeSHAP)
- **Dashboard:** portfolio overview, customer scoring, model performance with SHAP, approval-strategy and
  **collections prioritisation** simulators, drift monitoring with simulated scenarios
- **Responsible AI:** gender excluded from the model; segment fairness reported in the [model card](docs/model_card.md)

## Model comparison

Full-data runs on identical splits (train 196,806 · validation 49,202 · test 61,503). The model is **selected on
validation AUC**. Test metrics are reported only. All numbers come from tracked MLflow runs.

| Model | Val AUC | Test AUC | Gini | KS | Brier | Recall @ top 10% | Fit time |
|---|---|---|---|---|---|---|---|
| Logistic regression (benchmark) | 0.746 | 0.751 | 0.501 | 0.371 | 0.0682 | 33.3% | 8 s |
| Random forest | 0.743 | 0.747 | 0.494 | 0.366 | 0.0685 | 33.3% | 91 s |
| **LightGBM (champion v2)** | **0.763** | **0.769** | **0.537** | **0.400** | **0.0670** | **35.5%** | 17 s |
| LightGBM + sigmoid calibration | 0.763 | 0.769 | 0.537 | 0.400 | 0.0671 | 35.5% | 17 s |
| LightGBM + isotonic calibration | 0.763 | 0.768 | 0.536 | 0.397 | 0.0672 | 35.5% | 17 s |

LightGBM improves on the logistic benchmark by **+0.036 Gini**. Calibration layers (fitted on the validation split)
were tested and **not adopted**: the uncalibrated champion's mean PD is already within 0.1 pp of the observed default
rate, and neither layer improved the test Brier score.

## Quick start

```bash
git clone https://github.com/Reorapetse-Molefe99/credit-risk-ml-platform.git
cd credit-risk-ml-platform
uv sync                                         # Python 3.13 environment from uv.lock
# (or: pip install -r requirements.txt && pip install -e .)
```

The API and dashboard work **immediately** with the committed champion model. Without the dataset, the dashboard runs on
a synthetic demo portfolio.

```bash
uv run uvicorn api.main:app --port 8000          # http://localhost:8000/docs
uv run streamlit run dashboard/app.py            # http://localhost:8501
# or everything in containers:
docker compose up --build                        # MLflow :5000 · API :8000 · dashboard :8501
```

Score an application:

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 27000,
  "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2000, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.5}'
# {"probability_of_default": 0.065, "risk_grade": "C", "decision": "approve", "reason_codes": [...]}
```

## Reproducing the model

1. Download `application_train.csv` from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
   competition and save it as `data/raw/HomeLoanData.csv`.
2. Run:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000   # terminal 1

uv run python -m credit_risk.train --model logreg --sample 30000 --register     # baseline
uv run python -m credit_risk.train --model lightgbm --register                  # full data
uv run python -m credit_risk.evaluate --promote                                 # champion/challenger, export to artifacts/
uv run python -m credit_risk.monitoring                                         # drift report vs reference
uv run python -m credit_risk.monitoring --scenario recession                    # simulated drift
```

Tuning: `uv run python -m credit_risk.train --model lightgbm --tune --trials 30 --sample 60000`.
Challengers: `--model rf`, and `--calibrate sigmoid|isotonic` for a calibration layer fitted on the validation split.
The lifecycle is described in [docs/architecture.md](docs/architecture.md).

## Repository layout

```
.github/workflows/ci.yml   lint, tests, requirements check, Docker build and API smoke test
configs/model_config.yaml  all settings: data, model params, MLflow, promotion, grades, decisions, strategy, monitoring
src/credit_risk/           ingestion · validation · features · train · calibration · evaluate · predict · monitoring
api/                       FastAPI service (main.py, schemas.py)
dashboard/                 Streamlit app (app.py + 5 pages)
artifacts/                 exported champion model + metrics.json
render.yaml, .streamlit/   public deployment: Render (API) and Streamlit Community Cloud (dashboard)
notebooks/                 01 exploratory analysis · 02 feature engineering · 03 model experiments
tests/                     45 tests on synthetic data, incl. a demo-mode render of every dashboard page
docs/                      architecture · model card · data dictionary · agents guide
data/                      raw / processed / reference (git-ignored)
```

## Documentation

- [Project specification](PROJECT_SPEC.md): objectives, scope, requirements, milestones
- [Architecture](docs/architecture.md): components, design decisions, MLflow lifecycle, CI
- [Model card](docs/model_card.md): intended use, performance, fairness, limitations
- [Data dictionary](docs/data_dictionary.md): all 122 columns, splits, engineered features
- [Deployment](docs/deployment.md): public demo on Render and Streamlit Community Cloud
- [Agents guide](docs/agents_guide.md): Claude Code subagents for this project

## Development

```bash
uv run ruff check . && uv run pytest -q          # same checks as CI
```

## License

[MIT](LICENSE) © 2026 Reorapetse Molefe. The Home Credit dataset is not included and is subject to the
competition's terms.

This is a research and portfolio demonstration, not an authorised lending or collections decision system.
