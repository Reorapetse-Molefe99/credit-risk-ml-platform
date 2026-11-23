# Credit Risk ML Platform

Probability-of-default (PD) platform on the Home Credit application data (307,511 loans, `TARGET`=1 is default,
8.1% default rate). MLflow tracks and registers models. FastAPI serves them, and a Streamlit dashboard presents
the portfolio, scoring, performance, strategy and monitoring views. Start with `PROJECT_SPEC.md` and
`docs/architecture.md`.

## Layout
- `configs/model_config.yaml`: every setting (paths, splits, excluded features, hyper-parameters, MLflow names,
  promotion rules, risk grades, decision cut-offs, strategy and monitoring thresholds)
- `src/credit_risk/`: `ingestion` (load/split/reference/synthetic), `validation`, `features` (add_features and
  preprocessing), `train` (pipelines, MLflow, tuning), `evaluate` (metrics, champion/challenger, export),
  `predict` (load, score, grade, decide, reason codes), `monitoring` (PSI)
- `api/` (FastAPI), `dashboard/` (Streamlit; `app.py` also holds the shared cached loaders), `artifacts/` (exported champion)
- `notebooks/` 01 EDA, 02 features, 03 experiments · `docs/` architecture, model card, data dictionary, agents guide
- `.claude/agents/`: project subagents (see `docs/agents_guide.md`)

## Commands (always via `uv run`)
- MLflow server (needed for train/evaluate): `uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000`
- Quick experiment: `uv run python -m credit_risk.train --model lightgbm --sample 40000`
- Full run and register: `uv run python -m credit_risk.train --model lightgbm --register`
- Tune: `uv run python -m credit_risk.train --model xgboost --tune --trials 30 --sample 60000`
- Validate the latest version: `uv run python -m credit_risk.evaluate` (`--promote` only when asked; it also exports to `artifacts/`)
- API: `uv run uvicorn api.main:app --reload --port 8000` · Dashboard: `uv run streamlit run dashboard/app.py`
- Batch score: `uv run python -m credit_risk.predict --input <csv>` · Drift: `uv run python -m credit_risk.monitoring`
- Quality gate (same as CI): `uv run ruff check . && uv run pytest -q`
- After changing dependencies: `uv lock` and then
  `uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt` (CI checks they match)

## Rules
- The test split is the final hold-out. Select features and hyper-parameters on `val_*` metrics only.
- `CODE_GENDER` is excluded on purpose (protected attribute). Never add protected attributes or obvious proxies as features.
- Never commit loan-level data (`data/` is git-ignored) or print individual customer rows. Tests use `synthetic_applications()`.
- Moving the `@champion` alias changes what production serves. Only do it when the user explicitly asks, then commit
  the updated `artifacts/` (CI builds the Docker image from it).
- New features go in `features.py::add_features` (row-wise, stateless, NaN-tolerant) and in `ENGINEERED_FEATURES`.
- Use matplotlib through `matplotlib.figure.Figure`, not pyplot, in `src/`. Forcing a backend breaks notebooks.
- Use `--sample` for exploration. Use full-data runs for anything that gets registered.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow's tracing-skill log hint.
