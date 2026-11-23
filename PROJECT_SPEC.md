# Project specification: Credit Risk ML Platform

## 1. Problem

Lenders must decide, at the point of application, whether to grant a loan and on what terms. Approving applicants
who later default creates credit losses. Declining good applicants loses revenue and excludes customers. The goal is a
**probability-of-default (PD) model** that ranks and quantifies applicant risk, together with the operational platform
around it: tracking, validation, deployment, explanation and monitoring.

## 2. Objectives

| # | Objective | Measure of success | Status |
|---|---|---|---|
| O1 | Predict PD for new applications | Test AUC ≥ 0.75 (Gini ≥ 0.50), monotonic deciles | ✅ AUC 0.769, Gini 0.537 |
| O2 | Well-calibrated PDs usable for pricing and expected loss | Mean PD within 0.5 pp of observed; Brier below the logistic baseline | ✅ 8.00% vs 8.07%; Brier 0.0670 vs 0.0679 |
| O3 | Reproducible, auditable model lifecycle | Every model traceable to code, params, data fingerprint and metrics in MLflow | ✅ |
| O4 | Controlled promotion | Champion/challenger gate with rollback | ✅ `@champion` / `@previous_champion` |
| O5 | Real-time and batch scoring | REST API with validation and reason codes; batch CLI | ✅ |
| O6 | Business-facing insight | Dashboard: portfolio, customer scoring, performance and SHAP, approval and collections-prioritisation strategy, monitoring | ✅ top 10% captures 35.5% of defaults (lift 3.5x) |
| O7 | Responsible use | Protected attributes excluded; fairness reported; explanations for declines | ✅ model card |
| O8 | Engineering quality | Automated tests and CI on every push; containerised deployment | ✅ |
| O9 | Public demonstration | Dashboard and API reachable at public URLs with demo data | ✅ API on Render, dashboard on Streamlit Community Cloud (see `docs/deployment.md`) |

## 3. Scope

**In scope:** application scoring for cash and revolving consumer loans using the Home Credit `application_train`
table; binary default target; experiment tracking and registry (MLflow); FastAPI service; Streamlit dashboard; Docker.

**Out of scope (for now):** the bureau, previous-application and instalment tables (future feature work);
behavioural or collections scoring; IFRS 9 lifetime PD and ECL staging; regulatory IRB calibration; authentication on the API.

## 4. Users and use cases

| User | Need | Platform feature |
|---|---|---|
| Credit risk analyst / modeller | Build, compare and document models | Notebooks, `credit_risk.train`, MLflow UI |
| Model validator | Independent evidence before approval | `credit_risk.evaluate`, model card, model-validator agent |
| Underwriter | Decision and reasons for a specific applicant | API `/predict` reason codes, dashboard page 2 |
| Credit strategy manager | Choose a cut-off balancing growth, risk and profit | Dashboard page 4 |
| Portfolio manager | Understand risk mix and expected loss | Dashboard page 1 |
| MLOps engineer | Deploy, roll back, detect drift | Registry aliases, Docker/compose, `credit_risk.monitoring`, dashboard page 5 |

## 5. Functional requirements

1. **Ingestion and validation:** load raw CSV and standardise types. Block training on schema, target, duplicate-ID or range
   errors. Create reproducible stratified 64/16/20 splits.
2. **Features:** domain features computed inside the model pipeline (identical in training and serving).
3. **Training:** logistic regression, random forest, histogram GBM, XGBoost and LightGBM. Optional Optuna tuning and an
   optional calibration layer fitted on the validation split. Every run logged to MLflow with metrics (AUC, Gini, KS,
   PR-AUC, Brier, log-loss, recall and lift at top 10%, calibration), plots, datasets and the model.
4. **Evaluation and promotion:** challenger vs champion on the hold-out set; minimum AUC uplift; score-shift warning;
   alias-based promotion with export to `artifacts/`.
5. **Scoring:** PD → risk grade (A–E) → decision (approve / refer / decline), thresholds in config. Single and batch API
   endpoints with input validation. Reason codes for individual applications.
6. **Dashboard:** five pages (portfolio overview, customer scoring, model performance with SHAP, strategy simulator with
   approval cut-offs and collections prioritisation, model monitoring). Runs on synthetic demo data when the loan-level
   data is absent, so it can be deployed publicly.
7. **Monitoring:** score and feature PSI against the development reference; simulated drift scenarios that report data
   and prediction drift only; reports to file and MLflow.

## 6. Non-functional requirements

- **Reproducibility:** pinned dependencies (`uv.lock`, `requirements.txt`), fixed seeds, data fingerprints on runs.
- **Security and privacy:** no loan-level data in git or model artifacts; skops serialisation with a type allow-list;
  container runs as a non-root user.
- **Testability:** test suite on synthetic data, runnable without the dataset or an MLflow server, in about 10 seconds.
- **Portability:** one Docker image for the API, dashboard and MLflow; `compose.yaml` for the full stack.
- **Performance:** single-application scoring well under 100 ms in-process; batch scoring of the 61k test set in seconds.

## 7. Data

Home Credit Default Risk `application_train`: 307,511 applications × 122 columns, 8.07% default rate.
Not redistributed. See `docs/data_dictionary.md` for columns, splits and engineered features.

## 8. Milestones

| Milestone | Deliverables | Status |
|---|---|---|
| M1 Foundation | Repo structure, config, environment, MLflow server, CI | ✅ |
| M2 Data | Ingestion, validation, splits, EDA notebook, data dictionary | ✅ |
| M3 Modelling | Feature engineering, five algorithms (incl. random forest), tuning, experiments notebook | ✅ |
| M4 Lifecycle | Registry, champion/challenger, export, model card | ✅ |
| M5 Serving | FastAPI, batch scoring, Docker, compose | ✅ |
| M6 Insight | Streamlit dashboard (5 pages) | ✅ |
| M7 Challengers and demo | Random forest challenger; validation-split calibration layer (tested, not adopted); recall and lift at top 10%; global SHAP and confusion matrix; collections prioritisation simulator; simulated drift scenarios; demo-mode dashboard test; Render and Streamlit Cloud configuration | ✅ |
| M8 Public launch | Deploy API (Render) and dashboard (Streamlit Cloud); README screenshots, live links and video | ⏳ deployed, screenshots in README; video to do |
| M9 Next | Bureau and previous-application features (tables in `data/raw/`); out-of-time style validation; WoE scorecard challenger; scheduled monitoring; API authentication | ⏳ |

## 9. Risks and assumptions

- The target is a proxy for default, and there is no time dimension, so validation is random rather than out-of-time.
- Only booked loans have outcomes, so there is no reject inference.
- Using age may be restricted in some jurisdictions. See the model card before any production use.
