# Model card: `credit-risk-pd` v2 (champion)

| | |
|---|---|
| **Model** | LightGBM gradient-boosted trees in a scikit-learn pipeline (feature engineering → imputation and one-hot encoding → classifier) |
| **Version** | Registry `credit-risk-pd` v2, alias `@champion`, exported to `artifacts/model` |
| **Owner** | Reorapetse Molefe |
| **Date** | November 2026 |
| **Output** | Probability of default (PD) in [0, 1], mapped to risk grades A–E and an approve / refer / decline recommendation |
| **Code** | `src/credit_risk/` · training run and lineage in MLflow (run ID in `artifacts/metrics.json`) |

## Intended use

- **Primary use:** rank-order and estimate the default risk of **new consumer loan applications** at origination,
  to support credit decisions, risk-based pricing and portfolio expected-loss estimates.
- **Users:** credit risk analysts, underwriters (refer band), and portfolio and strategy teams.
- **Decision support, not decision automation:** the `refer` band (PD 10–20%) goes to a human underwriter.
  Declines must be explainable to the applicant, which is what the reason codes from `/predict` are for.

**Out of scope:** behavioural or collections scoring of existing accounts; regulatory capital (IRB) PD estimation
(this model is not through-the-cycle calibrated); populations or products materially different from the development
data (e.g. mortgages, SME lending, other countries) without re-validation.

## Training data

- Home Credit Default Risk `application_train` (307,511 applications, 122 columns, 8.07% default rate);
  see `docs/data_dictionary.md`.
- Stratified split: train 196,806 (64%), validation 49,202 (16%), test 61,503 (20%).
- **Target:** payment difficulties on early instalments. This is a proxy for default, not a regulatory
  90-days-past-due definition.
- Single application snapshot with no time dimension, so the split is random rather than out-of-time.
  Performance on future cohorts may be lower.

## Features

- 119 raw application inputs and 13 engineered features (`docs/data_dictionary.md#engineered-features`).
- **Excluded:** `CODE_GENDER` (protected attribute). The application ID is also excluded.
- Strongest drivers: the external bureau-score aggregates (`EXT_SOURCE_MEAN/MIN/MAX`), then age, employment
  tenure, loan structure (`GOODS_TO_CREDIT`, `ANNUITY_TO_CREDIT`), education and region rating.
- Hyper-parameters: `configs/model_config.yaml → model.params.lightgbm` (800 trees, learning rate 0.03, 31 leaves).

## Performance (hold-out test set, n = 61,503)

| Metric | Value | Validation |
|---|---|---|
| ROC AUC | **0.769** | 0.763 |
| Gini | **0.537** | 0.527 |
| KS | 0.400 | |
| PR AUC | 0.259 | (base rate 0.081) |
| Brier score | 0.0670 | |
| Mean predicted PD vs observed default rate | 8.00% vs 8.07% | |
| Score PSI (validation vs test) | 0.0005 | |

- **Rank ordering:** default rate falls in every decile from riskiest to safest. The riskiest 20% of
  applicants contain **54% of all defaults**.
- **Calibration:** good overall (mean PD within 0.1 pp of the observed rate). The decile-level reliability curve is in
  `artifacts/metrics.json` and on dashboard page 3.
- **Baseline:** logistic regression (v1, trained on a 30k sample) test AUC 0.745. The champion improves on it by
  +0.024 AUC (+0.047 Gini).
- **Challengers on the full data** (same splits, selected on validation AUC):

  | Model | Val AUC | Test AUC | Test Gini | Test Brier |
  |---|---|---|---|---|
  | Logistic regression | 0.746 | 0.751 | 0.501 | 0.0682 |
  | Random forest | 0.743 | 0.747 | 0.494 | 0.0685 |
  | **LightGBM (this model)** | **0.763** | **0.769** | **0.537** | **0.0670** |
  | LightGBM + sigmoid calibration | 0.763 | 0.769 | 0.537 | 0.0671 |
  | LightGBM + isotonic calibration | 0.763 | 0.768 | 0.536 | 0.0672 |

  The calibration layers were fitted on the validation split and evaluated on test. Neither improved the Brier
  score, so the champion stays uncalibrated.
- **Collections prioritisation:** working the riskiest 10% of loans captures **35.5%** of defaults (lift 3.5x
  over random selection). This is default capture, not defaults prevented. A PD model can't show that contacting a
  customer changes the outcome.

### Decision outcomes on the test set (approve < 10% ≤ refer < 20% ≤ decline)

| Decision | Share | Observed default rate |
|---|---|---|
| approve | 74.8% | 4.24% |
| refer | 16.8% | 13.98% |
| decline | 8.4% | 30.31% |

## Segment performance and fairness

| Segment | n | Observed DR | Mean PD | AUC | Approval rate | Bad rate (approved) |
|---|---|---|---|---|---|---|
| Female | 40,561 | 6.99% | 7.37% | 0.761 | 77.9% | 3.94% |
| Male | 20,940 | 10.17% | 9.29% | 0.770 | 68.8% | 4.89% |
| Age < 30 | 8,905 | 11.30% | 11.54% | 0.748 | 56.8% | 4.89% |
| Age 30–44 | 24,744 | 8.83% | 8.81% | 0.776 | 71.8% | 4.33% |
| Age 45–59 | 20,666 | 6.85% | 6.68% | 0.756 | 81.0% | 4.13% |
| Age 60+ | 7,188 | 4.97% | 4.82% | 0.732 | 89.7% | 3.77% |
| Cash loans | 55,672 | 8.35% | 8.28% | 0.768 | 73.7% | 4.29% |
| Revolving loans | 5,831 | 5.42% | 5.60% | 0.750 | 85.6% | 3.83% |

Observations:

1. **Gender:** gender is excluded, but approval rates differ (77.9% F vs 68.8% M), which tracks the
   observed default-rate gap (7.0% vs 10.2%). Because gender is excluded, the model **under-predicts PD for
   men (9.3% vs 10.2%) and over-predicts for women (7.4% vs 7.0%)**. Discrimination (AUC) is similar for both.
   Whether this is acceptable is a policy decision. Monitor it every release.
2. **Age** is used directly (`DAYS_BIRTH`, `AGE_YEARS`). Applicants under 30 are approved far less often (57% vs 90% for 60+),
   in line with their risk, and calibration is good within every age band. Age may be a protected or restricted
   characteristic in some jurisdictions. Confirm it is permitted before production use, or retrain without it.
3. Discrimination is lowest for applicants aged 60+ (AUC 0.73), who are mostly pensioners with little employment information.

## Limitations and risks

- **No out-of-time validation**, since the data has no application dates. Expect some performance decay on new cohorts.
- **Proxy target:** "payment difficulties" is not a default event as regulators define it.
- **Missing bureau data:** `EXT_SOURCE_1` is 56% missing. Thin-file applicants get less reliable scores.
- **Feedback loop / reject inference:** only approved, booked loans have outcomes. The model has not seen
  previously rejected applicants.
- **Self-reported income** is a weak predictor here. Affordability should also be checked by separate policy rules.

## Monitoring and maintenance

- Score PSI and feature PSI against `data/reference/` for every new cohort (`credit_risk.monitoring`, dashboard page 5).
  Warning at PSI > 0.10, investigate or redevelop at > 0.25.
- Track AUC, calibration and segment metrics as outcomes mature, and re-run the fairness table above for each candidate.
- Every candidate must go through `credit_risk.evaluate` (champion/challenger) before promotion. The previous champion is
  kept as `@previous_champion` for rollback.

## How to reproduce

```bash
uv sync
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000
uv run python -m credit_risk.train --model lightgbm --register --rebuild-splits
uv run python -m credit_risk.evaluate --promote
```
