# Data dictionary

**Source:** Home Credit Default Risk, `application_train` table (Kaggle competition data). One row per loan application.
The file is **not redistributed** in this repository. Place it at `data/raw/HomeLoanData.csv`.

- 307,511 applications, 122 columns, default rate 8.07%
- Statistics below are computed on the **training split** (196,806 rows)
- Model inputs: every column marked `feature` (119). Integer columns are cast to float so the served model accepts missing values.
- `DAYS_*` columns count days relative to the application date (negative = in the past).

## Splits

| split | share | rows | used for |
|---|---|---|---|
| train | 64% | 196,806 | model fitting |
| val | 16% | 49,202 | model / feature / hyper-parameter selection |
| test | 20% | 61,503 | final hold-out: champion/challenger and reported metrics |

Stratified on `TARGET`, `random_state = 42` (`configs/model_config.yaml`).

## Identifier & target

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `SK_ID_CURR` | numeric | 0.0% | 1e+05 … 4.563e+05 | id | Application (loan) ID |
| `TARGET` | numeric | 0.0% | 0 … 1 | target | 1 = client had payment difficulties (late payment > X days on at least one of the first Y instalments); 0 = all other cases |

## Applicant

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `CODE_GENDER` | categorical | 0.0% | 3 categories | excluded | Gender of the client (**excluded from modelling**) |
| `FLAG_OWN_CAR` | categorical | 0.0% | 2 categories | feature | Client owns a car |
| `FLAG_OWN_REALTY` | categorical | 0.0% | 2 categories | feature | Client owns a house or flat |
| `CNT_CHILDREN` | numeric | 0.0% | 0 … 19 | feature | Number of children |
| `NAME_TYPE_SUITE` | categorical | 0.4% | 7 categories | feature | Who accompanied the client when applying |
| `NAME_INCOME_TYPE` | categorical | 0.0% | 8 categories | feature | Income type (working, pensioner, state servant, ...) |
| `NAME_EDUCATION_TYPE` | categorical | 0.0% | 5 categories | feature | Highest education level |
| `NAME_FAMILY_STATUS` | categorical | 0.0% | 6 categories | feature | Family status |
| `NAME_HOUSING_TYPE` | categorical | 0.0% | 6 categories | feature | Housing situation (own, rented, with parents, ...) |
| `DAYS_BIRTH` | numeric | 0.0% | -2.523e+04 … -7,489 | feature | Age in days at application (negative) |
| `DAYS_EMPLOYED` | numeric | 0.0% | -1.791e+04 … 3.652e+05 | feature | Days in current employment at application (negative; 365243 = not employed) |
| `DAYS_REGISTRATION` | numeric | 0.0% | -2.467e+04 … 0 | feature | Days since the client last changed their registration (negative) |
| `DAYS_ID_PUBLISH` | numeric | 0.0% | -7,197 … 0 | feature | Days since the client last changed the ID document used to apply (negative) |
| `OWN_CAR_AGE` | numeric | 66.1% | 0 … 91 | feature | Age of the client's car |
| `FLAG_MOBIL` | numeric | 0.0% | 0 … 1 | feature | Provided a mobile phone |
| `FLAG_EMP_PHONE` | numeric | 0.0% | 0 … 1 | feature | Provided a work phone |
| `FLAG_WORK_PHONE` | numeric | 0.0% | 0 … 1 | feature | Provided a home phone |
| `FLAG_CONT_MOBILE` | numeric | 0.0% | 0 … 1 | feature | Mobile phone was reachable |
| `FLAG_PHONE` | numeric | 0.0% | 0 … 1 | feature | Provided a home phone |
| `FLAG_EMAIL` | numeric | 0.0% | 0 … 1 | feature | Provided an email address |
| `OCCUPATION_TYPE` | categorical | 31.3% | 18 categories | feature | Occupation |
| `CNT_FAM_MEMBERS` | numeric | 0.0% | 1 … 20 | feature | Number of family members |
| `ORGANIZATION_TYPE` | categorical | 0.0% | 58 categories | feature | Type of organisation where the client works |
| `DAYS_LAST_PHONE_CHANGE` | numeric | 0.0% | -4,292 … 0 | feature | Days since the client last changed phone (negative) |

## Loan & application

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `NAME_CONTRACT_TYPE` | categorical | 0.0% | 2 categories | feature | Cash loan or revolving loan |
| `AMT_INCOME_TOTAL` | numeric | 0.0% | 2.565e+04 … 1.8e+07 | feature | Annual income of the client |
| `AMT_CREDIT` | numeric | 0.0% | 4.5e+04 … 4.05e+06 | feature | Credit amount of the loan |
| `AMT_ANNUITY` | numeric | 0.0% | 1,616 … 2.25e+05 | feature | Loan annuity (annual repayment) |
| `AMT_GOODS_PRICE` | numeric | 0.1% | 4.05e+04 … 4.05e+06 | feature | Price of the goods the loan is for (consumer loans) |
| `WEEKDAY_APPR_PROCESS_START` | categorical | 0.0% | 7 categories | feature | Weekday the application was started |
| `HOUR_APPR_PROCESS_START` | numeric | 0.0% | 0 … 23 | feature | Hour the application was started |

## Address & region

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `REGION_POPULATION_RELATIVE` | numeric | 0.0% | 0.00029 … 0.07251 | feature | Normalised population of the client's region (higher = more populated) |
| `REGION_RATING_CLIENT` | numeric | 0.0% | 1 … 3 | feature | Lender's rating of the region where the client lives (1–3) |
| `REGION_RATING_CLIENT_W_CITY` | numeric | 0.0% | 1 … 3 | feature | Lender's rating of the region, taking the city into account (1–3) |
| `REG_REGION_NOT_LIVE_REGION` | numeric | 0.0% | 0 … 1 | feature | Permanent address region ≠ contact address region |
| `REG_REGION_NOT_WORK_REGION` | numeric | 0.0% | 0 … 1 | feature | Permanent address region ≠ work address region |
| `LIVE_REGION_NOT_WORK_REGION` | numeric | 0.0% | 0 … 1 | feature | Contact address region ≠ work address region |
| `REG_CITY_NOT_LIVE_CITY` | numeric | 0.0% | 0 … 1 | feature | Permanent address city ≠ contact address city |
| `REG_CITY_NOT_WORK_CITY` | numeric | 0.0% | 0 … 1 | feature | Permanent address city ≠ work address city |
| `LIVE_CITY_NOT_WORK_CITY` | numeric | 0.0% | 0 … 1 | feature | Contact address city ≠ work address city |

## Bureau & external

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `EXT_SOURCE_1` | numeric | 56.3% | 0.01457 … 0.9516 | feature | Normalised score from external data source 1 (credit bureau) |
| `EXT_SOURCE_2` | numeric | 0.2% | 8.17e-08 … 0.855 | feature | Normalised score from external data source 2 |
| `EXT_SOURCE_3` | numeric | 19.8% | 0.0005273 … 0.896 | feature | Normalised score from external data source 3 |
| `OBS_30_CNT_SOCIAL_CIRCLE` | numeric | 0.3% | 0 … 348 | feature | Observations of the client's social surroundings with 30 DPD (days past due) |
| `DEF_30_CNT_SOCIAL_CIRCLE` | numeric | 0.3% | 0 … 34 | feature | Number of the client's social surroundings that defaulted at 30 DPD |
| `OBS_60_CNT_SOCIAL_CIRCLE` | numeric | 0.3% | 0 … 344 | feature | Observations of the client's social surroundings with 60 DPD |
| `DEF_60_CNT_SOCIAL_CIRCLE` | numeric | 0.3% | 0 … 24 | feature | Number of the client's social surroundings that defaulted at 60 DPD |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | numeric | 13.5% | 0 … 3 | feature | Credit-bureau enquiries in the hour before application |
| `AMT_REQ_CREDIT_BUREAU_DAY` | numeric | 13.5% | 0 … 9 | feature | Credit-bureau enquiries in the day before application (excluding the last hour) |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | numeric | 13.5% | 0 … 8 | feature | Credit-bureau enquiries in the week before application (excluding the last day) |
| `AMT_REQ_CREDIT_BUREAU_MON` | numeric | 13.5% | 0 … 27 | feature | Credit-bureau enquiries in the month before application (excluding the last week) |
| `AMT_REQ_CREDIT_BUREAU_QRT` | numeric | 13.5% | 0 … 19 | feature | Credit-bureau enquiries in the 3 months before application (excluding the last month) |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | numeric | 13.5% | 0 … 25 | feature | Credit-bureau enquiries in the year before application (excluding the last 3 months) |

## Building / housing

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `APARTMENTS_AVG` | numeric | 50.7% | 0 … 1 | feature | Building info, normalised (average): apartments |
| `BASEMENTAREA_AVG` | numeric | 58.4% | 0 … 1 | feature | Building info, normalised (average): basementarea |
| `YEARS_BEGINEXPLUATATION_AVG` | numeric | 48.7% | 0 … 1 | feature | Building info, normalised (average): years beginexpluatation |
| `YEARS_BUILD_AVG` | numeric | 66.5% | 0 … 1 | feature | Building info, normalised (average): years build |
| `COMMONAREA_AVG` | numeric | 69.8% | 0 … 1 | feature | Building info, normalised (average): commonarea |
| `ELEVATORS_AVG` | numeric | 53.2% | 0 … 1 | feature | Building info, normalised (average): elevators |
| `ENTRANCES_AVG` | numeric | 50.3% | 0 … 1 | feature | Building info, normalised (average): entrances |
| `FLOORSMAX_AVG` | numeric | 49.7% | 0 … 1 | feature | Building info, normalised (average): floorsmax |
| `FLOORSMIN_AVG` | numeric | 67.9% | 0 … 1 | feature | Building info, normalised (average): floorsmin |
| `LANDAREA_AVG` | numeric | 59.3% | 0 … 1 | feature | Building info, normalised (average): landarea |
| `LIVINGAPARTMENTS_AVG` | numeric | 68.3% | 0 … 1 | feature | Building info, normalised (average): livingapartments |
| `LIVINGAREA_AVG` | numeric | 50.2% | 0 … 1 | feature | Building info, normalised (average): livingarea |
| `NONLIVINGAPARTMENTS_AVG` | numeric | 69.4% | 0 … 1 | feature | Building info, normalised (average): nonlivingapartments |
| `NONLIVINGAREA_AVG` | numeric | 55.1% | 0 … 1 | feature | Building info, normalised (average): nonlivingarea |
| `APARTMENTS_MODE` | numeric | 50.7% | 0 … 1 | feature | Building info, normalised (mode): apartments |
| `BASEMENTAREA_MODE` | numeric | 58.4% | 0 … 1 | feature | Building info, normalised (mode): basementarea |
| `YEARS_BEGINEXPLUATATION_MODE` | numeric | 48.7% | 0 … 1 | feature | Building info, normalised (mode): years beginexpluatation |
| `YEARS_BUILD_MODE` | numeric | 66.5% | 0 … 1 | feature | Building info, normalised (mode): years build |
| `COMMONAREA_MODE` | numeric | 69.8% | 0 … 1 | feature | Building info, normalised (mode): commonarea |
| `ELEVATORS_MODE` | numeric | 53.2% | 0 … 1 | feature | Building info, normalised (mode): elevators |
| `ENTRANCES_MODE` | numeric | 50.3% | 0 … 1 | feature | Building info, normalised (mode): entrances |
| `FLOORSMAX_MODE` | numeric | 49.7% | 0 … 1 | feature | Building info, normalised (mode): floorsmax |
| `FLOORSMIN_MODE` | numeric | 67.9% | 0 … 1 | feature | Building info, normalised (mode): floorsmin |
| `LANDAREA_MODE` | numeric | 59.3% | 0 … 1 | feature | Building info, normalised (mode): landarea |
| `LIVINGAPARTMENTS_MODE` | numeric | 68.3% | 0 … 1 | feature | Building info, normalised (mode): livingapartments |
| `LIVINGAREA_MODE` | numeric | 50.2% | 0 … 1 | feature | Building info, normalised (mode): livingarea |
| `NONLIVINGAPARTMENTS_MODE` | numeric | 69.4% | 0 … 1 | feature | Building info, normalised (mode): nonlivingapartments |
| `NONLIVINGAREA_MODE` | numeric | 55.1% | 0 … 1 | feature | Building info, normalised (mode): nonlivingarea |
| `APARTMENTS_MEDI` | numeric | 50.7% | 0 … 1 | feature | Building info, normalised (median): apartments |
| `BASEMENTAREA_MEDI` | numeric | 58.4% | 0 … 1 | feature | Building info, normalised (median): basementarea |
| `YEARS_BEGINEXPLUATATION_MEDI` | numeric | 48.7% | 0 … 1 | feature | Building info, normalised (median): years beginexpluatation |
| `YEARS_BUILD_MEDI` | numeric | 66.5% | 0 … 1 | feature | Building info, normalised (median): years build |
| `COMMONAREA_MEDI` | numeric | 69.8% | 0 … 1 | feature | Building info, normalised (median): commonarea |
| `ELEVATORS_MEDI` | numeric | 53.2% | 0 … 1 | feature | Building info, normalised (median): elevators |
| `ENTRANCES_MEDI` | numeric | 50.3% | 0 … 1 | feature | Building info, normalised (median): entrances |
| `FLOORSMAX_MEDI` | numeric | 49.7% | 0 … 1 | feature | Building info, normalised (median): floorsmax |
| `FLOORSMIN_MEDI` | numeric | 67.9% | 0 … 1 | feature | Building info, normalised (median): floorsmin |
| `LANDAREA_MEDI` | numeric | 59.3% | 0 … 1 | feature | Building info, normalised (median): landarea |
| `LIVINGAPARTMENTS_MEDI` | numeric | 68.3% | 0 … 1 | feature | Building info, normalised (median): livingapartments |
| `LIVINGAREA_MEDI` | numeric | 50.2% | 0 … 1 | feature | Building info, normalised (median): livingarea |
| `NONLIVINGAPARTMENTS_MEDI` | numeric | 69.4% | 0 … 1 | feature | Building info, normalised (median): nonlivingapartments |
| `NONLIVINGAREA_MEDI` | numeric | 55.1% | 0 … 1 | feature | Building info, normalised (median): nonlivingarea |
| `FONDKAPREMONT_MODE` | categorical | 68.4% | 4 categories | feature | Building info: capital-repairs fund type |
| `HOUSETYPE_MODE` | categorical | 50.1% | 3 categories | feature | Building info: house type |
| `TOTALAREA_MODE` | numeric | 48.2% | 0 … 1 | feature | Building info: total area |
| `WALLSMATERIAL_MODE` | categorical | 50.8% | 7 categories | feature | Building info: wall material |
| `EMERGENCYSTATE_MODE` | categorical | 47.4% | 2 categories | feature | Building info: emergency state |

## Documents

| column | type | missing | range / levels | use | description |
|---|---|---|---|---|---|
| `FLAG_DOCUMENT_2` | numeric | 0.0% | 0 … 1 | feature | Client provided document 2 |
| `FLAG_DOCUMENT_3` | numeric | 0.0% | 0 … 1 | feature | Client provided document 3 |
| `FLAG_DOCUMENT_4` | numeric | 0.0% | 0 … 1 | feature | Client provided document 4 |
| `FLAG_DOCUMENT_5` | numeric | 0.0% | 0 … 1 | feature | Client provided document 5 |
| `FLAG_DOCUMENT_6` | numeric | 0.0% | 0 … 1 | feature | Client provided document 6 |
| `FLAG_DOCUMENT_7` | numeric | 0.0% | 0 … 1 | feature | Client provided document 7 |
| `FLAG_DOCUMENT_8` | numeric | 0.0% | 0 … 1 | feature | Client provided document 8 |
| `FLAG_DOCUMENT_9` | numeric | 0.0% | 0 … 1 | feature | Client provided document 9 |
| `FLAG_DOCUMENT_10` | numeric | 0.0% | 0 … 1 | feature | Client provided document 10 |
| `FLAG_DOCUMENT_11` | numeric | 0.0% | 0 … 1 | feature | Client provided document 11 |
| `FLAG_DOCUMENT_12` | numeric | 0.0% | 0 … 1 | feature | Client provided document 12 |
| `FLAG_DOCUMENT_13` | numeric | 0.0% | 0 … 1 | feature | Client provided document 13 |
| `FLAG_DOCUMENT_14` | numeric | 0.0% | 0 … 1 | feature | Client provided document 14 |
| `FLAG_DOCUMENT_15` | numeric | 0.0% | 0 … 1 | feature | Client provided document 15 |
| `FLAG_DOCUMENT_16` | numeric | 0.0% | 0 … 1 | feature | Client provided document 16 |
| `FLAG_DOCUMENT_17` | numeric | 0.0% | 0 … 1 | feature | Client provided document 17 |
| `FLAG_DOCUMENT_18` | numeric | 0.0% | 0 … 1 | feature | Client provided document 18 |
| `FLAG_DOCUMENT_19` | numeric | 0.0% | 0 … 1 | feature | Client provided document 19 |
| `FLAG_DOCUMENT_20` | numeric | 0.0% | 0 … 1 | feature | Client provided document 20 |
| `FLAG_DOCUMENT_21` | numeric | 0.0% | 0 … 1 | feature | Client provided document 21 |

## Engineered features

Created by `credit_risk.features.add_features` inside the model pipeline (so they are also computed at scoring time).

| feature | definition |
|---|---|
| `DAYS_EMPLOYED_ANOMALY` | 1 if DAYS_EMPLOYED carried the 365243 'not employed' sentinel |
| `EMPLOYED_YEARS` | Years in current employment |
| `AGE_YEARS` | Applicant age in years |
| `EMPLOYED_TO_AGE` | Share of life spent in current employment |
| `CREDIT_TO_INCOME` | Loan amount / annual income |
| `ANNUITY_TO_INCOME` | Annual repayment / annual income (debt-service burden) |
| `ANNUITY_TO_CREDIT` | Annual repayment / loan amount (inverse ~ term in years) |
| `GOODS_TO_CREDIT` | Goods price / loan amount (loan-to-value proxy) |
| `INCOME_PER_PERSON` | Income / family members |
| `EXT_SOURCE_MEAN` | Mean of available external bureau scores |
| `EXT_SOURCE_MIN` | Minimum external bureau score |
| `EXT_SOURCE_MAX` | Maximum external bureau score |
| `EXT_SOURCE_NULLS` | Number of missing external bureau scores (thin-file indicator) |

## Model outputs

| field | description |
|---|---|
| `probability_of_default` | Predicted PD, 0–1 |
| `risk_grade` | A–E from PD bands: A < 2%, B < 5%, C < 10%, D < 20%, E < 100% |
| `decision` | approve (PD < 10%), refer, decline (PD ≥ 20%) |
| `reason_codes` | Top drivers of an individual PD (log-odds contributions; API `/predict`) |
