"""Request / response models for the scoring API.

`LoanApplication` declares and validates the most important application fields. Any other raw
column from the data dictionary may be sent too (extra fields are allowed); anything not sent is
treated as missing and imputed by the model pipeline. Protected attributes (e.g. CODE_GENDER) are
accepted but ignored – they are not model inputs.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "refer", "decline"]


class LoanApplication(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "SK_ID_CURR": 100002,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 1,
                "REGION_RATING_CLIENT": 2,
                "EXT_SOURCE_1": 0.083,
                "EXT_SOURCE_2": 0.263,
                "EXT_SOURCE_3": 0.139,
            }
        },
    )

    SK_ID_CURR: int | None = Field(None, description="Application ID (echoed back, not a model input)")
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] | None = None
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None
    CNT_CHILDREN: float | None = Field(None, ge=0, le=30)
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income")
    AMT_CREDIT: float = Field(..., gt=0, description="Loan amount")
    AMT_ANNUITY: float | None = Field(None, ge=0, description="Annual loan repayment")
    AMT_GOODS_PRICE: float | None = Field(None, ge=0, description="Price of goods financed")
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    DAYS_BIRTH: float | None = Field(None, ge=-36500, le=0, description="Age in days, negative (-365 x years)")
    DAYS_EMPLOYED: float | None = Field(None, description="Days employed, negative; 365243 = not employed")
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None
    CNT_FAM_MEMBERS: float | None = Field(None, ge=0, le=40)
    REGION_RATING_CLIENT: float | None = Field(None, ge=1, le=3)
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1, description="External bureau score 1 (normalised)")
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1, description="External bureau score 2 (normalised)")
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1, description="External bureau score 3 (normalised)")


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=10_000)


class ReasonCode(BaseModel):
    feature: str
    contribution: float = Field(..., description="Log-odds contribution; > 0 increases risk")
    direction: str


class Prediction(BaseModel):
    application_id: int | None = None
    probability_of_default: float = Field(..., ge=0, le=1)
    risk_grade: str
    decision: Decision


class PredictionResponse(Prediction):
    model_version: str
    reason_codes: list[ReasonCode] | None = None


class BatchResponse(BaseModel):
    model_version: str
    n_applications: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    algorithm: str | None = None
    model_uri: str
    n_input_features: int
    test_metrics: dict[str, float] = {}
    risk_grades: dict[str, float]
    decision_thresholds: dict[str, float]
