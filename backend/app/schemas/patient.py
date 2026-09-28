"""PREP API Schemas"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age at discharge")
    gender: str = Field(..., description="M or F")
    insurance: str = Field(..., description="Medicare / Medicaid / Private / Other")
    prior_admissions_12mo: int = Field(..., ge=0, le=50)
    los_days: float = Field(..., ge=0.0, le=365.0, description="Length of stay in days")
    n_diagnoses: int = Field(..., ge=0, le=100)
    n_procedures: int = Field(..., ge=0, le=100)
    charlson_index: int = Field(..., ge=0, le=24)
    n_medications: int = Field(..., ge=0, le=100)
    high_risk_med: int = Field(..., ge=0, le=1, description="1 if high-risk medication present")
    emergency_adm: int = Field(..., ge=0, le=1, description="1 if emergency/urgent admission")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v.upper() not in ("M", "F"):
            raise ValueError("gender must be M or F")
        return v.upper()

    @field_validator("insurance")
    @classmethod
    def validate_insurance(cls, v):
        valid = {"Medicare", "Medicaid", "Private", "Other"}
        if v not in valid:
            raise ValueError(f"insurance must be one of {valid}")
        return v


class ContributionItem(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[float]
    direction: str


class ExplainResponse(BaseModel):
    base_value: float
    sum_contributions: float
    model_output_logit: float
    contributions: list[ContributionItem]
    explanation_method: str = "heuristic_approximation"
    explanation_label: str = "Heuristic approximation — not SHAP"
    disclaimer: str = (
        "These contributions are an explanation aid, not proof that changing a "
        "feature would change the patient's outcome."
    )


class PredictionResponse(BaseModel):
    estimated_risk: float
    risk_band: str
    above_threshold: bool
    threshold: float
    threshold_method: str
    model_version: str
    model_type: str
    validation_state: str
    timestamp: str
    disclaimer: str = (
        "PREP is a research/hackathon prototype. "
        "Not clinically validated. Not for patient-care decisions."
    )
