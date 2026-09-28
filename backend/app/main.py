"""
PREP FastAPI Backend
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).parent.parent.parent  # PREP/ project root
BACKEND_DIR = Path(__file__).parent.parent
DEMO_DIR = BASE_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"

# The calibration/hand-off code in ml/ is the single source of truth, so the API
# reuses it rather than re-implementing the arithmetic. Required because
# uvicorn is normally started from backend/, where the repo root is not on
# sys.path.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.patient import PatientInput, PredictionResponse, ExplainResponse  # noqa: E402
from app.services.predictor import PREPPredictor  # noqa: E402
from app.monitoring.drift import (
    compute_feature_drift, compute_prediction_drift, drift_summary,
)
from ml.fairness.audit import fairness_summary

BASE_DIR = Path(__file__).parent.parent.parent  # PREP/ project root
DEMO_DIR = BASE_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="PREP API",
    description="Predict Readmission Estimation of Patient — Research Prototype",
    version="0.1.0",
)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load predictor at startup ─────────────────────────────────────────────────
predictor = PREPPredictor(MODELS_DIR)

# ── In-memory prediction log (no PII) ────────────────────────────────────────
_prediction_log: list[dict] = []


def _hash_input(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "model_version": predictor.model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "PREP is a research/hackathon prototype. Not clinically validated.",
    }


@app.get("/v1/model")
def model_info():
    return predictor.model_info()


@app.get("/v1/fairness")
def fairness_report():
    """Return the fairness audit from the latest available training report."""
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {
            "available": False,
            "dimensions": {},
            "summary": fairness_summary({}),
            "dataset": None,
            "result_label": None,
            "simulated": False,
            "message": "No saved training fairness report is available.",
        }

    with open(path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    dimensions = report.get("fairness_detail", {})
    if not isinstance(dimensions, dict):
        raise HTTPException(500, "Saved fairness report has an invalid structure.")

    return {
        "available": bool(dimensions),
        "dimensions": dimensions,
        "summary": fairness_summary(dimensions),
        "dataset": report.get("dataset"),
        "result_label": report.get("result_label"),
        "simulated": report.get("source") == "demo"
        or report.get("result_label") == "SIMULATED",
        "message": None if dimensions else "Saved report contains no fairness dimensions.",
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if not predictor.is_loaded:
        raise HTTPException(503, "Model not loaded. Run the ML pipeline first.")

    result = predictor.predict(patient.model_dump())

    # Log prediction (no PII)
    _prediction_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": predictor.model_version,
        "hashed_input": _hash_input(patient.model_dump()),
        "estimated_risk": result["estimated_risk"],
        "above_threshold": result["above_threshold"],
    })

    return result


@app.post("/v1/explain", response_model=ExplainResponse)
def explain(patient: PatientInput):
    if not predictor.is_loaded:
        raise HTTPException(503, "Model not loaded. Run the ML pipeline first.")
    return predictor.explain(patient.model_dump())


@app.get("/metrics")
def metrics():
    """Return prediction log summary (no PII)."""
    if not _prediction_log:
        return {"n_predictions": 0, "log": []}
    risks = [e["estimated_risk"] for e in _prediction_log]
    return {
        "n_predictions": len(_prediction_log),
        "mean_estimated_risk": round(float(np.mean(risks)), 4),
        "pct_above_threshold": round(
            float(np.mean([e["above_threshold"] for e in _prediction_log])), 4
        ),
        "recent": _prediction_log[-10:],
    }


@app.get("/v1/demo/patients")
def demo_patients():
    """Return synthetic demo patients."""
    path = DEMO_DIR / "demo_patients.json"
    if not path.exists():
        raise HTTPException(404, "Demo data not found. Run demo/generate_demo_data.py")
    with open(path) as f:
        return json.load(f)


@app.get("/v1/demo/cohort")
def demo_cohort():
    path = DEMO_DIR / "synthetic_cohort.csv"
    if not path.exists():
        raise HTTPException(404, "Demo cohort not found. Run demo/generate_demo_data.py")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.post("/v1/drift/check")
def drift_check(simulate: bool = False):
    """
    Check for data drift.
    simulate=true introduces a synthetic distribution shift for demonstration.
    """
    path = DEMO_DIR / "drift_demo.csv"
    if not path.exists():
        raise HTTPException(404, "Drift demo data not found.")

    df = pd.read_csv(path)
    ref = df[df["period"] == "reference"]
    cur = df[df["period"] == ("drifted" if simulate else "reference")]

    feature_drift = compute_feature_drift(ref, cur)
    pred_drift = compute_prediction_drift(
        ref["estimated_risk"].values,
        cur["estimated_risk"].values,
    )
    summary = drift_summary(feature_drift, pred_drift)

    return {
        "simulated": simulate,
        "summary": summary,
        "feature_drift": feature_drift,
        "prediction_drift": pred_drift,
        "disclaimer": "SIMULATED drift demonstration — not real clinical data.",
    }
