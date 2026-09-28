"""
PREP Predictor Service
Loads model artifacts and runs prediction + explanation.
Falls back to demo scoring if no trained model is available.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "prior_admissions_12mo", "los_days", "n_diagnoses", "n_procedures",
    "charlson_index", "n_medications", "high_risk_med", "emergency_adm",
    "age", "gender_enc",
    "insurance_Medicare", "insurance_Medicaid", "insurance_Private",
]

FEATURE_LABELS = {
    "prior_admissions_12mo": "Prior Admissions (12 mo)",
    "los_days": "Length of Stay (days)",
    "n_diagnoses": "Number of Diagnoses",
    "n_procedures": "Number of Procedures",
    "charlson_index": "Charlson Comorbidity Index",
    "n_medications": "Number of Medications",
    "high_risk_med": "High-Risk Medication",
    "emergency_adm": "Emergency Admission",
    "age": "Age",
    "gender_enc": "Gender (Male)",
    "insurance_Medicare": "Insurance: Medicare",
    "insurance_Medicaid": "Insurance: Medicaid",
    "insurance_Private": "Insurance: Private",
}


def _encode(raw: dict) -> dict:
    """Encode raw patient dict into model feature dict."""
    gender = raw.get("gender", "F")
    insurance = raw.get("insurance", "Other")
    return {
        "prior_admissions_12mo": raw["prior_admissions_12mo"],
        "los_days": raw["los_days"],
        "n_diagnoses": raw["n_diagnoses"],
        "n_procedures": raw["n_procedures"],
        "charlson_index": raw["charlson_index"],
        "n_medications": raw["n_medications"],
        "high_risk_med": raw["high_risk_med"],
        "emergency_adm": raw["emergency_adm"],
        "age": raw["age"],
        "gender_enc": 1 if str(gender).upper() == "M" else 0,
        "insurance_Medicare": 1 if insurance == "Medicare" else 0,
        "insurance_Medicaid": 1 if insurance == "Medicaid" else 0,
        "insurance_Private": 1 if insurance == "Private" else 0,
    }


def _demo_risk(encoded: dict) -> float:
    """
    Deterministic demo scoring — NOT a trained model.
    SIMULATED result for hackathon demonstration only.
    """
    s = 0.15
    s += encoded["prior_admissions_12mo"] * 0.08
    s += min(encoded["los_days"], 30) * 0.005
    s += encoded["charlson_index"] * 0.015
    s += encoded["n_medications"] * 0.004
    s += encoded["high_risk_med"] * 0.06
    s += encoded["emergency_adm"] * 0.03
    s += max(0, encoded["age"] - 65) * 0.002
    return float(np.clip(s, 0.02, 0.95))


def _demo_shap(encoded: dict, risk: float) -> tuple[list[dict], float]:
    """Return heuristic demo contributions; these are not SHAP values."""
    base = 0.15
    weights = {
        "prior_admissions_12mo": 0.08,
        "los_days": 0.005,
        "charlson_index": 0.015,
        "n_medications": 0.004,
        "high_risk_med": 0.06,
        "emergency_adm": 0.03,
        "age": 0.002,
        "n_diagnoses": 0.001,
        "n_procedures": 0.001,
        "gender_enc": 0.0,
        "insurance_Medicare": 0.0,
        "insurance_Medicaid": 0.0,
        "insurance_Private": 0.0,
    }
    contribs = []
    for feat in FEATURE_ORDER:
        val = encoded.get(feat, 0)
        if feat == "age":
            shap_val = max(0, val - 65) * weights[feat]
        elif feat == "los_days":
            shap_val = min(val, 30) * weights[feat]
        else:
            shap_val = val * weights.get(feat, 0)
        contribs.append({
            "feature": FEATURE_LABELS.get(feat, feat),
            "shap_value": round(shap_val, 5),
            "feature_value": float(val),
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })
    contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return contribs, base


# Artifacts are written by ml/training/train_pipeline.py. Whichever model the
# validation split selected is the one served; LightGBM is preferred when both
# exist because tree explanations are the ones the Explainability page expects.
CANDIDATE_ARTIFACTS = ("lightgbm.pkl", "logreg.pkl")


class PREPPredictor:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._artifact = None
        self._meta_path = None
        self.model_version = "demo-v0.1"
        self.model_type = "DEMO"
        self.threshold = 0.30
        self.validation_state = "NOT YET VALIDATED"
        self._load()

    def _load(self):
        """Load a trained artifact if present; otherwise run in demo mode."""
        try:
            import joblib
            for name in CANDIDATE_ARTIFACTS:
                path = self.models_dir / name
                if not path.exists():
                    continue
                self._artifact = joblib.load(path)
                meta = self._artifact.get("meta", {})
                self.model_version = meta.get("model_version", f"{path.stem}-v1.0")
                self.model_type = self._artifact.get("model_type", path.stem)
                self.threshold = float(self._artifact.get("threshold", 0.30))
                self.validation_state = meta.get("validation_state", "PROTOTYPE")
                self._meta_path = path.with_suffix(".json")
                print(f"[PREP] Loaded {self.model_type} model: {path}")
                return
        except Exception as e:
            print(f"[PREP] Could not load model: {e}")

        print("[PREP] Running in DEMO MODE — synthetic scoring only.")

    @property
    def is_loaded(self) -> bool:
        return True  # Always available (demo fallback)

    def model_info(self) -> dict:
        meta = {}
        if self._meta_path is not None and self._meta_path.exists():
            with open(self._meta_path) as f:
                meta = json.load(f)
        evaluation = None
        report_label = None
        report_path = self.models_dir / "metrics.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as report_file:
                report = json.load(report_file)
            report_matches = (
                self._artifact is not None
                and report.get("selected_model") == self.model_type
                and report.get("source") == meta.get("source")
            )
            if report_matches:
                evaluation = report.get("evaluation")
                report_label = report.get("result_label")
        return {
            "model_version": self.model_version,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "threshold_method": meta.get("threshold_method", "Youden-J on validation set"),
            "validation_state": self.validation_state,
            "calibration_method": meta.get("calibration_method", "N/A"),
            "training_date": meta.get("training_date", "N/A"),
            "dataset": meta.get("dataset", "DEMO — synthetic data"),
            "source": meta.get("source", "demo"),
            "result_label": meta.get("result_label", "SIMULATED"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_test": meta.get("n_test"),
            "prevalence_test": meta.get("prevalence_test"),
            "evaluation": evaluation,
            "evaluation_label": report_label,
            "disclaimer": "PREP is a research/hackathon prototype. Not clinically validated.",
        }

    def _apply_calibration(self, risk: float) -> float:
        """Apply the validation-fitted recalibrator stored in the artifact.

        Falls back to the raw probability if the recalibrator is absent or
        cannot be applied, so a calibration problem never takes the API down.
        """
        recalibrator = (self._artifact or {}).get("recalibrator")
        if recalibrator is None:
            return risk
        try:
            from ml.calibration.recalibrate import apply_recalibration
            return float(apply_recalibration(recalibrator, np.array([risk]))[0])
        except Exception as e:
            print(f"[PREP] Recalibration unavailable, serving raw probability: {e}")
            return risk

    def predict(self, raw: dict) -> dict:
        encoded = _encode(raw)
        ts = datetime.now(timezone.utc).isoformat()

        if self._artifact and self.model_type != "DEMO":
            model = self._artifact["model"]
            scaler = self._artifact.get("scaler")
            X = pd.DataFrame([encoded])[FEATURE_ORDER]
            if scaler:
                X_in = scaler.transform(X)
            else:
                X_in = X.values
            risk = float(model.predict_proba(X_in)[0, 1])
            risk = self._apply_calibration(risk)
        else:
            risk = _demo_risk(encoded)

        band = "Lower" if risk < 0.20 else "Moderate" if risk < 0.40 else "Higher"

        return {
            "estimated_risk": round(risk, 4),
            "risk_band": band,
            "above_threshold": risk >= self.threshold,
            "threshold": self.threshold,
            "threshold_method": "Youden-J on validation set",
            "model_version": self.model_version,
            "model_type": self.model_type,
            "validation_state": self.validation_state,
            "timestamp": ts,
        }

    def explain(self, raw: dict) -> dict:
        encoded = _encode(raw)

        if self._artifact and self.model_type not in ("DEMO",):
            try:
                from ml.explainability.shap_explainer import (
                    compute_shap_values,
                    get_shap_explainer,
                    patient_explanation,
                )

                model = self._artifact["model"]
                X = pd.DataFrame([encoded])[FEATURE_ORDER]
                explainer = get_shap_explainer(model)
                shap_vals, base_value = compute_shap_values(explainer, X)
                explanation = patient_explanation(
                    shap_vals,
                    base_value,
                    FEATURE_ORDER,
                    X.iloc[0].to_numpy(),
                    0,
                    top_k=len(FEATURE_ORDER),
                )
                explanation["contributions"] = [
                    {
                        **item,
                        "feature": FEATURE_LABELS.get(item["feature"], item["feature"]),
                    }
                    for item in explanation["contributions"]
                ]
                explanation["explanation_method"] = "shap_tree_explainer"
                explanation["explanation_label"] = (
                    "SHAP values for the raw trained-model output"
                )
                explanation["disclaimer"] = (
                    "SHAP values explain the raw trained-model output before "
                    "recalibration. They do not prove causation or that changing "
                    "a feature would change the patient's outcome."
                )
                return explanation
            except Exception as e:
                print(f"[PREP] SHAP unavailable; returning labeled heuristic: {e}")

        risk = _demo_risk(encoded)
        contribs, base = _demo_shap(encoded, risk)
        return {
            "base_value": round(base, 5),
            "sum_contributions": round(sum(c["shap_value"] for c in contribs), 5),
            "model_output_logit": round(risk, 5),
            "contributions": contribs,
            "explanation_method": "heuristic_approximation",
            "explanation_label": (
                "Synthetic-score heuristic — not SHAP"
                if self.model_type == "DEMO"
                else "Heuristic approximation — not an explanation of the trained model"
            ),
            "disclaimer": (
                "These heuristic contributions are not SHAP values. In trained-model "
                "mode, they are based on the synthetic demo formula and do not explain "
                "the active model."
            ),
        }
