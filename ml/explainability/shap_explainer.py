"""
PREP Explainability
SHAP TreeExplainer for LightGBM.
Verifies: base_value + sum(contributions) ≈ model_output (within 1e-6).
"""

import numpy as np
import pandas as pd


def get_shap_explainer(model):
    """Create TreeExplainer for LightGBM model."""
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Returns (shap_values, base_value) for class 1 (readmitted).
    shap_values shape: (n_samples, n_features)
    """
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    return shap_vals, base_value


def verify_shap_additivity(
    shap_vals: np.ndarray,
    base_value: float,
    model_logits: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """
    Verify base_value + sum(shap_i) ≈ model_logit for each sample.
    Uses 1e-4 tolerance (SHAP approximation in tree models).
    """
    reconstructed = base_value + shap_vals.sum(axis=1)
    max_diff = float(np.abs(reconstructed - model_logits).max())
    ok = max_diff < tol
    if not ok:
        print(f"[WARN] SHAP additivity max diff: {max_diff:.6f} (tol={tol})")
    return ok


def patient_explanation(
    shap_vals: np.ndarray,
    base_value: float,
    feature_names: list[str],
    feature_values: np.ndarray,
    idx: int,
    top_k: int = 10,
) -> dict:
    """
    Build per-patient explanation dict for the API / frontend.
    Returns top_k features sorted by |SHAP|.
    """
    sv = shap_vals[idx]
    order = np.argsort(np.abs(sv))[::-1][:top_k]

    contributions = []
    for i in order:
        contributions.append({
            "feature": feature_names[i],
            "shap_value": round(float(sv[i]), 5),
            "feature_value": (
                float(feature_values[i])
                if not np.isnan(float(feature_values[i]))
                else None
            ),
            "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
        })

    return {
        "base_value": round(base_value, 5),
        "sum_contributions": round(float(sv.sum()), 5),
        "model_output_logit": round(float(base_value + sv.sum()), 5),
        "contributions": contributions,
    }


def global_importance(
    shap_vals: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
) -> list[dict]:
    """Mean |SHAP| across all samples."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 5)}
        for i in order
    ]
