"""
PREP Drift Monitoring
KL divergence for feature drift, 2σ alert for prediction drift.
"""

import numpy as np
import pandas as pd

KL_THRESHOLD = 0.05
PRED_DRIFT_SIGMA = 2.0

MONITORED_FEATURES = [
    "prior_admissions_12mo", "los_days", "n_diagnoses",
    "n_procedures", "charlson_index", "n_medications",
    "age", "high_risk_med", "emergency_adm",
]


def kl_divergence_numeric(
    ref: np.ndarray,
    cur: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-10,
) -> float:
    """KL divergence using reference decile bins."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    bin_edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    p, _ = np.histogram(ref, bins=bin_edges, density=True)
    q, _ = np.histogram(cur, bins=bin_edges, density=True)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def kl_divergence_categorical(
    ref: np.ndarray,
    cur: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """KL divergence for categorical features using category proportions."""
    categories = np.union1d(np.unique(ref), np.unique(cur))
    p = np.array([np.mean(ref == c) + eps for c in categories])
    q = np.array([np.mean(cur == c) + eps for c in categories])
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict:
    """Compute KL divergence for each monitored feature."""
    if numeric_features is None:
        numeric_features = [f for f in MONITORED_FEATURES if f in ref_df.columns]
    if categorical_features is None:
        categorical_features = []

    results = {}
    for feat in numeric_features:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue
        kl = kl_divergence_numeric(
            ref_df[feat].values.astype(float),
            cur_df[feat].values.astype(float),
        )
        results[feat] = {
            "kl_divergence": round(kl, 5),
            "alert": kl > KL_THRESHOLD,
            "type": "numeric",
        }

    for feat in categorical_features:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue
        kl = kl_divergence_categorical(
            ref_df[feat].astype(str).values,
            cur_df[feat].astype(str).values,
        )
        results[feat] = {
            "kl_divergence": round(kl, 5),
            "alert": kl > KL_THRESHOLD,
            "type": "categorical",
        }

    return results


def compute_prediction_drift(
    ref_predictions: np.ndarray,
    cur_predictions: np.ndarray,
) -> dict:
    """Alert if current mean predicted risk deviates > 2σ from reference."""
    ref_mean = float(np.mean(ref_predictions))
    ref_std = float(np.std(ref_predictions))
    cur_mean = float(np.mean(cur_predictions))

    if ref_std < 1e-8:
        return {
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "ref_std": round(ref_std, 4),
            "z_score": 0.0,
            "alert": False,
        }

    z = abs(cur_mean - ref_mean) / ref_std
    return {
        "ref_mean": round(ref_mean, 4),
        "cur_mean": round(cur_mean, 4),
        "ref_std": round(ref_std, 4),
        "z_score": round(float(z), 3),
        "alert": z > PRED_DRIFT_SIGMA,
    }


def drift_summary(feature_drift: dict, prediction_drift: dict) -> dict:
    """Aggregate drift status for dashboard."""
    feature_alerts = [f for f, v in feature_drift.items() if v["alert"]]
    pred_alert = prediction_drift.get("alert", False)
    any_alert = bool(feature_alerts) or pred_alert

    return {
        "status": "DRIFT DETECTED" if any_alert else "STABLE",
        "alert": any_alert,
        "feature_alerts": feature_alerts,
        "prediction_drift_alert": pred_alert,
        "n_features_monitored": len(feature_drift),
        "n_features_drifted": len(feature_alerts),
        "kl_threshold": KL_THRESHOLD,
        "prediction_sigma_threshold": PRED_DRIFT_SIGMA,
    }
