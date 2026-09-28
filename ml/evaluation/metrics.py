"""
PREP Evaluation
Computes discrimination, calibration, threshold metrics, and net benefit.
Includes base-rate Brier comparison as required by PREP specification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() > 0:
            scores.append(roc_auc_score(y_true[idx], y_proba[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pct_flagged = y_pred.mean()
    alerts_per_1000 = pct_flagged * 1000
    return {
        "sensitivity": round(float(sensitivity), 4),
        "ppv": round(float(ppv), 4),
        "pct_flagged": round(float(pct_flagged), 4),
        "alerts_per_1000": round(float(alerts_per_1000), 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Calibration slope, intercept, Brier score, base-rate Brier."""
    brier = float(brier_score_loss(y_true, y_proba))
    prevalence = float(y_true.mean())
    brier_base = prevalence * (1 - prevalence)  # PREP spec requirement

    # Calibration slope/intercept (Cox calibration): logistic regression of the
    # outcome on the model's log-odds. Using log(p) here instead of log-odds
    # would misreport the slope — a perfectly calibrated model must give ~1 / ~0.
    from sklearn.linear_model import LogisticRegression
    p_clipped = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1 - 1e-7)
    log_odds = np.log(p_clipped / (1.0 - p_clipped))
    cal_model = LogisticRegression(fit_intercept=True)
    cal_model.fit(log_odds.reshape(-1, 1), y_true)
    slope = float(cal_model.coef_[0][0])
    intercept = float(cal_model.intercept_[0])

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)

    return {
        "brier_score": round(brier, 4),
        "brier_base_rate": round(brier_base, 4),
        "brier_skill": round(1 - brier / brier_base, 4),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 4),
        "calibration_curve": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
    }


def compute_net_benefit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Decision curve analysis: net benefit at each threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        nb_model.append(float(nb))
        nb_all.append(float(y_true.mean() - (1 - y_true.mean()) * t / (1 - t)))
        nb_none.append(0.0)
    return {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": nb_none,
    }


def full_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str = "model",
) -> dict:
    """Run all evaluation metrics."""
    auc_roc = float(roc_auc_score(y_true, y_proba))
    auc_prc = float(average_precision_score(y_true, y_proba))
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_proba)

    return {
        "model": model_name,
        "n_test": int(len(y_true)),
        "prevalence": round(float(y_true.mean()), 4),
        "auc_roc": round(auc_roc, 4),
        "auc_roc_ci_95": [ci_lo, ci_hi],
        "auc_prc": round(auc_prc, 4),
        "threshold": round(threshold, 4),
        "threshold_method": "Youden-J on validation set",
        **compute_threshold_metrics(y_true, y_proba, threshold),
        **compute_calibration_metrics(y_true, y_proba),
    }
