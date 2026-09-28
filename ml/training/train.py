"""
PREP Model Training
Trains Logistic Regression and LightGBM with validation-set threshold selection.
Threshold is NEVER selected using the test set.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb

RANDOM_SEED = 42


def youden_threshold_on_val(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Select threshold using Youden's J on VALIDATION set only."""
    fpr, tpr, thresholds = roc_curve(y_val, proba_val)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    best_auc, best_C, best_model = 0.0, 1.0, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, random_state=RANDOM_SEED,
            class_weight="balanced", solver="lbfgs",
        )
        clf.fit(X_tr, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_va)[:, 1])
        if auc > best_auc:
            best_auc, best_C, best_model = auc, C, clf

    proba_val = best_model.predict_proba(X_va)[:, 1]
    threshold = youden_threshold_on_val(y_val.values, proba_val)

    print(f"[LogReg] best_C={best_C}  val_AUC={best_auc:.4f}  threshold={threshold:.4f}")
    return {
        "model": best_model,
        "scaler": scaler,
        "threshold": threshold,
        "val_auc": best_auc,
        "model_type": "LogisticRegression",
        "best_C": best_C,
    }


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    param_grid = [
        {"num_leaves": nl, "learning_rate": lr, "n_estimators": n}
        for nl in [31, 63]
        for lr in [0.05, 0.1]
        for n in [300, 500]
    ]

    best_auc, best_params, best_model = 0.0, None, None
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

    for params in param_grid:
        clf = lgb.LGBMClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, clf

    proba_val = best_model.predict_proba(X_val)[:, 1]
    threshold = youden_threshold_on_val(y_val.values, proba_val)

    print(f"[LightGBM] best_params={best_params}  val_AUC={best_auc:.4f}  threshold={threshold:.4f}")
    return {
        "model": best_model,
        "scaler": None,
        "threshold": threshold,
        "val_auc": best_auc,
        "model_type": "LightGBM",
        "best_params": best_params,
    }


def save_model_artifact(artifact: dict, path: Path, meta: dict, recalibrator=None) -> None:
    """Save model, threshold, calibration and metadata as one joblib artifact.

    ``recalibrator`` is the object fitted on the VALIDATION set by
    ml/calibration/recalibrate.py. The backend applies it to raw probabilities
    so the risk it serves matches the calibrated numbers in metrics.json.
    """
    payload = {
        "model": artifact["model"],
        "scaler": artifact.get("scaler"),
        "threshold": artifact["threshold"],
        "model_type": artifact["model_type"],
        "recalibrator": recalibrator,
        "meta": meta,
    }
    joblib.dump(payload, path)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in {**meta, "threshold": artifact["threshold"],
                                      "model_type": artifact["model_type"],
                                      "val_auc": artifact["val_auc"]}.items()
                   if isinstance(v, (str, int, float, bool, list, dict, type(None)))}, f, indent=2)
    print(f"[SAVED] {path}")
