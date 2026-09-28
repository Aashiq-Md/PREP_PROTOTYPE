"""
PREP Calibration
Probability recalibration fitted on the VALIDATION set only.

PREP specification:
  - raw model probabilities are recalibrated before thresholding
  - the fitted parameters come from validation data, never from test data
  - Brier score is reported against the base-rate Brier score

Two methods are provided:

``logistic`` (default)
    Platt-style single-variable logistic regression on the model logit
    (``p_cal = sigmoid(a * logit(p) + b)``). One intercept + one slope, so it is
    stable on the small validation splits a prototype produces.

``isotonic``
    Monotone step function. Listed as future work in the README; provided for
    comparison, but it needs far more validation data than logistic fitting to
    avoid overfitting.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-7
METHODS = ("logistic", "isotonic")


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of raw probabilities, clipped away from 0 and 1.

    Must be the true log-odds ``log(p / (1 - p))``: fitting a recalibrator on
    ``log(p)`` instead would still produce a monotone map, but the fitted slope
    and intercept would no longer be interpretable as the standard calibration
    slope/intercept (a perfectly calibrated model would not give slope ~1).
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class LogisticRecalibrator:
    """Recalibrate raw probabilities with a fitted slope and intercept.

    ``p_cal = sigmoid(slope * logit(p) + intercept)``
    """

    def __init__(self) -> None:
        self.slope: float | None = None
        self.intercept: float | None = None
        self.fitted = False
        self.n_fit: int = 0

    def fit(self, y_val: np.ndarray, p_val: np.ndarray) -> "LogisticRecalibrator":
        """Fit on validation labels and raw validation probabilities."""
        y_val = np.asarray(y_val, dtype=int)
        p_val = np.asarray(p_val, dtype=float)
        if len(y_val) != len(p_val):
            raise ValueError("y_val and p_val must have equal length")
        if y_val.sum() == 0 or y_val.sum() == len(y_val):
            raise ValueError("validation set must contain both classes to calibrate")

        model = LogisticRegression(fit_intercept=True, max_iter=1000)
        model.fit(logit(p_val).reshape(-1, 1), y_val)
        self.slope = float(model.coef_[0][0])
        self.intercept = float(model.intercept_[0])
        self.n_fit = int(len(y_val))
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Apply the fitted recalibration to raw probabilities."""
        if not self.fitted:
            raise RuntimeError("recalibrator must be fitted before use")
        z = self.slope * logit(p) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    # sklearn-style alias, so callers can treat it like any fitted estimator
    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.transform(p)

    def to_dict(self) -> dict:
        return {
            "method": "logistic",
            "slope": round(self.slope, 6) if self.slope is not None else None,
            "intercept": round(self.intercept, 6) if self.intercept is not None else None,
            "n_fit": self.n_fit,
        }


def fit_logistic_recalibration(
    y_val: np.ndarray, p_val: np.ndarray
) -> LogisticRecalibrator:
    """Fit logistic (Platt) recalibration on the validation set."""
    return LogisticRecalibrator().fit(y_val, p_val)


def fit_isotonic_recalibration(
    y_val: np.ndarray, p_val: np.ndarray
) -> IsotonicRegression:
    """Fit isotonic calibration on the validation set (future-work option)."""
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)
    if len(y_val) != len(p_val):
        raise ValueError("y_val and p_val must have equal length")
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        raise ValueError("validation set must contain both classes to calibrate")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def fit_recalibrator(method: str, y_val: np.ndarray, p_val: np.ndarray):
    """Fit the requested recalibration method on validation data."""
    if method == "logistic":
        return fit_logistic_recalibration(y_val, p_val)
    if method == "isotonic":
        return fit_isotonic_recalibration(y_val, p_val)
    raise ValueError(f"unknown calibration method: {method!r} (expected {METHODS})")


def apply_recalibration(recalibrator, p: np.ndarray) -> np.ndarray:
    """Apply a fitted recalibrator (logistic or isotonic) to probabilities."""
    if isinstance(recalibrator, LogisticRecalibrator):
        return recalibrator.transform(p)
    return np.asarray(recalibrator.predict(np.asarray(p, dtype=float)), dtype=float)


def describe_recalibrator(recalibrator) -> dict:
    """JSON-safe description of a fitted recalibrator, for model metadata."""
    if isinstance(recalibrator, LogisticRecalibrator):
        return recalibrator.to_dict()
    return {"method": "isotonic", "n_fit": None}


def calibration_report(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    method: str = "logistic",
) -> dict:
    """Fit on validation, then report the Brier score before/after on test data.

    The test set is used ONLY to report the effect of a calibration that was
    already fitted on validation data — it never influences the fit.
    """
    from ml.evaluation.metrics import compute_calibration_metrics

    recalibrator = fit_recalibrator(method, y_val, p_val)
    p_test_cal = apply_recalibration(recalibrator, p_test)

    before = compute_calibration_metrics(y_test, p_test)
    after = compute_calibration_metrics(y_test, p_test_cal)

    return {
        "method": method,
        "fitted_on": "validation set",
        "recalibrator": describe_recalibrator(recalibrator),
        "before": {
            "brier_score": before["brier_score"],
            "calibration_slope": before["calibration_slope"],
            "calibration_intercept": before["calibration_intercept"],
        },
        "after": {
            "brier_score": after["brier_score"],
            "calibration_slope": after["calibration_slope"],
            "calibration_intercept": after["calibration_intercept"],
        },
        "brier_base_rate": before["brier_base_rate"],
        "brier_skill_after": after["brier_skill"],
        "improved": bool(after["brier_score"] < before["brier_score"]),
    }
