"""
PREP Fairness Module
Audits model performance across demographic subgroups.
Race/ethnicity is used for AUDITING ONLY — never as a model feature.
Minimum 30 readmission events required for pass/fail classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

AUC_GAP_LIMIT = 0.05
FNR_GAP_LIMIT = 0.10
MIN_EVENTS = 30  # PREP spec: groups with < 30 readmissions → INSUFFICIENT EVENTS
MIN_SUBGROUP_N = 10

SUBGROUP_DEFINITIONS = {
    "Race/Ethnicity": ("race", ["White", "Black/AA", "Hispanic", "Asian", "Other/Unknown"]),
    "Gender": ("gender", ["M", "F"]),
    "Age Group": ("age_group", ["18-50", "51-65", "66-75", "76-85", "85+"]),
    "Insurance": ("insurance", ["Medicare", "Medicaid", "Private", "Other"]),
}


def subgroup_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict | None:
    n = len(y_true)
    n_pos = int(y_true.sum())
    if n < 10:
        return None

    try:
        auc = round(float(roc_auc_score(y_true, y_proba)), 4)
    except Exception:
        auc = None

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else None
    ppv = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    flag_rate = round(float(y_pred.mean()), 4)
    obs_risk = round(float(y_true.mean()), 4)
    pred_risk = round(float(y_proba.mean()), 4)

    sufficient = n_pos >= MIN_EVENTS

    return {
        "n": n,
        "n_positive": n_pos,
        "sufficient_events": sufficient,
        "auc_roc": auc,
        "fnr": fnr,
        "ppv": ppv,
        "flag_rate": flag_rate,
        "observed_risk": obs_risk,
        "predicted_risk": pred_risk,
    }


def run_fairness_audit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Run full fairness audit across all subgroup dimensions.
    Returns structured results with gap calculations and pass/fail status.
    """
    results = {}

    for dimension, (col, categories) in SUBGROUP_DEFINITIONS.items():
        if col not in meta.columns:
            continue

        subgroups = []
        for cat in categories:
            mask = meta[col].astype(str).str.strip() == cat
            n = int(mask.sum())
            if n < MIN_SUBGROUP_N:
                subgroups.append({
                    "subgroup": cat,
                    "n": n,
                    "n_positive": int(y_true[mask].sum()),
                    "sufficient_events": False,
                    "auc_roc": None,
                    "fnr": None,
                    "ppv": None,
                    "flag_rate": None,
                    "observed_risk": None,
                    "predicted_risk": None,
                })
                continue
            m = subgroup_metrics(y_true[mask], y_proba[mask], threshold)
            if m:
                m["subgroup"] = cat
                subgroups.append(m)

        if not subgroups:
            continue

        # A gap can only be certified when at least two groups have enough
        # events, and no observed group is below the event-count minimum.
        sufficient = [s for s in subgroups if s["sufficient_events"]]
        aucs = [s["auc_roc"] for s in sufficient if s["auc_roc"] is not None]
        fnrs = [s["fnr"] for s in sufficient if s["fnr"] is not None]

        auc_gap = round(max(aucs) - min(aucs), 4) if len(aucs) > 1 else None
        fnr_gap = round(max(fnrs) - min(fnrs), 4) if len(fnrs) > 1 else None

        auc_pass = (auc_gap <= AUC_GAP_LIMIT) if auc_gap is not None else None
        fnr_pass = (fnr_gap <= FNR_GAP_LIMIT) if fnr_gap is not None else None
        has_insufficient_groups = any(
            not subgroup["sufficient_events"] for subgroup in subgroups
        )
        failed = auc_pass is False or fnr_pass is False
        evaluable = (
            not has_insufficient_groups
            and auc_pass is not None
            and fnr_pass is not None
        )
        status = "FAIL" if failed else "PASS" if evaluable else "INDETERMINATE"

        results[dimension] = {
            "subgroups": subgroups,
            "auc_gap": auc_gap,
            "fnr_gap": fnr_gap,
            "auc_pass": auc_pass,
            "fnr_pass": fnr_pass,
            "overall_pass": status == "PASS" if status != "INDETERMINATE" else None,
            "status": status,
            "insufficient_subgroups": [
                subgroup["subgroup"]
                for subgroup in subgroups
                if not subgroup["sufficient_events"]
            ],
        }

    return results


def fairness_summary(audit_results: dict) -> dict:
    """High-level summary for dashboard display."""
    dimensions = list(audit_results.keys())
    statuses = []
    for result in audit_results.values():
        subgroups = result.get("subgroups", [])
        insufficient = any(
            not subgroup.get("sufficient_events", False)
            for subgroup in subgroups
        )
        checks = (result.get("auc_pass"), result.get("fnr_pass"))
        if result.get("status") == "FAIL" or any(check is False for check in checks):
            status = "FAIL"
        elif (
            subgroups
            and not insufficient
            and all(check is not None and bool(check) for check in checks)
        ):
            status = "PASS"
        else:
            # Never trust a stored PASS without its underlying evidence.
            status = "INDETERMINATE"
        statuses.append(status)

    if "FAIL" in statuses:
        status = "FAIL"
    elif not statuses or "INDETERMINATE" in statuses:
        status = "INDETERMINATE"
    else:
        status = "PASS"

    return {
        "dimensions_audited": dimensions,
        "overall_pass": status == "PASS" if status != "INDETERMINATE" else None,
        "auc_gap_limit": AUC_GAP_LIMIT,
        "fnr_gap_limit": FNR_GAP_LIMIT,
        "min_events_required": MIN_EVENTS,
        "min_subgroup_n": MIN_SUBGROUP_N,
        "status": status,
    }
