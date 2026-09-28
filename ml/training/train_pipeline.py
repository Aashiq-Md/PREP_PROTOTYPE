"""
PREP Training Pipeline — CLI entry point

Runs the full PREP modelling path on whichever cohort it is given:

    load -> leakage check -> chronological 70/15/15 split -> median imputation
         -> LogisticRegression + LightGBM -> validation-set threshold
         -> validation-set logistic recalibration -> test evaluation
         -> fairness audit -> artifacts written to models/

Usage:
    py -3.14 -m ml.training.train_pipeline --source demo --synthetic-label
    py -3.14 -m ml.training.train_pipeline --source uci

Every reported number is produced from the cohort actually used. With
`--source demo` the cohort is synthetic, so the result is labelled SIMULATED
throughout (model card, API metadata, metrics.json) — it is a plumbing check,
not clinical performance.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.data.loader import (  # noqa: E402
    add_synthetic_label, describe_cohort, load_cohort,
)
from ml.features.pipeline import (  # noqa: E402
    AUDIT_ONLY_FEATURES, LEAKAGE_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES,
    TARGET, build_feature_matrix, chronological_split, impute_numeric,
)
from ml.training.train import (  # noqa: E402
    save_model_artifact, train_lightgbm, train_logistic_regression,
    youden_threshold_on_val,
)
from ml.evaluation.metrics import (  # noqa: E402
    compute_calibration_metrics, full_evaluation,
)
from ml.calibration.recalibrate import (  # noqa: E402
    apply_recalibration, calibration_report, fit_recalibrator,
    describe_recalibrator,
)
from ml.fairness.audit import fairness_summary, run_fairness_audit  # noqa: E402

MODELS_DIR = ROOT / "models"


def check_feature_leakage(df: pd.DataFrame) -> list[str]:
    """Assert no leakage/audit-only column is inside the model feature set.

    Returns the list of feature names that would be a problem (empty is good).
    """
    offending = [
        c for c in MODEL_FEATURES
        if c in LEAKAGE_FEATURES or c in AUDIT_ONLY_FEATURES
    ]
    if TARGET in MODEL_FEATURES:
        offending.append(TARGET)
    return offending


def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an ordering column for the chronological split.

    A cohort without admission timestamps cannot be split chronologically; a
    row-order sequence is used instead and flagged, so the split is never
    silently described as temporal when it is not.
    """
    if "admittime" in df.columns:
        return df
    df = df.copy()
    df["admittime"] = pd.date_range("2023-01-01", periods=len(df), freq="1D")
    df.attrs["synthetic_time_order"] = True
    print("[WARN] No admittime column — using synthetic row-order timestamps "
          "(SIMULATED ordering, not real chronology).")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PREP training pipeline")
    parser.add_argument("--source", default="demo", choices=["demo", "uci", "mimic"])
    parser.add_argument(
        "--synthetic-label", action="store_true",
        help="demo only: derive a SIMULATED label from the demo risk score",
    )
    parser.add_argument(
        "--calibration", default="logistic", choices=["logistic", "isotonic"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PREP] Loading cohort (source={args.source})")
    df = load_cohort(args.source)

    if TARGET not in df.columns:
        if args.source != "demo" or not args.synthetic_label:
            print(f"[ERROR] Cohort has no '{TARGET}' label. Supply a labelled "
                  "dataset, or for demo plumbing use --synthetic-label.")
            return 2
        df = add_synthetic_label(df, seed=args.seed)
        print("[WARN] Using a SIMULATED label derived from the demo risk score — "
              "these figures are a plumbing check, not clinical performance.")

    info = describe_cohort(df)
    print(f"[PREP] Cohort: {json.dumps(info)}")
    if info["degraded_features"]:
        print(f"[WARN] Not derivable from this dataset (set to 0): "
              f"{info['degraded_features']}")

    offending = check_feature_leakage(df)
    if offending:
        print(f"[ERROR] Leakage/audit-only columns inside the feature set: {offending}")
        return 3

    df = ensure_time_column(df)
    train, val, test = chronological_split(df)

    def prep(part):
        X, _ = build_feature_matrix(part)
        return X[MODEL_FEATURES].copy(), part[TARGET].astype(int).values

    X_train, y_train = prep(train)
    X_val, y_val = prep(val)
    X_test, y_test = prep(test)

    if y_train.sum() == 0 or y_val.sum() == 0 or y_test.sum() == 0:
        print(f"[ERROR] A split contains no positive events "
              f"(train={int(y_train.sum())}, val={int(y_val.sum())}, "
              f"test={int(y_test.sum())}) — enlarge the cohort or relabel.")
        return 4

    X_train, X_val, X_test, medians = impute_numeric(
        X_train, X_val, X_test, NUMERIC_FEATURES
    )

    print("[PREP] Training Logistic Regression (baseline)")
    logreg = train_logistic_regression(
        X_train, pd.Series(y_train), X_val, pd.Series(y_val)
    )
    print("[PREP] Training LightGBM")
    lgbm = train_lightgbm(X_train, pd.Series(y_train), X_val, pd.Series(y_val))

    candidates = {"LogisticRegression": logreg, "LightGBM": lgbm}
    selected = max(candidates, key=lambda k: candidates[k]["val_auc"])
    art = candidates[selected]
    print(f"[PREP] Selected by validation AUC: {selected} "
          f"(val_AUC={art['val_auc']:.4f})")

    def raw_proba(a, X):
        X_in = a["scaler"].transform(X) if a.get("scaler") is not None else X.values
        return a["model"].predict_proba(X_in)[:, 1]

    p_val_raw = raw_proba(art, X_val)
    p_test_raw = raw_proba(art, X_test)
    raw_threshold = float(art["threshold"])

    calibration = calibration_report(
        y_val, p_val_raw, y_test, p_test_raw, method=args.calibration
    )
    # Ship a calibration only if it improves VALIDATION Brier. Deciding on the
    # validation split (never the test split) keeps the reported test numbers
    # honest and stops a harmful recalibration from being served by the API.
    val_brier_raw = compute_calibration_metrics(y_val, p_val_raw)["brier_score"]
    val_fitted = fit_recalibrator(args.calibration, y_val, p_val_raw)
    val_brier_cal = compute_calibration_metrics(
        y_val, apply_recalibration(val_fitted, p_val_raw)
    )["brier_score"]

    if val_brier_cal < val_brier_raw:
        recalibrator = val_fitted
        p_test_cal = apply_recalibration(recalibrator, p_test_raw)
        calibration["applied"] = True
        calibration["decision"] = "applied (validation Brier improved)"
        print(f"[PREP] Calibration {args.calibration}: validation Brier "
              f"{val_brier_raw} -> {val_brier_cal} -> APPLIED")
    else:
        recalibrator = None
        p_test_cal = p_test_raw
        calibration["applied"] = False
        calibration["decision"] = "not applied (validation Brier did not improve)"
        print(f"[PREP] Calibration {args.calibration}: validation Brier "
              f"{val_brier_raw} -> {val_brier_cal} -> NOT APPLIED "
              f"(raw probabilities served)")
    calibration["validation_brier_raw"] = val_brier_raw
    calibration["validation_brier_calibrated"] = val_brier_cal

    # The threshold must be derived on the SAME scale the API serves. The
    # raw-scale threshold from train.py is meaningless once probabilities have
    # been recalibrated, so it is re-derived on served validation probabilities
    # — still validation-only; the test split is never touched.
    p_val_served = (
        apply_recalibration(recalibrator, p_val_raw)
        if recalibrator is not None else p_val_raw
    )
    threshold = youden_threshold_on_val(y_val, p_val_served)
    print(f"[PREP] Threshold on served validation probabilities: {threshold:.4f} "
          f"(raw-scale value was {raw_threshold:.4f})")

    art["threshold"] = threshold  # artifact and API must use the served threshold

    evaluation = full_evaluation(y_test, p_test_cal, threshold, model_name=selected)
    evaluation["calibration"] = calibration
    evaluation["threshold_source"] = "validation set (Youden's J, served scale)"
    evaluation["threshold_raw_scale"] = round(raw_threshold, 4)

    meta_cols = [
        c for c in ("race", "gender", "age_group", "insurance") if c in test.columns
    ]
    audit = run_fairness_audit(
        y_test, p_test_cal, test[meta_cols].reset_index(drop=True), threshold
    )
    fairness = fairness_summary(audit)
    print(f"[PREP] Fairness: {fairness['status']} "
          f"(dimensions: {', '.join(fairness['dimensions_audited']) or 'none'})")

    # Per-model comparison — each model recalibrated from its own val output.
    comparison = {}
    for name, cand in candidates.items():
        p_val_c = raw_proba(cand, X_val)
        p_test_c = raw_proba(cand, X_test)
        fitted_c = fit_recalibrator(args.calibration, y_val, p_val_c)
        val_before = compute_calibration_metrics(y_val, p_val_c)["brier_score"]
        val_after = compute_calibration_metrics(
            y_val, apply_recalibration(fitted_c, p_val_c)
        )["brier_score"]
        if val_after < val_before:
            p_test_c = apply_recalibration(fitted_c, p_test_c)
            p_val_served_c = apply_recalibration(fitted_c, p_val_c)
            cal_used = args.calibration
        else:
            p_val_served_c = p_val_c
            cal_used = "none"
        thr_c = youden_threshold_on_val(y_val, p_val_served_c)
        m = full_evaluation(y_test, p_test_c, thr_c, model_name=name)
        comparison[name] = {
            "val_auc": round(float(cand["val_auc"]), 4),
            "test_auc_roc": m["auc_roc"],
            "test_auc_prc": m["auc_prc"],
            "test_brier": m["brier_score"],
            "threshold": round(float(thr_c), 4),
            "calibration_applied": cal_used,
        }

    is_demo = args.source == "demo"
    meta = {
        "model_version": f"{'lgbm' if selected == 'LightGBM' else 'logreg'}-"
                         f"{args.source}-v0.1",
        "dataset": df.attrs.get("dataset_label", args.source),
        "source": args.source,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "prevalence_test": round(float(y_test.mean()), 4),
        "calibration_method": (
            describe_recalibrator(recalibrator)["method"]
            if recalibrator is not None
            else "none (validation Brier not improved)"
        ),
        "threshold_method": "Youden-J on validation set (served scale)",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "validation_state": (
            "NOT YET VALIDATED" if is_demo else "PROTOTYPE — not clinically validated"
        ),
        "median_imputation": {k: round(float(v), 4) for k, v in medians.items()},
        "result_label": "SIMULATED" if is_demo else "UNCALIBRATED PROTOTYPE",
        "disclaimer": "PREP is a research/hackathon prototype. Not clinically validated.",
    }

    art_path = out_dir / ("lightgbm.pkl" if selected == "LightGBM" else "logreg.pkl")
    save_model_artifact(art, art_path, meta, recalibrator=recalibrator)
    calibration_doc = (
        describe_recalibrator(recalibrator)
        if recalibrator is not None
        else {"method": "none", "reason": calibration["decision"]}
    )
    calibration_doc["applied"] = calibration["applied"]
    with open(out_dir / "recalibrator.json", "w") as f:
        json.dump(calibration_doc, f, indent=2)

    try:
        artifact_rel = str(art_path.relative_to(ROOT))
    except ValueError:  # custom --out outside the repo
        artifact_rel = str(art_path)

    report = {
        "generated": meta["training_date"],
        "source": args.source,
        "dataset": meta["dataset"],
        "result_label": meta["result_label"],
        "selected_model": selected,
        "threshold": round(threshold, 4),
        "threshold_source": "validation set (Youden's J)",
        "cohort": info,
        "evaluation": evaluation,
        "model_comparison": comparison,
        "fairness": fairness,
        "fairness_detail": audit,
        "artifact": artifact_rel,
        "disclaimer": meta["disclaimer"]
        + " Demo figures are SIMULATED, not clinical performance.",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[PREP] Artifact: {art_path}")
    print(f"[PREP] Report:   {out_dir / 'metrics.json'}")
    print(f"[PREP] Test AUC-ROC {evaluation['auc_roc']} "
          f"(95% CI {evaluation['auc_roc_ci_95']}), threshold {round(threshold, 4)}")
    if is_demo:
        print("[PREP] SIMULATED — synthetic cohort, synthetic label. "
              "Report these numbers only as a pipeline check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


