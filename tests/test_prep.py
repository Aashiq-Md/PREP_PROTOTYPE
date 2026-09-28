"""
PREP Test Suite
Tests for features, leakage, drift, and API schemas.
"""

import sys
import json
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from ml.features.pipeline import (
    leakage_check, encode_features, build_feature_matrix,
    chronological_split, LEAKAGE_FEATURES, AUDIT_ONLY_FEATURES,
)
from backend.app.monitoring.drift import (
    kl_divergence_numeric, kl_divergence_categorical,
    compute_feature_drift, compute_prediction_drift,
)
from ml.evaluation.metrics import (
    compute_threshold_metrics, compute_calibration_metrics,
)
from ml.data.loader import (
    DEMO_COHORT, EXPECTED_COLUMNS, TARGET, add_synthetic_label,
    describe_cohort, load_cohort, load_demo_cohort, load_mimic_cohort,
    load_uci_cohort,
)
from ml.calibration.recalibrate import (
    LogisticRecalibrator, apply_recalibration, calibration_report,
    describe_recalibrator, fit_recalibrator,
)
from ml.calibration.recalibrate import logit as recalibrate_logit
from ml.training.train_pipeline import check_feature_leakage, ensure_time_column
from ml.fairness.audit import fairness_summary, run_fairness_audit
from backend.app.services.predictor import PREPPredictor
from backend.app import main as prep_api
import run as prep_runner


# ── Feature pipeline tests ────────────────────────────────────────────────────

def make_cohort(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject_id": range(n),
        "age": rng.integers(20, 90, n),
        "gender": rng.choice(["M", "F"], n),
        "insurance": rng.choice(["Medicare", "Medicaid", "Private", "Other"], n),
        "prior_admissions_12mo": rng.integers(0, 5, n),
        "los_days": rng.exponential(4, n) + 1,
        "n_diagnoses": rng.integers(1, 20, n),
        "n_procedures": rng.integers(0, 10, n),
        "charlson_index": rng.integers(0, 12, n),
        "n_medications": rng.integers(0, 20, n),
        "high_risk_med": rng.integers(0, 2, n),
        "emergency_adm": rng.integers(0, 2, n),
        "readmitted_30d": rng.integers(0, 2, n),
        "admittime": pd.date_range("2018-01-01", periods=n, freq="3D"),
    })


class TestLeakageCheck:
    def test_no_leakage_clean_df(self):
        # A cohort with the outcome label removed has no leakage-risk columns
        df = make_cohort().drop(columns=["readmitted_30d"])
        assert leakage_check(df) == []

    def test_detects_target_label_as_leakage(self):
        # The outcome label must never be available to the model as a feature
        df = make_cohort()
        assert "readmitted_30d" in leakage_check(df)

    def test_detects_next_admittime(self):
        df = make_cohort()
        df["next_admittime"] = pd.NaT
        found = leakage_check(df)
        assert "next_admittime" in found

    def test_detects_race_in_features(self):
        df = make_cohort()
        df["race"] = "White"
        found = leakage_check(df)
        assert "race" in found

    def test_detects_days_to_next(self):
        df = make_cohort()
        df["days_to_next"] = 0
        found = leakage_check(df)
        assert "days_to_next" in found


class TestEncodeFeatures:
    def test_gender_encoding(self):
        df = pd.DataFrame({"gender": ["M", "F", "M"]})
        out = encode_features(df)
        assert list(out["gender_enc"]) == [1, 0, 1]

    def test_insurance_one_hot(self):
        df = pd.DataFrame({"insurance": ["Medicare", "Medicaid", "Private", "Other"]})
        out = encode_features(df)
        assert out["insurance_Medicare"].tolist() == [1, 0, 0, 0]
        assert out["insurance_Medicaid"].tolist() == [0, 1, 0, 0]
        assert out["insurance_Private"].tolist() == [0, 0, 1, 0]

    def test_zero_prior_admissions(self):
        df = make_cohort(10)
        df["prior_admissions_12mo"] = 0
        X, cols = build_feature_matrix(df)
        assert (X["prior_admissions_12mo"] == 0).all()

    def test_extreme_age(self):
        df = make_cohort(5)
        df["age"] = [18, 120, 65, 45, 90]
        X, cols = build_feature_matrix(df)
        assert X["age"].max() <= 120
        assert X["age"].min() >= 18


class TestChronologicalSplit:
    def test_split_proportions(self):
        df = make_cohort(200)
        train, val, test = chronological_split(df)
        assert len(train) == 140
        assert len(val) == 30
        assert len(test) == 30

    def test_no_patient_overlap(self):
        df = make_cohort(200)
        train, val, test = chronological_split(df)
        train_ids = set(train["subject_id"])
        val_ids = set(val["subject_id"])
        test_ids = set(test["subject_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_repeated_patients_are_kept_in_earliest_split_only(self):
        df = make_cohort(200)
        df.loc[141, "subject_id"] = df.loc[10, "subject_id"]
        df.loc[170, "subject_id"] = df.loc[145, "subject_id"]
        df.loc[185, "subject_id"] = df.loc[12, "subject_id"]

        train, val, test = chronological_split(df)
        train_ids = set(train["subject_id"])
        val_ids = set(val["subject_id"])
        test_ids = set(test["subject_id"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert 141 not in val.index
        assert 170 not in test.index
        assert 185 not in test.index

    def test_duplicate_patient_ids_do_not_cross_any_split_boundary(self):
        df = make_cohort(200)
        df.loc[[5, 75, 150], "subject_id"] = 9999
        train, val, test = chronological_split(df)

        for earlier, later in ((train, val), (train, test), (val, test)):
            assert set(earlier.subject_id).isdisjoint(set(later.subject_id))

    def test_patient_id_column_is_also_isolated(self):
        df = make_cohort(200).rename(columns={"subject_id": "patient_id"})
        df.loc[141, "patient_id"] = df.loc[10, "patient_id"]
        train, val, test = chronological_split(df)

        for earlier, later in ((train, val), (train, test), (val, test)):
            assert set(earlier.patient_id).isdisjoint(set(later.patient_id))

    def test_chronological_order(self):
        df = make_cohort(100)
        train, val, test = chronological_split(df)
        assert train["admittime"].max() <= val["admittime"].min()
        assert val["admittime"].max() <= test["admittime"].min()


# ── Drift monitoring tests ────────────────────────────────────────────────────

class TestDriftMonitoring:
    def test_no_drift_identical_distributions(self):
        rng = np.random.default_rng(0)
        data = rng.normal(50, 10, 500)
        kl = kl_divergence_numeric(data, data)
        assert kl < 0.05, f"Expected near-zero KL for identical data, got {kl}"

    def test_drift_detected_on_shifted_distribution(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(50, 10, 500)
        cur = rng.normal(80, 10, 500)  # large shift
        kl = kl_divergence_numeric(ref, cur)
        assert kl > 0.05, f"Expected KL > 0.05 for shifted distribution, got {kl}"

    def test_categorical_no_drift(self):
        ref = np.array(["A", "B", "C"] * 100)
        cur = np.array(["A", "B", "C"] * 100)
        kl = kl_divergence_categorical(ref, cur)
        assert kl < 0.05

    def test_categorical_drift(self):
        ref = np.array(["A"] * 90 + ["B"] * 10)
        cur = np.array(["A"] * 10 + ["B"] * 90)  # flipped proportions
        kl = kl_divergence_categorical(ref, cur)
        assert kl > 0.05

    def test_prediction_drift_alert(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0.20, 0.05, 500)
        cur = rng.normal(0.45, 0.05, 500)  # large shift
        result = compute_prediction_drift(ref, cur)
        assert result["alert"] is True

    def test_prediction_no_drift(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0.20, 0.05, 500)
        cur = rng.normal(0.21, 0.05, 500)  # tiny shift
        result = compute_prediction_drift(ref, cur)
        assert result["alert"] is False


# ── Evaluation metrics tests ──────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_threshold_metrics_perfect(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_threshold_metrics(y, p, threshold=0.5)
        assert m["sensitivity"] == 1.0
        assert m["ppv"] == 1.0

    def test_brier_base_rate(self):
        y = np.array([1, 0] * 50)
        p = np.array([0.5] * 100)
        m = compute_calibration_metrics(y, p)
        prevalence = 0.5
        expected_base = prevalence * (1 - prevalence)
        assert abs(m["brier_base_rate"] - expected_base) < 1e-6

    def test_brier_skill_positive_for_good_model(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 200)
        # Good model: probabilities close to true labels
        p = np.clip(y.astype(float) + rng.normal(0, 0.1, 200), 0.01, 0.99)
        m = compute_calibration_metrics(y, p)
        assert m["brier_skill"] > 0


class TestFairnessEvidence:
    def test_empty_audit_is_indeterminate_not_pass(self):
        summary = fairness_summary({})
        assert summary["status"] == "INDETERMINATE"
        assert summary["overall_pass"] is None

    def test_demo_explanation_is_labeled_as_non_shap(self, tmp_path):
        predictor = PREPPredictor(tmp_path)
        patient = {
            "age": 72, "gender": "F", "insurance": "Medicare",
            "prior_admissions_12mo": 2, "los_days": 8, "n_diagnoses": 11,
            "n_procedures": 4, "charlson_index": 5, "n_medications": 12,
            "high_risk_med": 1, "emergency_adm": 0,
        }

        explanation = predictor.explain(patient)
        assert explanation["explanation_method"] == "heuristic_approximation"
        assert "not SHAP" in explanation["explanation_label"]

    def test_fairness_api_recomputes_legacy_synthetic_pass_conservatively(
        self, tmp_path, monkeypatch
    ):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "metrics.json").write_text(json.dumps({
            "source": "demo",
            "result_label": "SIMULATED",
            "dataset": "SIMULATED test data",
            "fairness": {"overall_pass": True, "status": "PASS"},
            "fairness_detail": {
                "Gender": {
                    "overall_pass": None,
                    "auc_pass": None,
                    "fnr_pass": None,
                    "subgroups": [
                        {"subgroup": "M", "sufficient_events": False}
                    ],
                }
            },
        }), encoding="utf-8")
        monkeypatch.setattr(prep_api, "MODELS_DIR", models_dir)

        report = prep_api.fairness_report()

        assert report["simulated"] is True
        assert report["summary"]["status"] == "INDETERMINATE"
        assert report["summary"]["overall_pass"] is None

    def test_audit_with_insufficient_events_is_not_pass(self):
        y_true = np.array([1] * 35 + [0] * 35 + [1] * 5 + [0] * 25)
        y_proba = np.array([0.9] * 35 + [0.1] * 35 + [0.9] * 5 + [0.1] * 25)
        meta = pd.DataFrame({"gender": ["M"] * 70 + ["F"] * 30})

        audit = run_fairness_audit(y_true, y_proba, meta, threshold=0.5)
        dimension = audit["Gender"]
        summary = fairness_summary(audit)

        assert dimension["status"] == "INDETERMINATE"
        assert dimension["overall_pass"] is None
        assert dimension["insufficient_subgroups"] == ["F"]
        assert summary["status"] == "INDETERMINATE"
        assert summary["overall_pass"] is None

    def test_audit_with_small_subgroup_is_not_pass(self):
        y_true = np.array([1] * 35 + [0] * 35 + [1] * 4 + [0] * 4)
        y_proba = np.where(y_true == 1, 0.9, 0.1)
        meta = pd.DataFrame({"gender": ["M"] * 70 + ["F"] * 8})

        audit = run_fairness_audit(y_true, y_proba, meta, threshold=0.5)
        female = next(
            subgroup for subgroup in audit["Gender"]["subgroups"]
            if subgroup["subgroup"] == "F"
        )
        assert female["n"] == 8
        assert female["sufficient_events"] is False
        assert audit["Gender"]["status"] == "INDETERMINATE"

    def test_valid_sufficient_groups_can_pass(self):
        y_true = np.array([1] * 35 + [0] * 35 + [1] * 35 + [0] * 35)
        y_proba = np.where(y_true == 1, 0.9, 0.1)
        meta = pd.DataFrame({"gender": ["M"] * 70 + ["F"] * 70})

        audit = run_fairness_audit(y_true, y_proba, meta, threshold=0.5)
        summary = fairness_summary(audit)

        assert audit["Gender"]["status"] == "PASS"
        assert summary["status"] == "PASS"
        assert summary["overall_pass"] is True

    def test_saved_report_without_status_cannot_pass_insufficient_groups(self):
        old_shape_audit = {
            "Gender": {
                "subgroups": [{"sufficient_events": False}],
                "auc_pass": None,
                "fnr_pass": None,
                "overall_pass": None,
            }
        }
        summary = fairness_summary(old_shape_audit)
        assert summary["status"] == "INDETERMINATE"
        assert summary["overall_pass"] is None


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    @staticmethod
    def _demo_or_skip():
        if not DEMO_COHORT.exists():
            pytest.skip("demo cohort not generated yet")
        return load_demo_cohort()

    def test_demo_cohort_has_expected_features(self):
        df = self._demo_or_skip()
        assert len(df) > 0
        assert [c for c in EXPECTED_COLUMNS if c not in df.columns] == []
        assert "admittime" in df.columns

    def test_demo_cohort_is_unlabelled_by_default(self):
        # The demo cohort must never arrive with an invented outcome label.
        df = self._demo_or_skip()
        assert TARGET not in df.columns
        assert df.attrs["source"] == "demo"
        assert "SIMULATED" in df.attrs["dataset_label"]

    def test_synthetic_label_is_binary_and_reproducible(self):
        df = add_synthetic_label(self._demo_or_skip(), seed=7)
        again = add_synthetic_label(self._demo_or_skip(), seed=7)
        assert set(df[TARGET].unique()) <= {0, 1}
        assert df[TARGET].tolist() == again[TARGET].tolist()

    def test_synthetic_label_requires_risk_column(self):
        with pytest.raises(ValueError):
            add_synthetic_label(pd.DataFrame({"age": [50]}))

    def test_describe_cohort_reports_gaps(self):
        info = describe_cohort(self._demo_or_skip())
        assert info["source"] == "demo"
        assert info["has_label"] is False
        assert info["prevalence"] is None
        assert info["missing_expected"] == []

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError):
            load_cohort("not-a-source")

    def test_mimic_requires_credentialed_data(self):
        with pytest.raises(FileNotFoundError) as err:
            load_mimic_cohort()
        assert "physionet" in str(err.value).lower()

    def test_uci_schema_mapping(self, tmp_path):
        csv = tmp_path / "uci.csv"
        pd.DataFrame({
            "patient_nbr": [1, 2, 3],
            "age": ["[70-80)", "[50-60)", "[90-100)"],
            "gender": ["Male", "Female", "Female"],
            "payer_code": ["MC", "MD", "ZZ"],
            "race": ["Caucasian", "?", "AfricanAmerican"],
            "number_inpatient": [2, 0, 5],
            "time_in_hospital": [4, 2, 9],
            "number_diagnoses": [7, 3, 9],
            "num_procedures": [1, 0, 3],
            "num_medications": [12, 5, 20],
            "admission_type_id": [1, 3, 1],
            "readmitted": ["<30", "NO", ">30"],
        }).to_csv(csv, index=False)

        df = load_uci_cohort(path=csv)
        # Brackets become midpoints; age_group is derived from the midpoint
        assert df["age"].tolist() == [75.0, 55.0, 95.0]
        assert df["age_group"].tolist() == ["66-75", "51-65", "85+"]
        assert df["gender"].tolist() == ["M", "F", "F"]
        assert df["insurance"].tolist() == ["Medicare", "Medicaid", "Other"]
        assert df["race"].tolist() == ["Caucasian", "Other/Unknown", "AfricanAmerican"]
        # Emergency(1) / Urgent(2) only
        assert df["emergency_adm"].tolist() == [1, 0, 1]
        assert df[TARGET].tolist() == [1, 0, 0]
        # Not derivable from the UCI schema — declared, not invented
        assert df["charlson_index"].tolist() == [0, 0, 0]
        assert "charlson_index" in df.attrs["degraded_features"]

    def test_uci_requires_label_when_asked(self, tmp_path):
        csv = tmp_path / "uci_unlabelled.csv"
        pd.DataFrame({"patient_nbr": [1], "age": ["[60-70)"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError):
            load_uci_cohort(path=csv, require_label=True)

# ── Calibration tests ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_recalibrator_requires_fit_before_transform(self):
        with pytest.raises(RuntimeError):
            LogisticRecalibrator().transform(np.array([0.5]))

    def test_single_class_validation_set_rejected(self):
        y = np.zeros(50, dtype=int)
        with pytest.raises(ValueError):
            fit_recalibrator("logistic", y, np.full(50, 0.3))

    def test_unknown_method_rejected(self):
        y = np.array([0, 1] * 25)
        with pytest.raises(ValueError):
            fit_recalibrator("magic", y, np.full(50, 0.3))

    def test_already_calibrated_model_recovers_unit_slope(self):
        # If raw probabilities already equal the true event rate, no correction
        # is needed: fitted slope ~1, intercept ~0.
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.random(len(p)) < p).astype(int)
        rec = fit_recalibrator("logistic", y, p)
        assert abs(rec.slope - 1.0) < 0.25
        assert abs(rec.intercept) < 0.25

    def test_overconfident_probabilities_are_pulled_down(self):
        # Model says ~0.9 but only ~50% of those patients have the event.
        rng = np.random.default_rng(1)
        p = rng.uniform(0.80, 0.99, 4000)
        y = (rng.random(len(p)) < 0.5).astype(int)
        rec = fit_recalibrator("logistic", y, p)
        assert rec.transform(p).mean() < p.mean()

    def test_logit_is_log_odds(self):
        logit = recalibrate_logit
        np.testing.assert_allclose(logit(np.array([0.5])), [0.0], atol=1e-12)
        np.testing.assert_allclose(logit(np.array([0.75])), [np.log(3.0)], atol=1e-9)
        assert logit(np.array([0.9]))[0] > logit(np.array([0.6]))[0]

    def test_transform_matches_stated_sigmoid_form(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0.01, 0.99, 500)
        y = (rng.random(len(p)) < p).astype(int)
        rec = fit_recalibrator("logistic", y, p)
        # Independent log-odds, computed here rather than reusing the module's
        log_odds = np.log(p / (1.0 - p))
        expected = 1.0 / (1.0 + np.exp(-(rec.slope * log_odds + rec.intercept)))
        np.testing.assert_allclose(apply_recalibration(rec, p), expected, atol=1e-9)

    def test_isotonic_output_is_a_probability_vector(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.95, 500)
        y = (rng.random(len(p)) < p).astype(int)
        rec = fit_recalibrator("isotonic", y, p)
        out = apply_recalibration(rec, p)
        assert out.shape == p.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert describe_recalibrator(rec)["method"] == "isotonic"

    def test_calibration_report_is_fitted_on_validation(self):
        rng = np.random.default_rng(4)
        p_val = rng.uniform(0.05, 0.95, 400)
        y_val = (rng.random(400) < p_val).astype(int)
        p_test = rng.uniform(0.05, 0.95, 300)
        y_test = (rng.random(300) < p_test).astype(int)
        rep = calibration_report(y_val, p_val, y_test, p_test)
        assert rep["fitted_on"] == "validation set"
        assert set(rep["before"]) == {
            "brier_score", "calibration_slope", "calibration_intercept"
        }
        assert isinstance(rep["improved"], bool)
        assert rep["brier_base_rate"] > 0


# ── Training pipeline helper tests ────────────────────────────────────────────

class TestTrainingPipelineHelpers:
    def test_no_leakage_in_canonical_feature_set(self):
        assert check_feature_leakage(make_cohort()) == []

    def test_missing_time_column_is_synthesised_and_flagged(self):
        df = make_cohort(50).drop(columns=["admittime"])
        out = ensure_time_column(df)
        assert "admittime" in out.columns
        assert out.attrs["synthetic_time_order"] is True
        assert out["admittime"].is_monotonic_increasing

    def test_existing_time_column_is_preserved(self):
        df = make_cohort(20)
        original = df["admittime"].tolist()
        out = ensure_time_column(df)
        assert out["admittime"].tolist() == original
        assert "synthetic_time_order" not in out.attrs


class TestRunnerExitStatus:
    def test_test_command_propagates_pytest_failure(self, monkeypatch):
        monkeypatch.setattr(prep_runner, "preflight", lambda: True)
        monkeypatch.setattr(
            prep_runner.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1),
        )
        with pytest.raises(SystemExit) as exc:
            prep_runner.run_tests()
        assert exc.value.code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
