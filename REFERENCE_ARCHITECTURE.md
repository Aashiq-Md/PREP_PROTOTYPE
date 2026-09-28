# REFERENCE ARCHITECTURE ANALYSIS
## Source: github.com/Tomisin92/readmission-prediction

---

## Pipeline Overview

```
01_data_acquisition.py   → Cohort building from MIMIC-IV
02_feature_engineering.py → Feature matrix + train/val/test split
03_train_models.py        → LogReg, XGBoost, LightGBM + grid search
04_shap_analysis.py       → SHAP TreeExplainer + figures
05_fairness_analysis.py   → Subgroup equity evaluation
06_generate_paper_tables.py → LaTeX tables + metrics JSON
run_all.py                → Master pipeline runner
```

---

## Data Pipeline

- Loads MIMIC-IV hosp tables: admissions, patients, diagnoses, labevents, procedures, prescriptions
- Exclusions: age < 18, missing discharge, in-hospital death, LOS < 1 day
- 30-day readmission label: days_to_next_admission ≤ 30 (from discharge)
- Prior admissions: vectorized per-patient count within 365 days before each admission
- Charlson Comorbidity Index: ICD-10 prefix matching, capped at 24
- Lab features: last value per admission for 10 lab items (creatinine, eGFR, BNP, albumin, WBC, Hgb, Na, K, HbA1c, INR)
- Medication features: unique drug count, high-risk med flag, polypharmacy flag (≥5 meds)
- Demographics: age at admission, gender (binary), insurance (simplified), race (simplified)

---

## Feature Engineering

### Numeric features (17)
age, charlson_index, los_days, n_diagnoses, n_procedures, prior_admissions_12mo,
creatinine, egfr, bnp, albumin, wbc, hemoglobin, sodium, potassium, hba1c, inr, n_medications

### Binary features (4)
gender_enc, emergency_adm, high_risk_med, polypharmacy

### Categorical (one-hot encoded)
race_simple, insurance_simple, age_group

### Missingness indicators
Binary flags for each lab column (10 flags)

---

## Splitting

- **Reference uses**: chronological 70/15/15 by admittime sort
- No patient-level deduplication across splits in reference code
- Imputation: median from train set only, applied to val/test

---

## Models

### Logistic Regression
- L2 regularization, class_weight="balanced"
- Grid search over C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- StandardScaler applied
- Threshold: Youden's J on validation set

### XGBoost
- scale_pos_weight = neg/pos ratio
- Grid search: max_depth [4,6], lr [0.05,0.1], n_estimators [300,500]
- Threshold: Youden's J on validation set

### LightGBM
- scale_pos_weight = neg/pos ratio
- Grid search: num_leaves [31,63], lr [0.05,0.1], n_estimators [300,500]
- Early stopping (50 rounds) on validation AUC
- Threshold: Youden's J on validation set

---

## SHAP

- shap.TreeExplainer on LightGBM model
- Returns shap_values[1] for class 1 (readmitted)
- base_value = expected_value[1]
- Global importance: mean |SHAP| across test set
- Waterfall: per-patient, top-10 features
- Beeswarm: top-12 features

---

## Fairness

- Subgroups: Race/Ethnicity (5), Age Group (5), Gender (2), Insurance (4)
- Metrics per subgroup: AUC-ROC, FNR, PPV, prevalence, N, N_positive
- Global threshold: Youden's J on full test set
- Gap thresholds: AUC ≤ 0.05, FNR ≤ 0.10
- Equalized odds post-processing: per-subgroup Youden threshold (optional)
- Minimum subgroup size: 10 (reference uses 10, PREP spec uses 30 readmissions)

---

## Reference Results (MIMIC-IV Test Set, n=62,285)

| Model | AUC-ROC | AUC-PRC | F1 | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.675 | 0.326 | 0.381 | 0.224 |
| XGBoost | 0.696 | 0.346 | 0.394 | 0.217 |
| LightGBM | 0.689 | 0.333 | 0.390 | 0.146 |

**THESE ARE REFERENCE RESULTS — NOT PREP RESULTS**

---

## What PREP Adapts (Improvements)

| Area | Reference | PREP |
|---|---|---|
| Patient deduplication | Not enforced | Enforce no patient overlap across splits |
| Calibration | Raw probabilities used | Explicit recalibration + Brier vs base-rate |
| Threshold selection | Youden's J (test set risk) | Youden's J on **validation set only** |
| Race in model | Included as feature | Excluded from model, fairness audit only |
| Minimum subgroup | N≥10 | N_positive≥30 for pass/fail |
| Drift monitoring | Proposed architecture only | Implemented KL divergence + 2σ alert |
| Frontend | None | Full React dashboard |
| Demo mode | None | Synthetic patients + drift demo |
| Leakage checks | Not automated | Automated leakage validation |

---

## Components We Should NOT Copy Blindly

1. **Lab features** (creatinine, eGFR, BNP, etc.) — PREP spec does not include labs in primary feature set; they require careful leakage analysis
2. **Polypharmacy flag** — redundant with n_medications; PREP spec uses n_medications directly
3. **Equalized odds post-processing** — reference applies it; PREP spec audits but does not auto-adjust thresholds
4. **Youden threshold on test set** — reference computes threshold using test set; PREP must use validation set only
5. **Race as model feature** — reference includes race_simple in features; PREP explicitly excludes it from the model

---

## Useful Components to Adapt

1. Charlson Comorbidity Index ICD-10 mapping (comprehensive, well-tested)
2. High-risk medication list
3. Prior admissions vectorized computation pattern
4. SHAP TreeExplainer pattern (shap_values[1], base_value[1])
5. Subgroup metrics function structure
6. Bootstrap CI for AUC
7. Chronological 70/15/15 split logic
