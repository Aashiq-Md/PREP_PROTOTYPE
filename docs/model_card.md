# PREP Model Card

## Model Details

| Field | Value |
|---|---|
| Name | PREP — Predict Readmission Estimation of Patient |
| Version | Generated from the active artifact; see backend metadata |
| Type | LightGBM (primary) / Logistic Regression (baseline) |
| Task | Binary classification — 30-day hospital readmission |
| Status | **NOT YET VALIDATED** — hackathon prototype |

## Intended Use

PREP is intended as a **decision-support tool** for healthcare teams to identify patients who may be at higher risk of 30-day readmission at discharge.

PREP is **NOT** intended to:
- Diagnose patients
- Prescribe treatment
- Replace clinical judgment
- Make autonomous patient-care decisions
- Guarantee readmission or prevention

A clinician remains responsible for interpreting any output.

## Training Data

| Mode | Dataset | Notes |
|---|---|---|
| Primary | MIMIC-IV | Requires PhysioNet credentialing. Not included. |
| Fallback | UCI Diabetes 130-US | Diabetic patients only — not hospital-wide |
| Demo | Synthetic | Not real patient records |

## Features

| Feature | Type | Notes |
|---|---|---|
| prior_admissions_12mo | Numeric | Admissions in prior 12 months |
| los_days | Numeric | Length of stay |
| n_diagnoses | Numeric | ICD diagnosis count |
| n_procedures | Numeric | Procedure count |
| charlson_index | Numeric | Charlson Comorbidity Index |
| n_medications | Numeric | Unique medication count |
| high_risk_med | Binary | High-risk medication flag |
| emergency_adm | Binary | Emergency/urgent admission |
| age | Numeric | Age at discharge |
| gender | Binary | Encoded M=1 |
| insurance | Categorical | One-hot (Medicare/Medicaid/Private/Other) |

**Race/ethnicity: EXCLUDED from model. Used for fairness auditing only.**

## Performance

**NOT YET VALIDATED on real patient data.** Saved demo training results, when
present, are based on synthetic patients and a synthetic label; they are not
clinical performance estimates.

Reference results (NOT PREP results — from Adisa 2026 on MIMIC-IV):

| Model | AUC-ROC | Brier |
|---|---|---|
| Logistic Regression | 0.675 | 0.224 |
| XGBoost | 0.696 | 0.217 |
| LightGBM | 0.689 | 0.146 |

## Fairness

Audited across: Race/Ethnicity, Gender, Age Group, Insurance.
Limits: AUC gap ≤ 0.05, FNR gap ≤ 0.10.
Groups with < 30 readmission events: INSUFFICIENT EVENTS.
Fairness status is INDETERMINATE if no evaluable subgroup comparisons exist or
any observed subgroup is below the event minimum; unavailable evidence is never
reported as PASS.

## Explainability

Compatible trained tree models use SHAP TreeExplainer for raw model outputs.
The deterministic demo scorer uses heuristic contributions, which are labeled
as approximations and are not represented as SHAP. If SHAP fails for a trained
model, any fallback heuristic is explicitly labeled as not explaining that
trained model.

## Limitations

1. Not clinically validated
2. Demo mode uses synthetic scoring
3. UCI fallback is diabetic patients only
4. No prospective evaluation performed
5. Calibration not yet validated on real data
6. Threshold not yet validated on real data

## Ethical Considerations

- Race/ethnicity excluded from model features
- Fairness audited across demographic subgroups
- All predictions logged (hashed, no PII)
- Clinician retains decision authority
- Prototype status prominently disclosed

## Citation

If referencing the methodology foundation:
> Adisa, I.T. (2026). arXiv:2604.22535
