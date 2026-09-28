# PREP Architecture

## System Overview

```
                 ┌─────────────────────┐
                 │ Hospital / Demo Data │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ + Leakage Checks    │
                 └──────────┬──────────┘
                            ↓
              ┌───────────────────────────┐
              │     ML Model Pipeline     │
              │                           │
              │ Logistic │ LightGBM       │
              └────────────┬──────────────┘
                           ↓
                 ┌─────────────────────┐
                 │ Calibration +       │
                 │ Threshold Selection │
                 │ (validation set)    │
                 └──────────┬──────────┘
                            ↓
                  ┌──────────────────┐
                  │ FastAPI Backend  │
                  └────────┬─────────┘
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Prediction          Explainability     Monitoring
   /v1/predict         /v1/explain        /v1/drift
        ↓                  ↓                  ↓
        └──────────────┬───┴──────────────────┘
                       ↓
                React Dashboard
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Clinician       Fairness         Model Registry
   Interface        Audit
```

## Directory Structure

```
PREP/
├── frontend/          React + Vite + Tailwind
│   └── src/
│       ├── pages/     Dashboard, Assessment, Cohort, Explain, Fairness, Monitoring, Registry
│       ├── charts/    ShapChart, RiskGauge
│       ├── components/ Layout, UI primitives
│       └── services/  API client
│
├── backend/           FastAPI
│   └── app/
│       ├── main.py    Routes
│       ├── schemas/   Pydantic models
│       ├── services/  PREPPredictor
│       └── monitoring/ Drift detection
│
├── ml/                ML pipeline
│   ├── features/      Feature pipeline, leakage checks
│   ├── training/      LogReg + LightGBM training
│   ├── evaluation/    Metrics, calibration, net benefit
│   ├── explainability/ SHAP
│   └── fairness/      Subgroup audit
│
├── demo/              Synthetic data generation
├── models/            Trained model artifacts (gitignored)
├── tests/             pytest test suite
└── docs/              Architecture, methodology, model card
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | System health check |
| GET | /v1/model | Model metadata |
| GET | /v1/fairness | Saved audit report; indeterminate when evidence is unavailable |
| POST | /v1/predict | Generate risk estimate |
| POST | /v1/explain | SHAP explanation |
| GET | /metrics | Prediction log summary |
| GET | /v1/demo/patients | Synthetic demo patients |
| GET | /v1/demo/cohort | Synthetic cohort |
| POST | /v1/drift/check | Drift detection |

## Data Flow

1. Patient data entered at discharge
2. Features encoded (gender binary, insurance one-hot)
3. Race/ethnicity excluded from model input
4. LightGBM (or demo scorer) produces raw probability
5. Probability recalibrated (logistic recalibration)
6. Threshold applied (Youden's J from validation set)
7. Compatible trained tree models use SHAP TreeExplainer; demo scoring returns heuristic contributions labeled as non-SHAP
8. Result returned with risk band, threshold status, explanation

The demo cohort, labels, scores, training metrics, and drift data are synthetic.
They are plumbing demonstrations only; no clinical or real-data validation is
claimed. Patient rows whose subject IDs cross a chronological split boundary
are retained only in their earliest split.
