# PREP
## Predict Readmission Estimation of Patient

> A healthcare AI decision-support prototype that estimates 30-day hospital readmission risk at discharge — with explainability, fairness auditing, and drift monitoring.

**Research / Hackathon Prototype — Not Clinically Validated**

---

## Problem

Hospital readmissions within 30 days affect millions of patients. Existing prediction tools have three critical gaps:
1. Black-box predictions with no explanation
2. No fairness evaluation across demographic groups
3. No mechanism to detect model degradation over time

## Solution: Four Pillars

| Pillar | What it does |
|---|---|
| **PREDICT** | Estimate 30-day readmission risk at discharge |
| **EXPLAIN** | Show SHAP values for compatible trained trees or a labeled demo heuristic |
| **AUDIT** | Check fairness across demographic subgroups |
| **MONITOR** | Detect model and data drift over time |

---

## Architecture

```
Patient Data → Feature Engineering → ML Model → Calibration → FastAPI
                                                                  ↓
                                              React Dashboard ←───┤
                                                                  ↓
                                              Prediction + Explanation + Fairness + Monitoring
```

See [docs/architecture.md](docs/architecture.md) for full diagram.

---

## Features

- **Patient Assessment** — form-based risk estimation with SHAP explanation
- **Cohort View** — searchable, sortable table with risk filters
- **Explainability** — per-patient SHAP values for compatible trained tree models; demo-mode contributions are explicitly labeled heuristic approximations
- **Fairness Audit** — subgroup AUC, FNR, PPV across race, gender, age, insurance
- **Drift Monitoring** — KL divergence + 2σ prediction drift alert
- **Model Registry** — version tracking, calibration, threshold metadata
- **Demo Mode** — fully synthetic data, no real patient records required

---

## Dataset

| Mode | Dataset | Notes |
|---|---|---|
| Primary | MIMIC-IV | Requires PhysioNet credentialing |
| Fallback | UCI Diabetes 130-US | Diabetic patients only |
| Demo | Synthetic | Hackathon demonstration |

**MIMIC-IV data is never committed to this repository.**

---

## ML Methodology

- Chronological 70/15/15 row split (no random splitting); each patient is retained only in their earliest split, so later admissions crossing a split boundary are excluded
- Logistic Regression baseline + LightGBM
- Logistic recalibration on validation set
- Threshold selection: Youden's J on **validation set only**
- SHAP TreeExplainer for compatible trained tree models; heuristic demo explanations are not SHAP
- Race/ethnicity excluded from model features (fairness audit only)

See [docs/model_card.md](docs/model_card.md) for the current status and limitations.

---

## Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### Quick start (demo mode)

```bash
# Terminal 1 — generate demo data + start backend
python run.py all

# Terminal 2 — start frontend
cd frontend
npm run dev
```

Open: http://localhost:5173

### Backend only

```bash
python run.py backend
# API docs: http://localhost:8000/docs
```

### Generate demo data only

```bash
python run.py demo
```

### Run tests

```bash
python run.py test
# or
pytest tests/ -v
```

---

## Public Demo Deployment

`render.yaml` defines a FastAPI service and a static frontend. To deploy them
from GitHub, create a Blueprint in Render and connect this repository. Render
will read the blueprint and build both services. Once deployment completes,
open the `prep-frontend-tenq` service URL.

The blueprint sets `VITE_API_URL` on the frontend and `CORS_ORIGINS` on the API
to connect the two services. If you change either service name, update the
matching URL in `render.yaml` and redeploy. The free API service may take a
short time to wake after inactivity.

This is a public research prototype. Use synthetic demo data only; do not enter
real patient information.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | System health |
| GET | /v1/model | Model metadata |
| GET | /v1/fairness | Saved training fairness audit (when available) |
| POST | /v1/predict | Risk estimate |
| POST | /v1/explain | SHAP output or explicitly labeled heuristic explanation |
| GET | /metrics | Prediction log |
| POST | /v1/drift/check | Drift detection |

---

## Demo Mode

The full application runs without MIMIC-IV data using synthetic patients.

```bash
python demo/generate_demo_data.py
```

Generates:
- `synthetic_patients.csv` — 4 named demo cases
- `synthetic_cohort.csv` — 200 synthetic patients
- `drift_demo.csv` — reference + drifted period data

All synthetic data is clearly labelled. It is not real patient data.

---

## Limitations

- Not clinically validated
- Demo mode uses synthetic scoring, not a trained model
- Demo explanation contributions are heuristic approximations, not SHAP values
- Demo audit and training-report values are synthetic pipeline outputs, not real-data validation
- UCI fallback is diabetic patients only — not hospital-wide
- No prospective evaluation performed
- Prototype status — not a medical device

---

## Ethics & Privacy

- Race/ethnicity excluded from model features
- Fairness audited across demographic subgroups
- No patient-level data committed to repository
- Predictions logged with hashed inputs only (no PII)
- Clinician retains decision authority

---

## Reference

This project is methodologically inspired by:

> Adisa, I.T. (2026). An Integrated Framework for Explainable, Fair, and Observable Hospital Readmission Prediction: Development and Validation on MIMIC-IV. arXiv:2604.22535

Reference repository: https://github.com/Tomisin92/readmission-prediction

**Results from the reference study are NOT PREP results.** PREP is an independent implementation with methodological improvements.

---

## Future Work

- Train on MIMIC-IV and report validated metrics
- Isotonic calibration comparison
- Decision curve analysis UI
- FHIR-compatible API
- Prospective clinical evaluation

---

*PREP is a research/hackathon prototype. It is not clinically validated and must not be used to make patient-care decisions.*
