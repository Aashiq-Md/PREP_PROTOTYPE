"""
PREP Demo Data Generator
Generates synthetic patients for hackathon demonstration.
These are NOT real patient records.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent

FEATURE_COLS = [
    "age", "gender", "insurance",
    "prior_admissions_12mo", "los_days", "n_diagnoses",
    "n_procedures", "charlson_index", "n_medications",
    "high_risk_med", "emergency_adm",
]

INSURANCE_OPTS = ["Medicare", "Medicaid", "Private", "Other"]
GENDER_OPTS = ["M", "F"]
RACE_OPTS = ["White", "Black/AA", "Hispanic", "Asian", "Other/Unknown"]
AGE_GROUPS = ["18-50", "51-65", "66-75", "76-85", "85+"]


def age_group(age):
    if age <= 50: return "18-50"
    if age <= 65: return "51-65"
    if age <= 75: return "66-75"
    if age <= 85: return "76-85"
    return "85+"


def synthetic_risk(row):
    """Deterministic risk score for synthetic patients (NOT a clinical model)."""
    score = 0.15
    score += row["prior_admissions_12mo"] * 0.08
    score += min(row["los_days"], 30) * 0.005
    score += row["charlson_index"] * 0.015
    score += row["n_medications"] * 0.004
    score += row["high_risk_med"] * 0.06
    score += row["emergency_adm"] * 0.03
    score += max(0, row["age"] - 65) * 0.002
    return float(np.clip(score, 0.02, 0.95))


def risk_band(risk):
    if risk < 0.20: return "Lower"
    if risk < 0.40: return "Moderate"
    return "Higher"


# ── Four named demo patients ──────────────────────────────────────────────────
DEMO_PATIENTS = [
    {
        "patient_id": "DEMO-001",
        "label": "Lower Risk",
        "age": 45, "gender": "F", "insurance": "Private",
        "prior_admissions_12mo": 0, "los_days": 2, "n_diagnoses": 3,
        "n_procedures": 1, "charlson_index": 0, "n_medications": 2,
        "high_risk_med": 0, "emergency_adm": 0,
        "race": "White",
    },
    {
        "patient_id": "DEMO-002",
        "label": "Moderate Risk",
        "age": 67, "gender": "M", "insurance": "Medicare",
        "prior_admissions_12mo": 1, "los_days": 5, "n_diagnoses": 8,
        "n_procedures": 3, "charlson_index": 3, "n_medications": 7,
        "high_risk_med": 0, "emergency_adm": 1,
        "race": "Black/AA",
    },
    {
        "patient_id": "DEMO-003",
        "label": "Higher Risk",
        "age": 78, "gender": "M", "insurance": "Medicaid",
        "prior_admissions_12mo": 3, "los_days": 12, "n_diagnoses": 14,
        "n_procedures": 6, "charlson_index": 7, "n_medications": 15,
        "high_risk_med": 1, "emergency_adm": 1,
        "race": "Hispanic",
    },
    {
        "patient_id": "DEMO-004",
        "label": "Strong Explainability Example",
        "age": 72, "gender": "F", "insurance": "Medicare",
        "prior_admissions_12mo": 2, "los_days": 8, "n_diagnoses": 11,
        "n_procedures": 4, "charlson_index": 5, "n_medications": 12,
        "high_risk_med": 1, "emergency_adm": 0,
        "race": "Asian",
    },
]

for p in DEMO_PATIENTS:
    p["estimated_risk"] = synthetic_risk(p)
    p["risk_band"] = risk_band(p["estimated_risk"])
    p["age_group"] = age_group(p["age"])


# ── Synthetic cohort (200 patients) ──────────────────────────────────────────
def make_cohort(n=200):
    rows = []
    for i in range(n):
        age = int(RNG.integers(25, 90))
        prior = int(RNG.integers(0, 5))
        los = float(round(RNG.exponential(4) + 1, 1))
        n_diag = int(RNG.integers(1, 20))
        n_proc = int(RNG.integers(0, 10))
        cci = int(RNG.integers(0, 12))
        n_meds = int(RNG.integers(0, 20))
        hrm = int(RNG.random() < 0.3)
        emerg = int(RNG.random() < 0.5)
        gender = RNG.choice(GENDER_OPTS)
        insurance = RNG.choice(INSURANCE_OPTS)
        race = RNG.choice(RACE_OPTS)

        row = {
            "patient_id": f"SYN-{i+1:04d}",
            "age": age, "gender": gender, "insurance": insurance,
            "race": race, "age_group": age_group(age),
            "prior_admissions_12mo": prior, "los_days": los,
            "n_diagnoses": n_diag, "n_procedures": n_proc,
            "charlson_index": cci, "n_medications": n_meds,
            "high_risk_med": hrm, "emergency_adm": emerg,
        }
        row["estimated_risk"] = synthetic_risk(row)
        row["risk_band"] = risk_band(row["estimated_risk"])
        row["above_threshold"] = int(row["estimated_risk"] >= 0.30)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ordered admission timestamps so the ML pipeline's chronological split has
    # something to sort on. Synthetic dates — not real admission dates.
    df["admittime"] = pd.date_range("2023-01-01", periods=len(df), freq="1D")
    return df


# ── Reference drift data ──────────────────────────────────────────────────────
def make_drift_data(n=300):
    """Reference period data for drift demonstration."""
    rows = []
    for i in range(n):
        age = int(RNG.integers(30, 85))
        prior = int(RNG.integers(0, 4))
        los = float(round(RNG.exponential(4) + 1, 1))
        n_meds = int(RNG.integers(1, 15))
        cci = int(RNG.integers(0, 8))
        row = {
            "patient_id": f"REF-{i+1:04d}",
            "age": age,
            "prior_admissions_12mo": prior,
            "los_days": los,
            "n_medications": n_meds,
            "charlson_index": cci,
            "high_risk_med": int(RNG.random() < 0.25),
            "emergency_adm": int(RNG.random() < 0.45),
            "n_diagnoses": int(RNG.integers(2, 15)),
            "n_procedures": int(RNG.integers(0, 8)),
            "gender": RNG.choice(GENDER_OPTS),
            "insurance": RNG.choice(INSURANCE_OPTS),
            "race": RNG.choice(RACE_OPTS),
            "period": "reference",
        }
        row["estimated_risk"] = synthetic_risk(row)
        rows.append(row)

    # Drifted period: older patients, more prior admissions, more meds
    for i in range(n):
        age = int(RNG.integers(60, 95))          # shifted older
        prior = int(RNG.integers(2, 7))           # more prior admissions
        los = float(round(RNG.exponential(7) + 2, 1))  # longer stays
        n_meds = int(RNG.integers(8, 22))         # more medications
        cci = int(RNG.integers(4, 14))
        row = {
            "patient_id": f"DRF-{i+1:04d}",
            "age": age,
            "prior_admissions_12mo": prior,
            "los_days": los,
            "n_medications": n_meds,
            "charlson_index": cci,
            "high_risk_med": int(RNG.random() < 0.55),
            "emergency_adm": int(RNG.random() < 0.65),
            "n_diagnoses": int(RNG.integers(8, 22)),
            "n_procedures": int(RNG.integers(2, 12)),
            "gender": RNG.choice(GENDER_OPTS),
            "insurance": RNG.choice(INSURANCE_OPTS),
            "race": RNG.choice(RACE_OPTS),
            "period": "drifted",
        }
        row["estimated_risk"] = synthetic_risk(row)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Save demo patients
    pd.DataFrame(DEMO_PATIENTS).to_csv(OUT / "synthetic_patients.csv", index=False)
    with open(OUT / "demo_patients.json", "w") as f:
        json.dump(DEMO_PATIENTS, f, indent=2)
    print(f"[DONE] synthetic_patients.csv — {len(DEMO_PATIENTS)} demo patients")

    # Save cohort
    cohort = make_cohort(200)
    cohort.to_csv(OUT / "synthetic_cohort.csv", index=False)
    print(f"[DONE] synthetic_cohort.csv — {len(cohort)} patients")

    # Save drift data
    drift = make_drift_data(300)
    drift.to_csv(OUT / "drift_demo.csv", index=False)
    print(f"[DONE] drift_demo.csv — {len(drift)} rows (reference + drifted)")

    print("\nAll demo data is SYNTHETIC — not real patient records.")
