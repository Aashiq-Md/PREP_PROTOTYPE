"""
PREP Feature Pipeline
Defines the canonical feature set, leakage checks, and preprocessing.
"""

import numpy as np
import pandas as pd

# ── Canonical PREP feature set ────────────────────────────────────────────────
# Race/ethnicity is EXCLUDED from model features (fairness audit only)
MODEL_FEATURES = [
    "prior_admissions_12mo",
    "los_days",
    "n_diagnoses",
    "n_procedures",
    "charlson_index",
    "n_medications",
    "high_risk_med",
    "emergency_adm",
    "age",
    "gender_enc",
    "insurance_Medicare",
    "insurance_Medicaid",
    "insurance_Private",
]

NUMERIC_FEATURES = [
    "prior_admissions_12mo", "los_days", "n_diagnoses",
    "n_procedures", "charlson_index", "n_medications", "age",
]

BINARY_FEATURES = ["high_risk_med", "emergency_adm", "gender_enc"]

CATEGORICAL_FEATURES = ["insurance"]

TARGET = "readmitted_30d"

# Features that must NOT appear in the model (leakage / fairness audit only)
AUDIT_ONLY_FEATURES = ["race", "race_simple", "race_enc"]

# Post-discharge features that would cause leakage
LEAKAGE_FEATURES = [
    "next_admittime", "days_to_next", "readmitted_30d",
    "discharge_location", "deathtime",
]


def leakage_check(df: pd.DataFrame) -> list[str]:
    """Return list of leakage-risk columns found in dataframe."""
    found = [c for c in LEAKAGE_FEATURES if c in df.columns]
    race_in_model = [c for c in AUDIT_ONLY_FEATURES if c in df.columns]
    return found + race_in_model


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into model-ready columns.
    Operates on a copy; does not modify the input.
    """
    df = df.copy()

    # Gender
    if "gender" in df.columns:
        df["gender_enc"] = (df["gender"].str.upper() == "M").astype(int)

    # Insurance one-hot (reference category = Other)
    if "insurance" in df.columns:
        ins = df["insurance"].fillna("Other")
        df["insurance_Medicare"] = (ins == "Medicare").astype(int)
        df["insurance_Medicaid"] = (ins == "Medicaid").astype(int)
        df["insurance_Private"] = (ins == "Private").astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the model feature matrix from a cohort dataframe.
    Returns (X, feature_cols).
    """
    df = encode_features(df)
    available = [c for c in MODEL_FEATURES if c in df.columns]
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] Missing features (will be zero-filled): {missing}")
        for c in missing:
            df[c] = 0
        available = MODEL_FEATURES

    X = df[available].copy()
    return X, available


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "admittime",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by admission time: 70% train / 15% val / 15% test.
    Keeps all rows for a patient in their earliest chronological split.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    partitions = [
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    ]
    patient_col = next(
        (column for column in ("subject_id", "patient_id") if column in df.columns),
        None,
    )
    if patient_col is not None:
        seen_ids = set()
        isolated = []
        for partition in partitions:
            unseen = ~partition[patient_col].isin(seen_ids)
            isolated.append(partition.loc[unseen].copy())
            seen_ids.update(partition.loc[unseen, patient_col].dropna())
        removed = sum(
            len(before) - len(after)
            for before, after in zip(partitions, isolated)
        )
        if removed:
            print(
                f"[INFO] Removed {removed} later admissions whose patients "
                "were already assigned to an earlier split."
            )
        partitions = isolated

    train, val, test = partitions

    print(
        f"[INFO] Split: train={len(train):,} | val={len(val):,} | test={len(test):,}"
    )
    return train, val, test


def impute_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Median imputation fitted on train only."""
    medians = {
        c: float(X_train[c].median()) for c in numeric_cols if c in X_train.columns
    }
    for df in [X_train, X_val, X_test]:
        for c, med in medians.items():
            if c in df.columns:
                df[c] = df[c].fillna(med if not np.isnan(med) else 0)
    return X_train, X_val, X_test, medians
