"""
PREP Data Loading
Loads a cohort dataframe from one of three sources, in priority order.

| Mode  | Source                              | Status |
|-------|-------------------------------------|--------|
| demo  | `demo/synthetic_cohort.csv`          | Available — synthetic data |
| uci   | UCI Diabetes 130-US (`data/`)        | Available if you download it — diabetic patients only |
| mimic | MIMIC-IV (`data/mimic-iv/`)          | Requires PhysioNet credentialing; not bundled |

This loader never fabricates data and never invents labels. If a source is
missing it raises with an actionable message instead of quietly substituting
something else.

Column mapping gaps are reported honestly via :func:`describe_cohort` — for
example the UCI dataset cannot support a Charlson index or a high-risk
medication flag, so those are set to 0 and listed in ``degraded_features``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_COHORT = ROOT / "demo" / "synthetic_cohort.csv"
UCI_PATH = ROOT / "data" / "uci_diabetes_130us.csv"
MIMIC_DIR = ROOT / "data" / "mimic-iv"

TARGET = "readmitted_30d"
AUDIT_COLUMNS = ["race", "gender", "age_group", "insurance"]

# Columns the PREP feature pipeline expects, plus the label.
EXPECTED_COLUMNS = [
    "age", "gender", "insurance",
    "prior_admissions_12mo", "los_days", "n_diagnoses", "n_procedures",
    "charlson_index", "n_medications", "high_risk_med", "emergency_adm",
]

# MIMIC-IV hosp/ tables the cohort builder needs (informational).
MIMIC_REQUIRED_TABLES = [
    "admissions.csv", "patients.csv", "diagnoses_icd.csv",
    "procedures_icd.csv", "prescriptions.csv",
]


def age_group(age: float) -> str:
    """PREP age bands, matching demo/generate_demo_data.py."""
    if age <= 50:
        return "18-50"
    if age <= 65:
        return "51-65"
    if age <= 75:
        return "66-75"
    if age <= 85:
        return "76-85"
    return "85+"


def load_demo_cohort(path: Path | str | None = None) -> pd.DataFrame:
    """Load the synthetic demo cohort.

    This file has no outcome label: it is a demonstration cohort. Use
    :func:`add_synthetic_label` explicitly if you need a labelled frame for
    pipeline plumbing (the result is SIMULATED, not evidence).
    """
    path = Path(path) if path else DEMO_COHORT
    if not path.exists():
        raise FileNotFoundError(
            f"Demo cohort not found at {path}. Generate it first:\n"
            f"    py -3.14 demo/generate_demo_data.py"
        )
    df = pd.read_csv(path)
    if "age_group" not in df.columns and "age" in df.columns:
        df["age_group"] = df["age"].map(age_group)
    df.attrs["source"] = "demo"
    df.attrs["dataset_label"] = "SIMULATED — synthetic data, not real patients"
    return df


def add_synthetic_label(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach a SIMULATED binary outcome for pipeline plumbing only.

    The label is a Bernoulli draw from the synthetic risk score, so a model
    trained on it is predicting the demo generator's own formula. That is
    useful to prove the training path works end to end; it is NOT evidence of
    clinical performance and must never be reported as a PREP result.
    """
    if "estimated_risk" not in df.columns:
        raise ValueError("add_synthetic_label requires an 'estimated_risk' column")
    rng = np.random.default_rng(seed)
    df = df.copy()
    df[TARGET] = (rng.random(len(df)) < df["estimated_risk"].values).astype(int)
    df.attrs["dataset_label"] = (
        "SIMULATED — synthetic data with a synthetic outcome label"
    )
    return df


# UCI 130-US payer codes → PREP insurance categories (best-effort mapping).
UCI_PAYER_MAP = {
    "MC": "Medicare",
    "MD": "Medicaid",
    "HM": "Private",
    "SP": "Other", "UN": "Other", "CP": "Other", "SI": "Other",
    "DM": "Other", "CM": "Other", "OG": "Other", "PO": "Other",
    "WC": "Other", "OT": "Other", "FR": "Other", "NULL": "Other",
}


def _uci_age_to_int(value) -> float:
    """Convert a UCI bracket such as ``[70-80)`` to a midpoint integer age."""
    text = str(value).strip()
    digits = "".join(ch if (ch.isdigit() or ch == "-") else " " for ch in text)
    parts = [p for p in digits.replace("-", " ").split() if p.isdigit()]
    if not parts:
        return np.nan
    nums = [int(p) for p in parts]
    return float(sum(nums) / len(nums))


def load_uci_cohort(
    path: Path | str | None = None,
    require_label: bool = True,
) -> pd.DataFrame:
    """Map the UCI Diabetes 130-US dataset onto the PREP schema.

    Scope: diabetic patients only — this is a fallback used to exercise the
    pipeline, not a hospital-wide readmission model. Features the UCI schema
    cannot support (Charlson index, high-risk medication flag) are set to 0 and
    reported in ``df.attrs["degraded_features"]``.
    """
    path = Path(path) if path else UCI_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"UCI cohort not found at {path}.\n"
            "Download 'Diabetes 130-US hospitals for years 1999-2008' and save "
            "it there as uci_diabetes_130us.csv."
        )

    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=raw.index)

    out["subject_id"] = raw.get("patient_nbr", pd.Series(range(len(raw))))
    out["age"] = raw["age"].map(_uci_age_to_int) if "age" in raw else np.nan
    out["gender"] = (
        raw["gender"].map({"Male": "M", "Female": "F"}).fillna("F")
        if "gender" in raw else "F"
    )
    out["insurance"] = (
        raw["payer_code"].astype(str).str.upper().str.strip()
        .map(UCI_PAYER_MAP).fillna("Other")
        if "payer_code" in raw else "Other"
    )
    out["race"] = (
        raw["race"].fillna("Other/Unknown").replace({"?": "Other/Unknown"})
        if "race" in raw else "Other/Unknown"
    )

    out["prior_admissions_12mo"] = raw.get("number_inpatient", 0)
    out["los_days"] = raw.get("time_in_hospital", np.nan)
    out["n_diagnoses"] = raw.get("number_diagnoses", 0)
    out["n_procedures"] = raw.get("num_procedures", 0)
    out["n_medications"] = raw.get("num_medications", 0)
    # Not derivable from this schema — declared, not invented.
    out["charlson_index"] = 0
    out["high_risk_med"] = 0
    out["emergency_adm"] = (
        raw["admission_type_id"].isin([1, 2]).astype(int)  # Emergency / Urgent
        if "admission_type_id" in raw else 0
    )

    if "readmitted" in raw:
        out[TARGET] = (raw["readmitted"] == "<30").astype(int)
    elif require_label:
        raise ValueError(
            "UCI file has no 'readmitted' column — cannot build a labelled cohort"
        )

    out["age_group"] = out["age"].map(age_group)
    out.attrs["source"] = "uci"
    out.attrs["dataset_label"] = (
        "UCI Diabetes 130-US — diabetic inpatients only, not hospital-wide"
    )
    out.attrs["degraded_features"] = ["charlson_index", "high_risk_med"]
    return out


def load_mimic_cohort(mimic_dir: Path | str | None = None) -> pd.DataFrame:
    """MIMIC-IV is the primary dataset but requires PhysioNet credentialing.

    The credentialed data is never committed to this repository, so this
    function only reports what is missing. The cohort builder itself (timed
    readmission label, prior-admission counts, Charlson mapping, lab features)
    is described in REFERENCE_ARCHITECTURE.md and is out of scope for the
    prototype until credentialed access exists.
    """
    mimic_dir = Path(mimic_dir) if mimic_dir else MIMIC_DIR
    missing = [t for t in MIMIC_REQUIRED_TABLES if not (mimic_dir / t).exists()]

    if missing:
        raise FileNotFoundError(
            f"MIMIC-IV not available at {mimic_dir} (missing: {', '.join(missing)}).\n"
            "MIMIC-IV requires PhysioNet credentialing + a data use agreement:\n"
            "    https://physionet.org/content/mimiciv/\n"
            "Use demo or UCI mode until credentialed access exists."
        )

    raise NotImplementedError(
        "MIMIC-IV tables found, but the credentialed cohort builder is not part "
        "of this prototype. See REFERENCE_ARCHITECTURE.md for the pipeline it "
        "would implement. Use source='demo' or source='uci'."
    )


def load_cohort(source: str = "demo", **kwargs) -> pd.DataFrame:
    """Load a cohort by source name: 'demo', 'uci', or 'mimic'."""
    if source == "demo":
        return load_demo_cohort(**kwargs)
    if source == "uci":
        return load_uci_cohort(**kwargs)
    if source == "mimic":
        return load_mimic_cohort(**kwargs)
    raise ValueError("source must be one of: demo, uci, mimic")


def describe_cohort(df: pd.DataFrame) -> dict:
    """Summarise a loaded cohort: size, label balance, mapping gaps."""
    target = df[TARGET] if TARGET in df.columns else None
    return {
        "n_rows": int(len(df)),
        "source": df.attrs.get("source", "unknown"),
        "dataset_label": df.attrs.get("dataset_label", "unlabelled"),
        "has_label": target is not None,
        "n_positive": int(target.sum()) if target is not None else None,
        "prevalence": round(float(target.mean()), 4) if target is not None else None,
        "degraded_features": df.attrs.get("degraded_features", []),
        "missing_expected": [
            c for c in EXPECTED_COLUMNS if c not in df.columns
        ],
        "unusable_audit_columns": [
            c for c in AUDIT_COLUMNS if c not in df.columns
        ],
    }
