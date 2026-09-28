import { PageHeader, Disclaimer } from '../components/UI'

const SECTIONS = [
  {
    title: 'Problem Statement',
    content: `Hospital readmissions within 30 days represent a significant clinical and financial burden. Existing prediction tools often operate as black boxes, lack fairness evaluation, and provide no mechanism for detecting model degradation over time. PREP addresses these gaps.`,
  },
  {
    title: 'Data Sources',
    content: `Primary: MIMIC-IV (requires PhysioNet credentialing — not included in this repository).
Fallback: UCI Diabetes 130-US Hospitals dataset (diabetic patients only — cannot support hospital-wide claims).
Demo: Synthetic data generated for hackathon demonstration.`,
  },
  {
    title: 'Feature Set',
    content: `Model features (11 primary):
• prior_admissions_12mo — admissions in the 12 months before this admission
• los_days — length of stay in days
• n_diagnoses — number of ICD diagnoses
• n_procedures — number of procedures
• charlson_index — Charlson Comorbidity Index (ICD-10 mapping)
• n_medications — unique medication count
• high_risk_med — flag for high-risk medications (warfarin, insulin, opioids, etc.)
• emergency_adm — emergency or urgent admission flag
• age — age at discharge
• gender — binary encoded
• insurance — one-hot encoded (Medicare / Medicaid / Private / Other)

Race/ethnicity: EXCLUDED from model features. Used for fairness auditing only.`,
  },
  {
    title: 'Leakage Prevention',
    content: `Every feature must be computable from information available at discharge. Post-discharge information (future admissions, future diagnoses, future labs) is explicitly excluded. Automated leakage checks verify no post-discharge columns are present in the feature matrix.`,
  },
  {
    title: 'Chronological Split',
    content: `70% earliest admissions → TRAIN
15% next period → VALIDATION
15% latest period → TEST

Patient-level deduplication: patients appearing in train are removed from test to prevent leakage. This is a key improvement over random splitting.`,
  },
  {
    title: 'Models',
    content: `1. Logistic Regression — interpretable baseline, L2 regularization, class-balanced
2. LightGBM — primary explainable tree model, SHAP TreeExplainer compatible
3. XGBoost — optional comparison model

Model selection: the more complex model is only retained if it provides meaningful improvement and passes calibration and fairness criteria.`,
  },
  {
    title: 'Calibration',
    content: `Raw classifier probabilities are recalibrated using logistic recalibration on the validation set. Reported metrics: calibration slope, calibration intercept, Brier score, and base-rate Brier comparison (prevalence × (1 − prevalence)).`,
  },
  {
    title: 'Threshold Selection',
    content: `Threshold is selected using Youden's J statistic on the VALIDATION set only. The test set is never used for threshold selection. This is a critical methodological requirement of the PREP specification.`,
  },
  {
    title: 'Explainability',
    content: `SHAP TreeExplainer is applied to the LightGBM model. Per-patient explanations show: base value + feature contributions = model output. Additivity is verified (tolerance 1e-4). Global importance is computed as mean |SHAP| across the test set.`,
  },
  {
    title: 'Fairness',
    content: `Subgroup audit across: Race/Ethnicity, Gender, Age Group, Insurance.
Metrics: AUC-ROC, FNR, PPV, flag rate, observed vs predicted risk.
Limits: AUC gap ≤ 0.05, FNR gap ≤ 0.10.
Groups with fewer than 30 readmission events: INSUFFICIENT EVENTS — excluded from pass/fail.`,
  },
  {
    title: 'Drift Monitoring',
    content: `Feature drift: KL divergence using reference decile bins (numeric) or category proportions (categorical). Alert threshold: KL > 0.05.
Prediction drift: alert if current mean predicted risk deviates > 2σ from reference period mean.`,
  },
  {
    title: 'Limitations',
    content: `• PREP has not been trained on real patient data in this prototype.
• Demo predictions use synthetic scoring — not a trained model.
• UCI fallback represents diabetic patients only.
• No clinical validation has been performed.
• This is a hackathon prototype, not a medical device.
• A clinician remains responsible for interpreting any output.`,
  },
]

export default function Methodology() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <PageHeader title="Methodology" subtitle="ML methodology, design decisions, and PREP specification." />

      <Disclaimer text="PREP is a research/hackathon prototype. Methodology descriptions reflect the PREP specification. Items marked PROPOSED are design choices not yet validated. Items marked SIMULATED are demonstration-only." />

      <div className="space-y-4 mt-5">
        {SECTIONS.map(s => (
          <div key={s.title} className="card">
            <p className="text-sm font-semibold text-blue-300 mb-2">{s.title}</p>
            <pre className="text-xs text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">{s.content}</pre>
          </div>
        ))}
      </div>
    </div>
  )
}
