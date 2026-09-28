import { PageHeader, Disclaimer } from '../components/UI'
import { FlaskConical, Github, BookOpen } from 'lucide-react'

export default function About() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <PageHeader title="About PREP" />

      <div className="card mb-4">
        <div className="flex items-center gap-3 mb-4">
          <FlaskConical size={24} className="text-blue-400" />
          <div>
            <h2 className="text-lg font-bold text-white">PREP</h2>
            <p className="text-sm text-slate-400">Predict Readmission Estimation of Patient</p>
          </div>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">
          PREP is a healthcare AI decision-support prototype that estimates the risk of a patient's 30-day hospital readmission at discharge, explains the factors contributing to that estimate, audits model fairness across demographic groups, and monitors model and data drift over time.
        </p>
        <p className="text-sm text-slate-400 mt-3 leading-relaxed">
          PREP moves beyond prediction by making the model explainable, auditable, and observable — addressing the three main barriers to clinical AI adoption: black-box predictions, no deployment reliability infrastructure, and inadequate fairness evaluation.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="card">
          <p className="section-title mb-3">Four Pillars</p>
          <div className="space-y-2">
            {[
              ['PREDICT', 'Estimate 30-day readmission risk at discharge', 'text-blue-300'],
              ['EXPLAIN', 'Show which features drove the estimate (SHAP)', 'text-violet-300'],
              ['AUDIT', 'Check fairness across demographic subgroups', 'text-emerald-300'],
              ['MONITOR', 'Detect model and data drift over time', 'text-amber-300'],
            ].map(([label, desc, color]) => (
              <div key={label} className="flex gap-3">
                <span className={`text-xs font-bold w-16 flex-shrink-0 ${color}`}>{label}</span>
                <span className="text-xs text-slate-400">{desc}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <p className="section-title mb-3">Technology Stack</p>
          <div className="space-y-1 text-xs text-slate-400">
            {[
              ['ML', 'scikit-learn, LightGBM, XGBoost, SHAP'],
              ['Backend', 'Python, FastAPI, Pydantic'],
              ['Frontend', 'React, Vite, Tailwind CSS, Recharts'],
              ['Data', 'MIMIC-IV / UCI / Synthetic Demo'],
            ].map(([label, value]) => (
              <div key={label} className="flex gap-2">
                <span className="text-slate-500 w-16 flex-shrink-0">{label}</span>
                <span className="text-slate-300">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <p className="section-title mb-3">Reference</p>
        <div className="flex items-start gap-3">
          <BookOpen size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs text-slate-300 font-medium">
              Adisa, I.T. (2026). An Integrated Framework for Explainable, Fair, and Observable Hospital Readmission Prediction: Development and Validation on MIMIC-IV.
            </p>
            <p className="text-xs text-slate-500 mt-1">arXiv:2604.22535</p>
            <p className="text-xs text-slate-500 mt-1">
              Reference repository: github.com/Tomisin92/readmission-prediction
            </p>
            <p className="text-xs text-amber-400/70 mt-2 italic">
              Results reported in the reference study are NOT PREP results. PREP is an independent implementation.
            </p>
          </div>
        </div>
      </div>

      <Disclaimer text="PREP is not clinically validated and is not intended to diagnose, treat, or make patient-care decisions. Demonstration predictions are for research and prototype evaluation only. A clinician remains responsible for interpreting any output." />
    </div>
  )
}
