import { useState } from 'react'
import { api } from '../services/api'
import { PageHeader, RiskBadge, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import RiskGauge from '../charts/RiskGauge'
import ShapChart from '../charts/ShapChart'
import { RefreshCw, Play, User } from 'lucide-react'

const DEMO_PATIENTS = [
  { label: 'Lower Risk', age: 45, gender: 'F', insurance: 'Private', prior_admissions_12mo: 0, los_days: 2, n_diagnoses: 3, n_procedures: 1, charlson_index: 0, n_medications: 2, high_risk_med: 0, emergency_adm: 0 },
  { label: 'Moderate Risk', age: 67, gender: 'M', insurance: 'Medicare', prior_admissions_12mo: 1, los_days: 5, n_diagnoses: 8, n_procedures: 3, charlson_index: 3, n_medications: 7, high_risk_med: 0, emergency_adm: 1 },
  { label: 'Higher Risk', age: 78, gender: 'M', insurance: 'Medicaid', prior_admissions_12mo: 3, los_days: 12, n_diagnoses: 14, n_procedures: 6, charlson_index: 7, n_medications: 15, high_risk_med: 1, emergency_adm: 1 },
]

const BLANK = { age: '', gender: 'F', insurance: 'Medicare', prior_admissions_12mo: '', los_days: '', n_diagnoses: '', n_procedures: '', charlson_index: '', n_medications: '', high_risk_med: 0, emergency_adm: 0 }

export default function PatientAssessment() {
  const [form, setForm] = useState(BLANK)
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const toPayload = () => ({
    age: Number(form.age),
    gender: form.gender,
    insurance: form.insurance,
    prior_admissions_12mo: Number(form.prior_admissions_12mo),
    los_days: Number(form.los_days),
    n_diagnoses: Number(form.n_diagnoses),
    n_procedures: Number(form.n_procedures),
    charlson_index: Number(form.charlson_index),
    n_medications: Number(form.n_medications),
    high_risk_med: Number(form.high_risk_med),
    emergency_adm: Number(form.emergency_adm),
  })

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null); setExplanation(null)
    try {
      const payload = toPayload()
      const [pred, expl] = await Promise.all([api.predict(payload), api.explain(payload)])
      setResult(pred)
      setExplanation(expl)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const loadDemo = (demo) => {
    const { label, ...fields } = demo
    setForm({ ...BLANK, ...fields })
    setResult(null); setExplanation(null); setError(null)
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <PageHeader title="Patient Assessment" subtitle="Enter patient data at discharge to generate a 30-day readmission risk estimate." badge="DEMO MODE" />

      <Disclaimer text="This tool is for research demonstration only. Estimates are not clinically validated and must not influence patient-care decisions." />

      {/* Demo patient buttons */}
      <div className="flex gap-2 mt-4 flex-wrap">
        <span className="text-xs text-slate-500 self-center">Load demo:</span>
        {DEMO_PATIENTS.map(d => (
          <button key={d.label} onClick={() => loadDemo(d)} className="btn-secondary text-xs py-1">
            <User size={12} className="inline mr-1" />{d.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6 mt-5">
        {/* Form */}
        <div className="card space-y-4">
          <p className="section-title">Patient Data at Discharge</p>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Age" type="number" value={form.age} onChange={v => set('age', v)} placeholder="18–120" />
            <div>
              <label className="label">Gender</label>
              <select className="input-field" value={form.gender} onChange={e => set('gender', e.target.value)}>
                <option value="F">Female</option>
                <option value="M">Male</option>
              </select>
            </div>
            <div className="col-span-2">
              <label className="label">Insurance</label>
              <select className="input-field" value={form.insurance} onChange={e => set('insurance', e.target.value)}>
                {['Medicare', 'Medicaid', 'Private', 'Other'].map(o => <option key={o}>{o}</option>)}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Prior Admissions (12 mo)" type="number" value={form.prior_admissions_12mo} onChange={v => set('prior_admissions_12mo', v)} placeholder="0–50" />
            <Field label="Length of Stay (days)" type="number" value={form.los_days} onChange={v => set('los_days', v)} placeholder="1–365" step="0.5" />
            <Field label="Number of Diagnoses" type="number" value={form.n_diagnoses} onChange={v => set('n_diagnoses', v)} placeholder="0–100" />
            <Field label="Number of Procedures" type="number" value={form.n_procedures} onChange={v => set('n_procedures', v)} placeholder="0–100" />
            <Field label="Charlson Index" type="number" value={form.charlson_index} onChange={v => set('charlson_index', v)} placeholder="0–24" />
            <Field label="Number of Medications" type="number" value={form.n_medications} onChange={v => set('n_medications', v)} placeholder="0–100" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Toggle label="High-Risk Medication" value={form.high_risk_med} onChange={v => set('high_risk_med', v)} />
            <Toggle label="Emergency Admission" value={form.emergency_adm} onChange={v => set('emergency_adm', v)} />
          </div>

          <div className="flex gap-2 pt-2">
            <button onClick={handleSubmit} disabled={loading} className="btn-primary flex items-center gap-2">
              <Play size={14} />
              {loading ? 'Estimating…' : 'Generate Risk Estimate'}
            </button>
            <button onClick={() => { setForm(BLANK); setResult(null); setExplanation(null); setError(null) }} className="btn-ghost flex items-center gap-2">
              <RefreshCw size={14} /> Reset
            </button>
          </div>
        </div>

        {/* Result */}
        <div className="space-y-4">
          {loading && <div className="card"><LoadingSpinner /></div>}
          {error && <ErrorBox message={error} />}

          {result && (
            <div className="card">
              <p className="section-title mb-4">30-Day Readmission Risk Estimate</p>
              <RiskGauge risk={result.estimated_risk} />

              <div className="mt-4 space-y-2">
                <ResultRow label="Risk Band" value={<RiskBadge band={result.risk_band} />} />
                <ResultRow label="Above Threshold" value={result.above_threshold ? <span className="text-amber-300 font-medium">Yes</span> : <span className="text-emerald-300">No</span>} />
                <ResultRow label="Threshold" value={`${(result.threshold * 100).toFixed(0)}%`} />
                <ResultRow label="Threshold Method" value={result.threshold_method} />
                <ResultRow label="Model" value={result.model_type} />
                <ResultRow label="Validation State" value={result.validation_state} />
              </div>

              <p className="text-[10px] text-slate-500 mt-3 italic">
                Prototype risk category — not a clinical risk classification.
              </p>
            </div>
          )}

          {explanation && (
            <div className="card">
              <p className="section-title mb-1">Why This Score?</p>
              <p className="text-xs text-slate-500 mb-4">
                Factors contributing to the estimated risk (SHAP contributions).
              </p>
              <ShapChart contributions={explanation.contributions} topK={10} />
              <p className="text-[10px] text-slate-500 mt-3 italic">
                These contributions explain the model estimate. They do not prove that changing a feature would change the patient's outcome.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Field({ label, type, value, onChange, placeholder, step }) {
  return (
    <div>
      <label className="label">{label}</label>
      <input
        type={type}
        className="input-field"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        step={step}
        min={0}
      />
    </div>
  )
}

function Toggle({ label, value, onChange }) {
  return (
    <div>
      <label className="label">{label}</label>
      <div className="flex gap-2">
        {[0, 1].map(v => (
          <button
            key={v}
            onClick={() => onChange(v)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              value === v
                ? 'bg-blue-600/30 border-blue-500 text-blue-300'
                : 'bg-navy-800 border-navy-600 text-slate-400 hover:border-navy-500'
            }`}
          >
            {v === 1 ? 'Yes' : 'No'}
          </button>
        ))}
      </div>
    </div>
  )
}

function ResultRow({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-navy-800 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-xs text-slate-200 font-medium">{value}</span>
    </div>
  )
}
