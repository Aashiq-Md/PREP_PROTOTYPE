import { useState } from 'react'
import { api } from '../services/api'
import { PageHeader, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import ShapChart from '../charts/ShapChart'
import { Play, User } from 'lucide-react'

const DEMO_CASES = [
  { label: 'Strong Explainability', age: 72, gender: 'F', insurance: 'Medicare', prior_admissions_12mo: 2, los_days: 8, n_diagnoses: 11, n_procedures: 4, charlson_index: 5, n_medications: 12, high_risk_med: 1, emergency_adm: 0 },
  { label: 'Higher Risk', age: 78, gender: 'M', insurance: 'Medicaid', prior_admissions_12mo: 3, los_days: 12, n_diagnoses: 14, n_procedures: 6, charlson_index: 7, n_medications: 15, high_risk_med: 1, emergency_adm: 1 },
  { label: 'Lower Risk', age: 45, gender: 'F', insurance: 'Private', prior_admissions_12mo: 0, los_days: 2, n_diagnoses: 3, n_procedures: 1, charlson_index: 0, n_medications: 2, high_risk_med: 0, emergency_adm: 0 },
]

export default function Explainability() {
  const [explanation, setExplanation] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeCase, setActiveCase] = useState(null)

  const run = async (patient) => {
    setLoading(true); setError(null)
    try {
      const [pred, expl] = await Promise.all([api.predict(patient), api.explain(patient)])
      setPrediction(pred)
      setExplanation(expl)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const loadCase = (c) => {
    const { label, ...patient } = c
    setActiveCase(label)
    run(patient)
  }

  const increasing = explanation?.contributions.filter(c => c.shap_value > 0) ?? []
  const decreasing = explanation?.contributions.filter(c => c.shap_value < 0) ?? []

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <PageHeader title="Explainability" subtitle="Per-patient contributions or clearly labeled heuristic approximation." />

      <Disclaimer text={explanation?.disclaimer || "Explanations do not prove causation or that changing a feature would change the patient's outcome. Demo contributions are heuristic approximations, not SHAP values."} />

      {/* Demo case selector */}
      <div className="flex gap-2 mt-5 flex-wrap">
        <span className="text-xs text-slate-500 self-center">Load demo case:</span>
        {DEMO_CASES.map(c => (
          <button
            key={c.label}
            onClick={() => loadCase(c)}
            className={`btn-secondary text-xs py-1 ${activeCase === c.label ? 'border-blue-500 text-blue-300' : ''}`}
          >
            <User size={12} className="inline mr-1" />{c.label}
          </button>
        ))}
      </div>

      {loading && <div className="card mt-5"><LoadingSpinner /></div>}
      {error && <div className="mt-5"><ErrorBox message={error} /></div>}

      {explanation && prediction && (
        <div className="space-y-5 mt-5">
          {/* Summary */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <p className="section-title">Explanation Summary</p>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold text-white">{(prediction.estimated_risk * 100).toFixed(1)}%</span>
                <span className={`text-sm font-medium ${
                  prediction.risk_band === 'Higher' ? 'text-red-300' :
                  prediction.risk_band === 'Moderate' ? 'text-amber-300' : 'text-emerald-300'
                }`}>{prediction.risk_band} Risk</span>
              </div>
            </div>

            <p className={`text-xs mb-3 ${explanation.explanation_method === 'shap_tree_explainer' ? 'text-blue-300' : 'text-amber-300'}`}>
              {explanation.explanation_label}
            </p>

            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">{explanation.explanation_method === 'shap_tree_explainer' ? 'SHAP Base Value' : 'Heuristic Baseline'}</p>
                <p className="text-lg font-bold text-slate-300">{explanation.base_value?.toFixed(4)}</p>
                <p className="text-[10px] text-slate-600">{explanation.explanation_method === 'shap_tree_explainer' ? 'Raw model output' : 'Demo formula baseline'}</p>
              </div>
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">{explanation.explanation_method === 'shap_tree_explainer' ? 'Sum of SHAP Values' : 'Heuristic Contributions'}</p>
                <p className="text-lg font-bold text-blue-300">{explanation.sum_contributions >= 0 ? '+' : ''}{explanation.sum_contributions?.toFixed(4)}</p>
                <p className="text-[10px] text-slate-600">{explanation.explanation_method === 'shap_tree_explainer' ? 'Additive SHAP values' : 'Not a model attribution'}</p>
              </div>
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">{explanation.explanation_method === 'shap_tree_explainer' ? 'Raw Model Output' : 'Synthetic Heuristic Score'}</p>
                <p className="text-lg font-bold text-white">{explanation.model_output_logit?.toFixed(4)}</p>
                <p className="text-[10px] text-slate-600">{explanation.explanation_method === 'shap_tree_explainer' ? 'Base + SHAP values' : 'Not the trained model output'}</p>
              </div>
            </div>
          </div>

          {/* Contribution chart */}
          <div className="card">
            <p className="section-title mb-4">Feature Contributions</p>
            <div className="flex gap-4 text-xs mb-3">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-blue-500 inline-block" />Positive contribution</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-emerald-500 inline-block" />Negative contribution</span>
            </div>
            <ShapChart contributions={explanation.contributions} topK={13} />
          </div>

          {/* Breakdown tables */}
          <div className="grid grid-cols-2 gap-4">
            <ContribTable title="Positive Contributions" items={increasing} positive />
            <ContribTable title="Negative Contributions" items={decreasing} positive={false} />
          </div>
        </div>
      )}

      {!explanation && !loading && (
        <div className="card mt-5 text-center py-12">
          <p className="text-slate-500 text-sm">Select a demo case above to see an explanation.</p>
        </div>
      )}
    </div>
  )
}

function ContribTable({ title, items, positive }) {
  return (
    <div className="card">
      <p className={`text-xs font-semibold mb-3 ${positive ? 'text-blue-300' : 'text-emerald-300'}`}>{title}</p>
      {items.length === 0
        ? <p className="text-xs text-slate-600">None</p>
        : (
          <div className="space-y-1">
            {items.slice(0, 6).map((c, i) => (
              <div key={i} className="flex items-center justify-between py-1 border-b border-navy-800 last:border-0">
                <span className="text-xs text-slate-300 truncate mr-2">{c.feature}</span>
                <span className={`text-xs font-mono font-medium flex-shrink-0 ${positive ? 'text-blue-300' : 'text-emerald-300'}`}>
                  {c.shap_value >= 0 ? '+' : ''}{c.shap_value?.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        )
      }
    </div>
  )
}
