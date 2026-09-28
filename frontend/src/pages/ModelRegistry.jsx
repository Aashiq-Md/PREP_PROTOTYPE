import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { PageHeader, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import { Database } from 'lucide-react'

function fairnessDimensionStatus(dimension) {
  if (dimension.auc_pass === false || dimension.fnr_pass === false) return 'FAIL'
  if (dimension.subgroups?.some(group => !group.sufficient_events)) return 'INDETERMINATE'
  if (dimension.status === 'FAIL') return 'FAIL'
  if (dimension.status === 'INDETERMINATE') return 'INDETERMINATE'
  if (dimension.overall_pass === true) return 'PASS'
  return 'INDETERMINATE'
}

function display(value) {
  return value == null || value === '' ? 'Not available' : value
}

export default function ModelRegistry() {
  const [info, setInfo] = useState(null)
  const [fairness, setFairness] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.modelInfo(), api.fairnessReport()])
      .then(([model, audit]) => {
        setInfo(model)
        setFairness(audit)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const evaluation = info?.evaluation

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <PageHeader title="Model Registry" subtitle="Model artifact tracking, versioning, and available evaluation metadata." />

      <Disclaimer text="PREP has not been clinically validated. Any saved demo metrics or fairness audits are explicitly simulated and must not be interpreted as clinical performance." />

      {loading ? <LoadingSpinner /> : error ? <div className="mt-5"><ErrorBox message={error} /></div> : (
        <div className="space-y-4 mt-5">
          <div className="card">
            <div className="flex items-center gap-3 mb-4">
              <Database size={18} className="text-blue-400" />
              <p className="text-base font-semibold text-white">Active Model</p>
              <span className="badge-info">{display(info?.model_type)}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
              {[
                ['Model Version', info?.model_version],
                ['Model Type', info?.model_type],
                ['Dataset', info?.dataset],
                ['Training Date', info?.training_date],
                ['Calibration Method', info?.calibration_method],
                ['Threshold', info?.threshold == null ? null : `${(info.threshold * 100).toFixed(1)}%`],
                ['Threshold Method', info?.threshold_method],
                ['Validation State', info?.validation_state],
                ['Result Label', info?.result_label],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center justify-between gap-3 py-1.5 border-b border-navy-800 last:border-0">
                  <span className="text-xs text-slate-400">{label}</span>
                  <span className="text-xs text-slate-200 font-medium text-right">{display(value)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <p className="section-title mb-3">Saved Evaluation Metrics</p>
            {evaluation ? (
              <>
                <p className="text-xs text-amber-300 mb-3">
                  {info.evaluation_label || 'Evaluation result'} — evaluation from the matching saved training report; not clinical validation.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    ['Test N', evaluation.n_test],
                    ['Prevalence', evaluation.prevalence == null ? null : `${(evaluation.prevalence * 100).toFixed(1)}%`],
                    ['AUC-ROC', evaluation.auc_roc],
                    ['AUC-PRC', evaluation.auc_prc],
                    ['Brier Score', evaluation.brier_score],
                    ['Sensitivity', evaluation.sensitivity],
                    ['PPV', evaluation.ppv],
                    ['Threshold', evaluation.threshold],
                  ].map(([label, value]) => (
                    <div key={label} className="bg-navy-800 rounded-lg p-3">
                      <p className="text-xs text-slate-500">{label}</p>
                      <p className="text-sm font-semibold text-slate-200 mt-1">{display(value)}</p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-400">
                Not available for the active model. No matching saved evaluation report was found.
              </p>
            )}
          </div>

          <div className="card">
            <div className="flex items-center justify-between gap-3 mb-3">
              <p className="section-title">Saved Fairness Audit</p>
              <span className="text-xs text-amber-300">{fairness?.summary?.status || 'INDETERMINATE'}</span>
            </div>
            {fairness?.simulated && (
              <p className="text-xs text-amber-300 mb-3">SIMULATED audit — not clinical evidence.</p>
            )}
            {Object.entries(fairness?.dimensions || {}).length ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {Object.entries(fairness.dimensions).map(([name, dimension]) => {
                  const status = fairnessDimensionStatus(dimension)
                  return (
                    <div key={name} className="bg-navy-800 rounded-lg p-3">
                      <div className="flex justify-between gap-3">
                        <p className="text-xs text-slate-300 font-medium">{name}</p>
                        <span className={`text-xs ${status === 'PASS' ? 'text-emerald-300' : status === 'FAIL' ? 'text-red-300' : 'text-amber-300'}`}>
                          {status}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-500 mt-1">
                        ΔAUC: {display(dimension.auc_gap)} · ΔFNR: {display(dimension.fnr_gap)}
                      </p>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-sm text-slate-400">Not available. No saved fairness audit is present.</p>
            )}
            <p className="text-[10px] text-slate-500 mt-3">
              Insufficient subgroup evidence is reported as INDETERMINATE, not PASS.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
