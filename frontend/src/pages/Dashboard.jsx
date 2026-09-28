import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Activity, AlertTriangle, CheckCircle, Clock, TrendingUp, Users, Zap } from 'lucide-react'
import { api } from '../services/api'
import { StatCard, Disclaimer, LoadingSpinner } from '../components/UI'

const PILLARS = [
  { label: 'PREDICT', desc: 'Estimate 30-day readmission risk at discharge', color: 'text-blue-400', bg: 'bg-blue-900/20 border-blue-800/40' },
  { label: 'EXPLAIN', desc: 'Show which factors drove the estimate', color: 'text-violet-400', bg: 'bg-violet-900/20 border-violet-800/40' },
  { label: 'AUDIT',   desc: 'Check fairness across demographic groups', color: 'text-emerald-400', bg: 'bg-emerald-900/20 border-emerald-800/40' },
  { label: 'MONITOR', desc: 'Detect model and data drift over time', color: 'text-amber-400', bg: 'bg-amber-900/20 border-amber-800/40' },
]

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([api.health(), api.modelInfo(), api.metrics()])
      .then(([h, m, met]) => {
        if (h.status === 'fulfilled') setHealth(h.value)
        if (m.status === 'fulfilled') setModelInfo(m.value)
        if (met.status === 'fulfilled') setMetrics(met.value)
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">PREP Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">
          Predict Readmission Estimation of Patient — Research Prototype
        </p>
      </div>

      {/* Disclaimer */}
      <Disclaimer text="PREP is a research/hackathon prototype. It is not clinically validated and must not be used to make patient-care decisions. All demonstration predictions use synthetic data." />

      {/* Four Pillars */}
      <div className="grid grid-cols-4 gap-4 mt-6">
        {PILLARS.map(p => (
          <div key={p.label} className={`card border ${p.bg}`}>
            <p className={`text-lg font-bold ${p.color}`}>{p.label}</p>
            <p className="text-xs text-slate-400 mt-1">{p.desc}</p>
          </div>
        ))}
      </div>

      {/* System Status */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="card">
          <p className="section-title mb-3">System Status</p>
          {loading ? <LoadingSpinner /> : (
            <div className="space-y-2">
              <StatusRow label="API" value={health ? 'Online' : 'Offline'} ok={!!health} />
              <StatusRow label="Model" value={modelInfo?.model_type || 'DEMO'} ok />
              <StatusRow label="Version" value={modelInfo?.model_version || '—'} ok />
              <StatusRow label="Validation" value={modelInfo?.validation_state || 'NOT YET VALIDATED'} ok={false} />
              <StatusRow label="Dataset" value={modelInfo?.dataset || 'Synthetic Demo'} ok />
            </div>
          )}
        </div>

        <div className="card">
          <p className="section-title mb-3">Session Metrics</p>
          {loading ? <LoadingSpinner /> : (
            <div className="space-y-2">
              <StatusRow label="Predictions this session" value={metrics?.n_predictions ?? 0} ok />
              <StatusRow
                label="Mean estimated risk"
                value={metrics?.mean_estimated_risk != null ? `${(metrics.mean_estimated_risk * 100).toFixed(1)}%` : '—'}
                ok
              />
              <StatusRow
                label="Above threshold"
                value={metrics?.pct_above_threshold != null ? `${(metrics.pct_above_threshold * 100).toFixed(1)}%` : '—'}
                ok
              />
              <StatusRow label="Threshold" value={modelInfo?.threshold != null ? `${(modelInfo.threshold * 100).toFixed(0)}%` : '—'} ok />
              <StatusRow label="Threshold method" value={modelInfo?.threshold_method || '—'} ok />
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mt-6">
        <p className="section-title mb-3">Quick Actions</p>
        <div className="grid grid-cols-3 gap-4">
          <QuickAction to="/assess" icon={<Zap size={18} />} label="New Patient Assessment" desc="Enter patient data and generate risk estimate" />
          <QuickAction to="/fairness" icon={<Users size={18} />} label="Fairness Audit" desc="Review subgroup performance metrics" />
          <QuickAction to="/monitoring" icon={<Activity size={18} />} label="Drift Monitor" desc="Check for data and prediction drift" />
        </div>
      </div>

      {/* Flow diagram */}
      <div className="card mt-6">
        <p className="section-title mb-4">PREP Workflow</p>
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {['Patient Data', 'Feature Engineering', 'ML Model', 'Calibration', 'Prediction', 'Explanation', 'Fairness Audit', 'Monitoring'].map((step, i, arr) => (
            <div key={step} className="flex items-center gap-2">
              <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-1.5 text-xs text-slate-300 font-medium whitespace-nowrap">
                {step}
              </div>
              {i < arr.length - 1 && <span className="text-navy-600">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatusRow({ label, value, ok }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-navy-800 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <div className="flex items-center gap-1.5">
        {ok
          ? <CheckCircle size={12} className="text-emerald-400" />
          : <AlertTriangle size={12} className="text-amber-400" />
        }
        <span className="text-xs text-slate-200 font-medium">{String(value)}</span>
      </div>
    </div>
  )
}

function QuickAction({ to, icon, label, desc }) {
  return (
    <Link to={to} className="card hover:border-blue-700/50 hover:bg-navy-800/50 transition-colors cursor-pointer block">
      <div className="flex items-center gap-2 text-blue-400 mb-2">
        {icon}
        <span className="text-sm font-medium text-slate-200">{label}</span>
      </div>
      <p className="text-xs text-slate-500">{desc}</p>
    </Link>
  )
}
