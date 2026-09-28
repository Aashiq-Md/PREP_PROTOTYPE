import { useState } from 'react'
import { api } from '../services/api'
import { PageHeader, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { AlertTriangle, CheckCircle, Zap } from 'lucide-react'

export default function Monitoring() {
  const [driftResult, setDriftResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [simulated, setSimulated] = useState(false)

  const runDrift = async (simulate) => {
    setLoading(true); setError(null)
    try {
      const result = await api.driftCheck(simulate)
      setDriftResult(result)
      setSimulated(simulate)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const summary = driftResult?.summary
  const featureDrift = driftResult?.feature_drift ?? {}
  const predDrift = driftResult?.prediction_drift ?? {}

  const featureChartData = Object.entries(featureDrift).map(([feat, v]) => ({
    feature: feat.replace(/_/g, ' '),
    kl: v.kl_divergence,
    alert: v.alert,
  })).sort((a, b) => b.kl - a.kl)

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <PageHeader title="Model Monitoring" subtitle="Feature drift (KL divergence) and prediction drift (2σ alert)." />

      <Disclaimer text="Synthetic drift demonstration — not real clinical data. KL threshold ≤ 0.05. Prediction drift alert at 2 standard deviations from reference mean." />

      {/* Controls */}
      <div className="flex gap-3 mt-5">
        <button onClick={() => runDrift(false)} disabled={loading} className="btn-secondary flex items-center gap-2">
          <CheckCircle size={14} /> Check Stable Data
        </button>
        <button onClick={() => runDrift(true)} disabled={loading} className="btn-primary flex items-center gap-2">
          <Zap size={14} /> Simulate Data Drift
        </button>
      </div>

      {loading && <div className="card mt-5"><LoadingSpinner /></div>}
      {error && <div className="mt-5"><ErrorBox message={error} /></div>}

      {summary && (
        <div className="space-y-5 mt-5">
          {/* Alert banner */}
          <div className={`rounded-xl border p-4 flex items-center gap-3 ${
            summary.alert
              ? 'bg-red-950/40 border-red-700/50'
              : 'bg-emerald-950/40 border-emerald-700/50'
          }`}>
            {summary.alert
              ? <AlertTriangle size={20} className="text-red-400 flex-shrink-0" />
              : <CheckCircle size={20} className="text-emerald-400 flex-shrink-0" />
            }
            <div>
              <p className={`font-bold text-lg ${summary.alert ? 'text-red-300' : 'text-emerald-300'}`}>
                {summary.status}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                {simulated ? 'Simulated distribution shift detected' : 'No significant drift detected'}
                {' · '}
                {summary.n_features_drifted} of {summary.n_features_monitored} features drifted
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-3">
            <StatCard label="Features Monitored" value={summary.n_features_monitored} />
            <StatCard label="Features Drifted" value={summary.n_features_drifted} alert={summary.n_features_drifted > 0} />
            <StatCard label="Prediction Drift" value={predDrift.alert ? 'ALERT' : 'Stable'} alert={predDrift.alert} />
            <StatCard label="Z-Score" value={predDrift.z_score?.toFixed(2) ?? '—'} alert={predDrift.alert} />
          </div>

          {/* Prediction drift detail */}
          <div className="card">
            <p className="section-title mb-3">Prediction Drift</p>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">Reference Mean Risk</p>
                <p className="text-xl font-bold text-slate-300">{(predDrift.ref_mean * 100)?.toFixed(1)}%</p>
              </div>
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">Current Mean Risk</p>
                <p className={`text-xl font-bold ${predDrift.alert ? 'text-red-300' : 'text-slate-300'}`}>
                  {(predDrift.cur_mean * 100)?.toFixed(1)}%
                </p>
              </div>
              <div className="bg-navy-800 rounded-lg p-3">
                <p className="text-xs text-slate-500">Z-Score (alert at 2σ)</p>
                <p className={`text-xl font-bold ${predDrift.alert ? 'text-red-300' : 'text-emerald-300'}`}>
                  {predDrift.z_score?.toFixed(2)}σ
                </p>
              </div>
            </div>
          </div>

          {/* Feature drift chart */}
          <div className="card">
            <p className="section-title mb-3">Feature KL Divergence</p>
            <p className="text-xs text-slate-500 mb-4">KL threshold = {summary.kl_threshold} — bars above threshold indicate drift</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={featureChartData} margin={{ left: 0, right: 10, top: 4, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={false} />
                <XAxis dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-30} textAnchor="end" tickLine={false} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ background: '#1e3a5f', border: '1px solid #334e68', borderRadius: 8, fontSize: 11 }}
                  formatter={(v) => [v?.toFixed(4), 'KL Divergence']}
                />
                <ReferenceLine y={summary.kl_threshold} stroke="#ef4444" strokeDasharray="4 2" label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10 }} />
                <Bar dataKey="kl" radius={[3, 3, 0, 0]}
                  fill="#3b82f6"
                  label={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Feature drift table */}
          <div className="card overflow-x-auto">
            <p className="section-title mb-3">Feature Drift Detail</p>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-navy-700">
                  {['Feature', 'KL Divergence', 'Threshold', 'Status'].map(h => (
                    <th key={h} className="text-left py-2 px-3 text-slate-400 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(featureDrift).map(([feat, v]) => (
                  <tr key={feat} className="border-b border-navy-800 hover:bg-navy-800/30">
                    <td className="py-2 px-3 text-slate-300">{feat.replace(/_/g, ' ')}</td>
                    <td className="py-2 px-3 font-mono text-slate-300">{v.kl_divergence?.toFixed(5)}</td>
                    <td className="py-2 px-3 text-slate-500">{summary.kl_threshold}</td>
                    <td className="py-2 px-3">
                      {v.alert
                        ? <span className="text-red-300 flex items-center gap-1"><AlertTriangle size={11} />Drift</span>
                        : <span className="text-emerald-300 flex items-center gap-1"><CheckCircle size={11} />Stable</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!driftResult && !loading && (
        <div className="card mt-5 text-center py-12">
          <p className="text-slate-500 text-sm">Click "Check Stable Data" or "Simulate Data Drift" to run drift detection.</p>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, alert }) {
  return (
    <div className="card-sm text-center">
      <p className={`text-xl font-bold ${alert ? 'text-red-300' : 'text-slate-300'}`}>{value}</p>
      <p className="text-xs text-slate-500 mt-0.5">{label}</p>
    </div>
  )
}
