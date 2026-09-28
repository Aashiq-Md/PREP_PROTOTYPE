import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { PageHeader, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { AlertTriangle, CheckCircle, Info } from 'lucide-react'

const statusStyles = {
  PASS: 'text-emerald-300',
  FAIL: 'text-red-300',
  INDETERMINATE: 'text-amber-300',
}

function dimensionStatus(dimension) {
  if (dimension.auc_pass === false || dimension.fnr_pass === false) return 'FAIL'
  if (dimension.subgroups?.some(group => !group.sufficient_events)) return 'INDETERMINATE'
  if (dimension.status === 'FAIL') return 'FAIL'
  if (dimension.status === 'INDETERMINATE') return 'INDETERMINATE'
  if (dimension.overall_pass === true) return 'PASS'
  return 'INDETERMINATE'
}

function StatusLabel({ status }) {
  const Icon = status === 'PASS' ? CheckCircle : AlertTriangle
  return (
    <span className={`inline-flex items-center gap-1 ${statusStyles[status] || statusStyles.INDETERMINATE}`}>
      <Icon size={13} />
      {status || 'INDETERMINATE'}
    </span>
  )
}

export default function Fairness() {
  const [report, setReport] = useState(null)
  const [activeDim, setActiveDim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.fairnessReport()
      .then(setReport)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const dimensions = report?.dimensions || {}
  const entries = Object.entries(dimensions)
  const selectedName = dimensions[activeDim] ? activeDim : entries[0]?.[0]
  const selected = selectedName ? dimensions[selectedName] : null
  const summary = report?.summary

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <PageHeader title="Fairness & Governance" subtitle="Subgroup performance audit from the latest saved training report." />

      <Disclaimer text={report?.simulated
        ? `SIMULATED audit from ${report.dataset || 'synthetic data'}; not clinical evidence. Groups with fewer than ${summary?.min_events_required ?? 30} readmission events cannot support a PASS. Race/ethnicity is an audit-only field.`
        : `Audit results are based on the saved model evaluation report. Groups with fewer than ${summary?.min_events_required ?? 30} readmission events cannot support a PASS. Race/ethnicity is an audit-only field.`}
      />

      {loading && <div className="card mt-5"><LoadingSpinner /></div>}
      {error && <div className="mt-5"><ErrorBox message={error} /></div>}

      {!loading && !error && (
        <>
          <div className="card mt-5 flex items-center justify-between gap-4">
            <div>
              <p className="text-xs text-slate-500">Overall audit status</p>
              <p className={`text-lg font-semibold ${statusStyles[summary?.status] || statusStyles.INDETERMINATE}`}>
                {summary?.status || 'INDETERMINATE'}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {report?.available
                  ? `${report.result_label || 'Result label unavailable'} · ${report.dataset || 'Dataset unavailable'}`
                  : report?.message || 'No fairness report is available.'}
              </p>
            </div>
            <StatusLabel status={summary?.status || 'INDETERMINATE'} />
          </div>

          {entries.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
                {entries.map(([name, dimension]) => {
                  const status = dimensionStatus(dimension)
                  return (
                    <button
                      key={name}
                      type="button"
                      onClick={() => setActiveDim(name)}
                      className={`card-sm text-left transition-colors ${selectedName === name ? 'border-blue-500 bg-blue-900/10' : 'hover:border-navy-600'}`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <p className="text-xs font-medium text-slate-300">{name}</p>
                        <StatusLabel status={status} />
                      </div>
                      <p className="text-[10px] text-slate-500">
                        ΔAUC: {dimension.auc_gap == null ? 'Not available' : dimension.auc_gap.toFixed(3)}
                        {' · '}
                        ΔFNR: {dimension.fnr_gap == null ? 'Not available' : dimension.fnr_gap.toFixed(3)}
                      </p>
                    </button>
                  )
                })}
              </div>

              <div className="card mt-4 flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-400">
                <span>AUC gap limit: ≤ {summary?.auc_gap_limit ?? 'Not available'}</span>
                <span>FNR gap limit: ≤ {summary?.fnr_gap_limit ?? 'Not available'}</span>
                <span>Minimum events: {summary?.min_events_required ?? 'Not available'}</span>
              </div>

              {selected && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <SubgroupChart
                      title={`AUC-ROC by ${selectedName}`}
                      data={selected.subgroups.filter(group => group.auc_roc != null)}
                      dataKey="auc_roc"
                      color="#3b82f6"
                      gap={selected.auc_gap}
                      limit={summary?.auc_gap_limit}
                      yDomain={[0.5, 1]}
                      formatter={value => value?.toFixed(3)}
                    />
                    <SubgroupChart
                      title={`FNR by ${selectedName}`}
                      data={selected.subgroups.filter(group => group.fnr != null)}
                      dataKey="fnr"
                      color="#f59e0b"
                      gap={selected.fnr_gap}
                      limit={summary?.fnr_gap_limit}
                      yDomain={[0, 1]}
                      formatter={value => value?.toFixed(3)}
                    />
                  </div>

                  <div className="card mt-4 overflow-x-auto">
                    <div className="flex items-center justify-between mb-3">
                      <p className="section-title">{selectedName} — Detailed Metrics</p>
                      <StatusLabel status={dimensionStatus(selected)} />
                    </div>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-navy-700">
                          {['Subgroup', 'N', 'Events', 'AUC-ROC', 'FNR', 'PPV', 'Flag Rate', 'Evidence'].map(heading => (
                            <th key={heading} className="text-left py-2 px-3 text-slate-400 font-medium">{heading}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {selected.subgroups.map(group => (
                          <tr key={group.subgroup} className="border-b border-navy-800 hover:bg-navy-800/30">
                            <td className="py-2 px-3 font-medium text-slate-200">{group.subgroup}</td>
                            <td className="py-2 px-3 text-slate-400">{group.n ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-400">{group.n_positive ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-300">{group.auc_roc?.toFixed(3) ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-300">{group.fnr?.toFixed(3) ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-300">{group.ppv?.toFixed(3) ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-300">{group.flag_rate == null ? '—' : `${(group.flag_rate * 100).toFixed(0)}%`}</td>
                            <td className="py-2 px-3">
                              {group.sufficient_events
                                ? <span className="text-slate-300">Included in audit</span>
                                : <span className="text-amber-300">Insufficient events</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </>
          ) : (
            <div className="card mt-4 text-center py-10">
              <p className="text-sm text-amber-300">INDETERMINATE — no subgroup fairness results are available.</p>
              <p className="text-xs text-slate-500 mt-2">No PASS is inferred from missing audit data.</p>
            </div>
          )}

          <div className="card mt-4 flex gap-2 text-xs text-slate-500">
            <Info size={13} className="text-blue-400 flex-shrink-0 mt-0.5" />
            <span>Race/ethnicity is used for fairness auditing only and is excluded from model features. These audit results do not establish clinical validity.</span>
          </div>
        </>
      )}
    </div>
  )
}

function SubgroupChart({ title, data, dataKey, color, gap, limit, yDomain, formatter }) {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-slate-300">{title}</p>
        <span className="text-xs text-slate-400">
          Δ = {gap == null ? 'Not available' : `${gap.toFixed(3)} (limit ${limit})`}
        </span>
      </div>
      {data.length === 0
        ? <p className="text-xs text-amber-300 py-12 text-center">Insufficient data for this metric.</p>
        : (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={data} margin={{ left: 0, right: 10, top: 4, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={false} />
              <XAxis dataKey="subgroup" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-20} textAnchor="end" tickLine={false} />
              <YAxis domain={yDomain} tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={formatter} />
              <Tooltip
                contentStyle={{ background: '#1e3a5f', border: '1px solid #334e68', borderRadius: 8, fontSize: 11 }}
                formatter={value => [formatter(value), dataKey.toUpperCase()]}
              />
              {limit != null && <ReferenceLine y={limit} stroke="#ef4444" strokeDasharray="4 2" />}
              <Bar dataKey={dataKey} fill={color} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
    </div>
  )
}
