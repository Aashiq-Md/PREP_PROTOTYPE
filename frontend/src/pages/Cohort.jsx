import { useEffect, useState, useMemo } from 'react'
import { api } from '../services/api'
import { PageHeader, RiskBadge, Disclaimer, LoadingSpinner, ErrorBox } from '../components/UI'
import { Search, ArrowUpDown } from 'lucide-react'

export default function Cohort() {
  const [cohort, setCohort] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [riskFilter, setRiskFilter] = useState('All')
  const [thresholdFilter, setThresholdFilter] = useState('All')
  const [sortKey, setSortKey] = useState('estimated_risk')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    api.demoCohort()
      .then(setCohort)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    let rows = cohort
    if (search) rows = rows.filter(r => r.patient_id?.toLowerCase().includes(search.toLowerCase()))
    if (riskFilter !== 'All') rows = rows.filter(r => r.risk_band === riskFilter)
    if (thresholdFilter === 'Above') rows = rows.filter(r => r.above_threshold === 1)
    if (thresholdFilter === 'Below') rows = rows.filter(r => r.above_threshold === 0)
    return [...rows].sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0
      return sortDir === 'asc' ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    })
  }, [cohort, search, riskFilter, thresholdFilter, sortKey, sortDir])

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const summary = useMemo(() => ({
    total: cohort.length,
    higher: cohort.filter(r => r.risk_band === 'Higher').length,
    moderate: cohort.filter(r => r.risk_band === 'Moderate').length,
    lower: cohort.filter(r => r.risk_band === 'Lower').length,
    aboveThreshold: cohort.filter(r => r.above_threshold === 1).length,
  }), [cohort])

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <PageHeader title="Cohort" subtitle="Synthetic demonstration cohort — not real patient records." badge="SYNTHETIC DATA" />
      <Disclaimer text="All patient IDs and data in this cohort are synthetically generated for demonstration purposes only." />

      {/* Summary cards */}
      <div className="grid grid-cols-5 gap-3 mt-5">
        {[
          { label: 'Total', value: summary.total, color: 'text-slate-300' },
          { label: 'Higher Risk', value: summary.higher, color: 'text-red-300' },
          { label: 'Moderate Risk', value: summary.moderate, color: 'text-amber-300' },
          { label: 'Lower Risk', value: summary.lower, color: 'text-emerald-300' },
          { label: 'Above Threshold', value: summary.aboveThreshold, color: 'text-blue-300' },
        ].map(s => (
          <div key={s.label} className="card-sm text-center">
            <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-xs text-slate-500 mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex gap-3 mt-5 flex-wrap">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            className="input-field pl-8 w-48"
            placeholder="Search patient ID…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <FilterSelect label="Risk Band" value={riskFilter} onChange={setRiskFilter} options={['All', 'Lower', 'Moderate', 'Higher']} />
        <FilterSelect label="Threshold" value={thresholdFilter} onChange={setThresholdFilter} options={['All', 'Above', 'Below']} />
      </div>

      {/* Table */}
      <div className="card mt-4 overflow-x-auto">
        {loading ? <LoadingSpinner /> : error ? <ErrorBox message={error} /> : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-navy-700">
                {[
                  { key: 'patient_id', label: 'Patient ID' },
                  { key: 'age', label: 'Age' },
                  { key: 'los_days', label: 'LOS (days)' },
                  { key: 'prior_admissions_12mo', label: 'Prior Adm.' },
                  { key: 'charlson_index', label: 'Charlson' },
                  { key: 'estimated_risk', label: 'Est. Risk' },
                  { key: 'risk_band', label: 'Risk Band' },
                  { key: 'above_threshold', label: 'Threshold' },
                ].map(col => (
                  <th
                    key={col.key}
                    className="text-left py-2 px-3 text-slate-400 font-medium cursor-pointer hover:text-slate-200 select-none"
                    onClick={() => toggleSort(col.key)}
                  >
                    <span className="flex items-center gap-1">
                      {col.label}
                      <ArrowUpDown size={10} className="text-slate-600" />
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => (
                <tr key={row.patient_id} className={`border-b border-navy-800 hover:bg-navy-800/40 ${i % 2 === 0 ? '' : 'bg-navy-900/30'}`}>
                  <td className="py-2 px-3 font-mono text-slate-300">{row.patient_id}</td>
                  <td className="py-2 px-3 text-slate-300">{row.age}</td>
                  <td className="py-2 px-3 text-slate-300">{Number(row.los_days).toFixed(1)}</td>
                  <td className="py-2 px-3 text-slate-300">{row.prior_admissions_12mo}</td>
                  <td className="py-2 px-3 text-slate-300">{row.charlson_index}</td>
                  <td className="py-2 px-3 font-medium text-slate-200">{(row.estimated_risk * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3"><RiskBadge band={row.risk_band} /></td>
                  <td className="py-2 px-3">
                    {row.above_threshold === 1
                      ? <span className="text-amber-300 font-medium">Above</span>
                      : <span className="text-slate-500">Below</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="text-xs text-slate-600 mt-3">Showing {filtered.length} of {cohort.length} patients</p>
      </div>
    </div>
  )
}

function FilterSelect({ label, value, onChange, options }) {
  return (
    <select
      className="input-field w-auto"
      value={value}
      onChange={e => onChange(e.target.value)}
      aria-label={label}
    >
      {options.map(o => <option key={o}>{o}</option>)}
    </select>
  )
}
