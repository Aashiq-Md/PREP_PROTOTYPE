import { AlertTriangle } from 'lucide-react'

export function PageHeader({ title, subtitle, badge }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold text-white">{title}</h1>
        {badge && <span className="badge-info">{badge}</span>}
      </div>
      {subtitle && <p className="text-sm text-slate-400 mt-1">{subtitle}</p>}
    </div>
  )
}

export function StatCard({ label, value, sub, color = 'blue' }) {
  const colors = {
    blue:   'text-blue-300',
    green:  'text-emerald-300',
    amber:  'text-amber-300',
    red:    'text-red-300',
    slate:  'text-slate-300',
  }
  return (
    <div className="card">
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${colors[color]}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}

export function RiskBadge({ band }) {
  if (!band) return null
  const cls = {
    Lower:    'badge-lower',
    Moderate: 'badge-moderate',
    Higher:   'badge-higher',
  }
  return <span className={cls[band] || 'badge-info'}>{band}</span>
}

export function Disclaimer({ text }) {
  return (
    <div className="disclaimer-box flex gap-2">
      <AlertTriangle size={14} className="text-amber-400 flex-shrink-0 mt-0.5" />
      <span>{text}</span>
    </div>
  )
}

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export function ErrorBox({ message }) {
  return (
    <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-4 text-sm text-red-300">
      {message}
    </div>
  )
}

export function SectionDivider({ label }) {
  return (
    <div className="flex items-center gap-3 my-5">
      <div className="flex-1 h-px bg-navy-700" />
      <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</span>
      <div className="flex-1 h-px bg-navy-700" />
    </div>
  )
}
