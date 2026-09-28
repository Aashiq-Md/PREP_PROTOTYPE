import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine, ResponsiveContainer } from 'recharts'

const POS_COLOR = '#3b82f6'  // blue
const NEG_COLOR = '#10b981'  // emerald

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg p-3 text-xs shadow-xl">
      <p className="font-medium text-slate-200 mb-1">{d.feature}</p>
      <p className="text-slate-400">Value: <span className="text-slate-200">{d.feature_value ?? 'N/A'}</span></p>
      <p className={d.shap_value >= 0 ? 'text-blue-300' : 'text-emerald-300'}>
        Contribution: {d.shap_value >= 0 ? '+' : ''}{d.shap_value?.toFixed(4)}
      </p>
    </div>
  )
}

export default function ShapChart({ contributions = [], topK = 10 }) {
  const data = contributions
    .slice(0, topK)
    .map(c => ({
      feature: c.feature,
      shap_value: c.shap_value,
      feature_value: c.feature_value,
      abs: Math.abs(c.shap_value),
    }))
    .sort((a, b) => a.abs - b.abs)  // ascending for horizontal bar

  if (!data.length) return <p className="text-slate-500 text-sm">No explanation data.</p>

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#334e68' }}
          tickFormatter={v => v.toFixed(3)}
        />
        <YAxis
          type="category"
          dataKey="feature"
          width={180}
          tick={{ fill: '#cbd5e1', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
        <ReferenceLine x={0} stroke="#475569" strokeWidth={1} />
        <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.shap_value >= 0 ? POS_COLOR : NEG_COLOR} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
