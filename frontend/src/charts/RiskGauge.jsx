import { PieChart, Pie, Cell } from 'recharts'

const RADIAN = Math.PI / 180

function needle(value, cx, cy, iR, oR, color) {
  const ang = 180 - value * 180
  const sin = Math.sin(-RADIAN * ang)
  const cos = Math.cos(-RADIAN * ang)
  const r = 5
  const x0 = cx, y0 = cy
  const xba = x0 + r * sin, yba = y0 - r * cos
  const xbb = x0 - r * sin, ybb = y0 + r * cos
  const xp = x0 + (oR - 10) * cos, yp = y0 + (oR - 10) * sin
  return (
    <g>
      <circle cx={x0} cy={y0} r={r} fill={color} stroke="none" />
      <path d={`M${xba} ${yba} L${xbb} ${ybb} L${xp} ${yp} Z`} fill={color} />
    </g>
  )
}

export default function RiskGauge({ risk = 0 }) {
  const cx = 120, cy = 110, iR = 60, oR = 100
  const data = [
    { value: 20, color: '#10b981' },   // lower  0–20%
    { value: 20, color: '#f59e0b' },   // moderate 20–40%
    { value: 60, color: '#ef4444' },   // higher 40–100%
  ]
  const pct = Math.round(risk * 100)
  const needleVal = Math.min(Math.max(risk, 0), 1)

  return (
    <div className="flex flex-col items-center">
      <PieChart width={240} height={130}>
        <Pie
          data={data}
          cx={cx} cy={cy}
          startAngle={180} endAngle={0}
          innerRadius={iR} outerRadius={oR}
          dataKey="value"
          stroke="none"
        >
          {data.map((d, i) => <Cell key={i} fill={d.color} />)}
        </Pie>
        {needle(needleVal, cx, cy, iR, oR, '#e2e8f0')}
      </PieChart>
      <p className="text-4xl font-bold text-white -mt-2">{pct}%</p>
      <p className="text-xs text-slate-400 mt-1">Estimated 30-day readmission risk</p>
    </div>
  )
}
