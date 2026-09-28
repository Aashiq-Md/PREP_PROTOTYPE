import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard, UserSearch, Users, Lightbulb,
  Scale, Activity, Database, BookOpen, Info, FlaskConical,
} from 'lucide-react'

const NAV = [
  { to: '/',            icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/assess',      icon: UserSearch,      label: 'Patient Assessment' },
  { to: '/cohort',      icon: Users,           label: 'Cohort' },
  { to: '/explain',     icon: Lightbulb,       label: 'Explainability' },
  { to: '/fairness',    icon: Scale,           label: 'Fairness' },
  { to: '/monitoring',  icon: Activity,        label: 'Monitoring' },
  { to: '/registry',    icon: Database,        label: 'Model Registry' },
  { to: '/methodology', icon: BookOpen,        label: 'Methodology' },
  { to: '/about',       icon: Info,            label: 'About' },
]

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-navy-950 border-r border-navy-800 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-navy-800">
          <div className="flex items-center gap-2">
            <FlaskConical size={20} className="text-blue-400" />
            <span className="text-lg font-bold text-white tracking-tight">PREP</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-0.5 leading-tight">
            Predict Readmission<br />Estimation of Patient
          </p>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-300 font-medium'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-navy-800'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Prototype badge */}
        <div className="px-4 py-3 border-t border-navy-800">
          <div className="bg-amber-950/50 border border-amber-800/40 rounded-md px-2 py-1.5">
            <p className="text-[10px] text-amber-400/80 font-medium">RESEARCH PROTOTYPE</p>
            <p className="text-[9px] text-amber-500/60 mt-0.5">Not clinically validated</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-navy-950">
        <Outlet />
      </main>
    </div>
  )
}
