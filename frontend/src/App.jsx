import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import PatientAssessment from './pages/PatientAssessment'
import Cohort from './pages/Cohort'
import Explainability from './pages/Explainability'
import Fairness from './pages/Fairness'
import Monitoring from './pages/Monitoring'
import ModelRegistry from './pages/ModelRegistry'
import Methodology from './pages/Methodology'
import About from './pages/About'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="assess" element={<PatientAssessment />} />
        <Route path="cohort" element={<Cohort />} />
        <Route path="explain" element={<Explainability />} />
        <Route path="fairness" element={<Fairness />} />
        <Route path="monitoring" element={<Monitoring />} />
        <Route path="registry" element={<ModelRegistry />} />
        <Route path="methodology" element={<Methodology />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  )
}
