const BASE_URL = import.meta.env.VITE_API_URL || ''

const BASE = '/v1'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }

  return res.json()
}

export const api = {
  health: () => request('/health'),

  modelInfo: () => request(`${BASE}/model`),

  fairnessReport: () => request(`${BASE}/fairness`),

  predict: (patient) =>
    request(`${BASE}/predict`, {
      method: 'POST',
      body: JSON.stringify(patient),
    }),

  explain: (patient) =>
    request(`${BASE}/explain`, {
      method: 'POST',
      body: JSON.stringify(patient),
    }),

  demoPatients: () => request(`${BASE}/demo/patients`),

  demoCohort: () => request(`${BASE}/demo/cohort`),

  driftCheck: (simulate = false) =>
    request(`${BASE}/drift/check?simulate=${simulate}`, {
      method: 'POST',
    }),

  metrics: () => request('/metrics'),
}