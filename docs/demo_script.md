# PREP Hackathon Demo Script
## 3–5 Minute Presentation Flow

---

## Scene 1 — Problem (30 seconds)

> "Hospital readmissions within 30 days affect millions of patients and cost billions annually. Existing prediction tools have three critical gaps: they're black boxes, they don't check fairness, and they have no way to detect when they stop working."

Show: Dashboard → problem statement

---

## Scene 2 — PREP (30 seconds)

> "PREP addresses all three gaps with four pillars:"

Point to the four pillars on the dashboard:
- **PREDICT** — estimate risk at discharge
- **EXPLAIN** — show why
- **AUDIT** — check fairness
- **MONITOR** — detect drift

---

## Scene 3 — Patient Assessment (60 seconds)

Navigate to: Patient Assessment

> "Let's assess a high-risk patient."

Click: "Higher Risk" demo patient

Show the form auto-filling with:
- Age 78, Male, Medicaid
- 3 prior admissions, 12-day stay
- Charlson 7, 15 medications, high-risk med, emergency admission

Click: "Generate Risk Estimate"

> "PREP estimates a higher readmission risk — above the threshold."

---

## Scene 4 — Explanation (60 seconds)

The explanation chart appears automatically. The API labels trained-model SHAP
output separately from demo heuristic contributions.

> "PREP also shows a labeled explanation aid. In demo mode these contributions are heuristic examples, not SHAP values or model attributions."

Point to the demo heuristic chart:
- Prior admissions: largest positive contribution
- High-risk medication: second
- Length of stay: third

> "These demo contributions are not genuine SHAP values and do not explain a trained model. Nothing here establishes causation or clinical utility."

Navigate to: Explainability → load "Strong Explainability" case

---

## Scene 5 — Cohort (30 seconds)

Navigate to: Cohort

> "PREP can assess an entire cohort at once. Here we can filter by risk band, sort by estimated risk, and identify patients above the threshold."

Filter: Risk Band = Higher

---

## Scene 6 — Fairness (45 seconds)

Navigate to: Fairness

> "PREP can audit fairness across race, gender, age, and insurance when saved evaluation data is available. Synthetic and insufficient-event results are labeled and cannot produce a PASS."

Click through dimensions.

> "Critically — race is never a model input. It's used for auditing only."

Point to the audit status and evidence counts; use INDETERMINATE when evidence is insufficient.

---

## Scene 7 — Monitoring (45 seconds)

Navigate to: Monitoring

Click: "Check Stable Data"
> "No drift detected — the model is stable."

Click: "Simulate Data Drift"
> "Now we introduce a distribution shift — older patients, more prior admissions, longer stays."

Point to: DRIFT DETECTED banner, red bars above threshold.

> "PREP catches this automatically. Without monitoring, you'd never know your model had degraded."

---

## Scene 8 — Closing (30 seconds)

Navigate back to: Dashboard

> "PREP moves beyond prediction by making the model explainable, auditable, and observable."

> "Four pillars. One system. Built for trustworthy clinical AI."

Point to the four pillars one final time.

---

## Key Messages for Q&A

- "Race is excluded from the model — it's audit-only"
- "Threshold is selected on validation data, never test data"
- "Predictions in this prototype are logged with hashed inputs; this is not a substitute for a production privacy review"
- "This is a prototype — not clinically validated; demo results are synthetic"
- "The reference study validated on 415,000 MIMIC-IV admissions — we built on that foundation"
