# ICU Bed Assignment Optimization – End-to-End Guide

This document walks through every component of the hybrid ICU allocator so you can understand and operate the project without digging through the entire repo. It follows the flow: competition brief → data → modeling → optimization → explainability → UI/CLI usage → deployment tips.

---

## 1. Problem Context

- **Competition brief**: Cognitive CodeQuest 2025 requests a soft-computing solution that assigns 40 ICU beds to 150 patients, maximizing survival benefit, balancing nurse workload, and handling uncertainty in patient data.
- **Key objectives**  
  1. Assign ICU beds optimally  
  2. Maximize survival score  
  3. Balance nurse workload  
  4. Prioritize critical patients  
  5. Handle uncertain data  
- **Deliverables**: Bed assignment list, optimization score, method explanation, conflict-resolution logic, plus a user-friendly interface (Streamlit) so non-technical stakeholders can run predictions.

---

## 2. Data Layer

### 2.1 Synthetic yet realistic dataset

- Located under `data/`: `patients.csv` (150 rows) and `beds.csv` (40 rows).
- Generated via `scripts/generate_dataset.py`, which is grounded in SCCM Fact & Figures, CDC ICU surveillance, and MIMIC-IV/eICU distributions.
- Patient features include demographics, comorbidities, SOFA/APACHE-II, diagnosis groups, ventilator/dialysis needs and probabilities, lab values, LOS predictions, risk and recommendation scores, nurse intensity, and data uncertainty.
- Bed features capture ICU type, specialty, ventilator/dialysis readiness, nurse capacity, isolation, and monitoring.

### 2.2 Feature engineering

| File | Purpose |
| --- | --- |
| `src/data_loader.py` | Loads CSVs via pandas, normalizes features, and computes derived measures like severity score, stability score, resource demand, data quality, and survival proxy. |
| `build_feature_blocks` | Groups clinical, resource, logistics, and data-quality features for XAI summaries. |

### 2.3 Refreshing data

```bash
python scripts/generate_dataset.py
```

Adjust constants (age mean, SOFA distribution, ventilator rates, diagnosis mix) to reflect a specific hospital’s census.

---

## 3. Fuzzy Prioritization Layer

- Implemented in `src/fuzzy_priority.py`.
- Uses triangular/trapezoidal membership functions for severity (SOFA/APACHE), stability (MAP/lactate), and certainty (uncertainty column).
- Rule base examples:  
  - IF severity high AND instability high → priority = critical  
  - IF severity medium AND certainty high → priority = urgent  
  - IF severity low AND stability high → priority = routine  
- Produces:
  - `priority_score` (0–1) with band {critical, urgent, routine}  
  - `survival_score` blending fuzzy stability/certainty with survival proxy  
  - `rule_trace` capturing activation strengths for explainability

---

## 4. Genetic Algorithm (GA) Optimization

- Core logic in `src/ga_optimizer.py`.
- **Chromosome representation**: Array of length 40 (one gene per bed) storing patient IDs or `-1` (vacant). Repair operators ensure no duplicate patients.
- **Fitness function**: Weighted sum of survival average, priority average, bed utilization, minus penalties for ventilator/dialysis violations, nurse overload, specialty mismatches, and unfair specialty distribution.
- **Operators**: Tournament selection, uniform crossover with repair, and mutation (swap or reassign from priority pool). Initial population biased by fuzzy priority queue.
- **Outputs**: Best chromosome, metrics (survival, priority, utilization, nurse ratio, penalties), conflict list, and a detailed assignment DataFrame describing each bed/patient match with reason codes.

---

## 5. Explainability (XAI)

- `src/xai.py` merges assignments with feature blocks to produce:
  - Feature influence summary (average normalized severity, stability, resource demand, data quality).
  - Priority mix (share of critical/urgent/routine assignments).
  - Conflict log describing each specialty/equipment exception.
  - Method explanation string combining metrics and fuzzy rationale, used in reports/UI.

---

## 6. Pipeline Runner & CLI

- `src/run_pipeline.py` exposes two entry points:
  - `run_pipeline(...)` Python function returning `(assignments_df, report_dict)`. Facilitates programmatic or UI usage.
  - CLI via `python -m src.run_pipeline --patients data/patients.csv --beds data/beds.csv --generations 120 --population 80 --assignment_csv out/assignments.csv --report_json out/report.json`
- CLI output:
  - CSV `out/assignments.csv` with bed-level decisions + reasons.
  - JSON `out/report.json` containing optimization metrics, method explanation, conflict-resolution log, and pointer to the CSV.

---

## 7. Streamlit Web App

- Located at `app/streamlit_app.py`.
- Features:
  - Gradient-themed minimalist UI with card layout.
  - File uploaders for patient/bed CSVs (defaults fall back to `data/`).
  - Sidebar sliders for GA generations and population, and verbose toggle for console logs.
  - Metrics panel, plain-language summary, detailed table, narrative explanations for each patient, conflict log context, and three download buttons (assignments, report, layman explanations).
  - Path bootstrapping ensures `src` imports work even when Streamlit is run outside the repo.
- Run with:

```bash
pip install -r requirements.txt  # ensure streamlit is installed
streamlit run app/streamlit_app.py
# or: python -m streamlit run app/streamlit_app.py
```

If `streamlit` command isn’t recognized, run via `python -m streamlit` or add Python Scripts directory to PATH.

---

## 8. Outputs & Interpretation

1. **Assignment CSV**  
   - Each row includes bed info, assigned patient ID, patient specialty, priority/survival scores, ventilator/dialysis needs, nurse intensity, and human-readable reason (e.g., “specialty match; ventilator provided; priority 0.82”).  
   - Vacant beds are explicitly labeled with explanation.

2. **Report JSON**  
   - `optimization_score`: survival/priority averages, utilization, nurse ratio, penalty breakdowns.  
   - `method_explanation`: prose summary referencing fuzzy+GA hybrid approach and metric values.  
   - `conflict_resolution`: list of specialty/equipment mismatches that require manual review.

3. **Layman explanations CSV (from Streamlit)**  
   - Presents each assignment in plain language (urgency, specialty context, equipment needs, rationale).

---

## 9. Extending or Customizing

| Need | Suggested Action |
| --- | --- |
| Enforce strict specialty matches | Increase mismatch penalty in `BedAssignmentGA._evaluate` or modify `_select_candidate_for_bed` to filter by specialty. |
| Different hospital census | Adjust generation constants or load real CSV exports; the pipeline doesn’t assume synthetic data. |
| Alternate soft-computing method | Swap GA with PSO/ACO inside `ga_optimizer.py`, leveraging the same fuzzy scores. |
| Visualization | Add matplotlib/plotly charts (bed utilization bars, nurse load histogram) either in CLI outputs or Streamlit. |
| Deployment | Containerize with Docker (Python + Streamlit), expose via HTTPS, and secure file uploads. |

---

## 10. Quick Start Checklist

1. **Install dependencies** (Python ≥3.9, pandas, numpy, streamlit, etc.).
2. **Generate or load data** (`python scripts/generate_dataset.py` or replace CSVs).
3. **Run CLI** to validate pipeline and produce baseline outputs.
4. **Launch Streamlit app** for interactive use and plain-language reporting.
5. **Review conflict log** and adjust penalties/constraints to match hospital policy.
6. **Document decisions** using the generated CSV/JSON/layman explanation exports.

With this guide, you should be able to navigate every layer (data, fuzzy logic, GA, XAI, UI) and operate or extend the ICU bed assignment project confidently.

