# ICU Bed Assignment Optimization

Hybrid soft-computing project for Cognitive CodeQuest 2025. Objective: assign 40 ICU beds to 150 patients while maximizing survival benefit, balancing nurse workload, honoring specialty/ventilator constraints, and handling uncertainty.

## Dataset
- Data regenerated via `scripts/generate_dataset.py`, grounded in SCCM Fact & Figures, MIMIC-IV/eICU cohort summaries, and CDC ICU diagnosis mixes.
- `data/patients.csv` (150 rows) captures: demographics (age, sex, weight), comorbidities (count + Charlson), acuity scores (SOFA, APACHE-II), diagnosis group, admission type, ventilator/dialysis needs + probabilities, labs (lactate, MAP), LOS prediction, fuzzy inputs (risk, recommendation, uncertainty), and nurse intensity.
- `data/beds.csv` (40 rows) includes ICU type, specialty capability, ventilator/dialysis readiness, nurse capacity, isolation availability, and advanced monitoring support.
- Regenerate anytime with `python scripts/generate_dataset.py` (seeded for reproducibility); tweak script constants to mirror other regions/hospitals.

### Data Provenance & Assumptions
- **Acuity scores**: SOFA/APACHE-II distributions centered on values reported in SCCM Critical Care Statistics (median SOFA ≈ 7, APACHE-II ≈ 22 for mixed ICUs).
- **Ventilator usage**: Baseline probability 0.42 reflecting SCCM/eICU mechanical ventilation rates (40–45%), modulated by SOFA and ARDS/COVID diagnoses.
- **Diagnosis mix**: Sepsis, cardiac failure, neuro events, trauma, respiratory failure, renal failure, post-op care, and COVID-era ARDS proportions derived from CDC ICU surveillance briefs (2020–2023) and MIMIC-IV cohort summaries.
- **Nursing workload**: Nurse intensity and capacity follow SCCM staffing guidance (3.2–5.8 patients per nurse depending on acuity).
- **Uncertainty**: Elevated for emergency admissions to mimic incomplete documentation scenarios noted in eICU Collaborative Database metadata.

## Hybrid Modeling Blueprint
1. **Data ingestion** (`src/data_loader.py`)  
   - Load CSVs via pandas, normalize key features, compute derived ratios (e.g., ventilator shortfall risk, nurse load factors).  
   - Split patient features into *clinical severity*, *resource demand*, *data quality*, and *logistics* buckets for downstream explainability.
2. **Fuzzy inference** (`src/fuzzy_priority.py`)  
   - Triangular membership functions for severity (SOFA/APACHE), stability (MAP/lactate), and certainty (uncertainty column).  
   - Rule base yields `priority_score` and `survival_score` plus linguistic explanation fragments; defuzzify via centroid.
3. **Genetic Algorithm core** (`src/ga_optimizer.py`)  
   - Chromosome = vector length 40 (bed IDs) referencing patient IDs or `-1`.  
   - Fitness combines survival (sum of fuzzy survival scores), workload penalty (nurse intensity vs capacity), constraint penalties (ventilator/dialysis mismatches), and fairness (spread specialties).  
   - Operators: tournament selection, uniform crossover with repair, swap/mutate/resample operations guided by fuzzy priority queue.
4. **XAI layer** (`src/xai.py`)  
   - For each assignment, return score breakdown (priority contribution, workload impact, constraint delta) and highlight top fuzzy rules triggered.  
   - Global explanations: feature importance by averaging absolute contributions across assignments.
5. **Pipeline runner** (`src/run_pipeline.py`)  
   - Steps: load data → compute fuzzy scores → run GA for configurable generations → emit JSON/CSV outputs: bed assignment list, optimization score, textual method summary, conflict-resolution log.  
   - Optionally dump charts (matplotlib) illustrating nurse load balance and priority coverage.

### CLI Usage (planned)
```
python -m src.run_pipeline --patients data/patients.csv --beds data/beds.csv \
    --generations 120 --population 80 --report out/report.json
```
Outputs include assignment CSV, optimization metrics JSON, and XAI explanation pack.

## Streamlit App
- Launch UI with `streamlit run app/streamlit_app.py`.
- Upload custom patient/bed CSVs or rely on bundled defaults.
- Tune GA generations/population from sidebar, run optimizer, inspect metrics, conflicts, and download assignment/report artifacts for sharing.

## Hybrid GA + Fuzzy Strategy
1. **Feature normalization**  
   Scale/clip patient factors to [0,1]; derive composite severity, resource demand, and priority tiers.
2. **Fuzzy inference (patient priority)**  
   - Inputs: risk score, vitals severity, comorbidity tier, uncertainty.  
   - Linguistic sets: {low, medium, high} with triangular/trapezoidal membership functions.  
   - Rules capture expert heuristics (e.g., IF risk is high AND vitals high THEN priority = critical).  
   - Defuzzify to obtain `priority_score` driving GA fitness and tie-breaking.
3. **Chromosome representation**  
   Length 40 array (one gene per bed) storing assigned patient ID or -1 (empty). Companion lookup ensures each patient appears ≤ 1 time; unassigned patients tracked externally.
4. **Fitness function**  
   - Reward: sum of (priority_score * specialty match factor) + ventilator compliance boost.  
   - Penalties: unmatched ventilator need, nurse overload (quadratic penalty vs capacity), specialty mismatch, high-uncertainty assignments.  
   - Diversity term discourages over-allocation to single specialty.
5. **GA operators**  
   - Initialization via greedy heuristic biased by priority_score.  
   - Selection: tournament or roulette on normalized fitness.  
   - Crossover: uniform/one-point with repair operator to remove duplicates and reinsert high-priority unassigned patients.  
   - Mutation: swap two beds, replace gene with high-priority waitlist patient, or flip null assignment to relieve nurse overload.
6. **Hybrid loop**  
   After each GA generation, recompute fuzzy priority with updated uncertainty (e.g., degrade confidence for long-wait patients). Optionally apply local fuzzy-guided hill-climb on elite solutions.
7. **Outputs**  
   - Assignment table with patient + bed metadata and reason codes.  
   - Fitness breakdown (survival, workload, constraint penalties).  
   - Conflict-resolution explanation referencing fuzzy rules.

## Next Steps
1. Implement data loaders and preprocessing utilities in `src/data_loader.py`.
2. Define fuzzy system (e.g., with `scikit-fuzzy`) in `src/fuzzy_priority.py`.
3. Build GA engine in `src/ga_optimizer.py`; integrate with fuzzy module.  
4. Provide evaluation notebook (`notebooks/analysis.ipynb`) showing optimization score and visuals (workload distribution, priority coverage).
5. Package CLI entry point (e.g., `python -m src.run`) to produce final assignments + report JSON/CSV.

Let me know if you want me to start coding the modules or adjust the hybrid design.

