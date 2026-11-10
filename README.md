# EduGuard AI
### Academic Intelligence & Early Warning Platform
**Tagline:** *Detect Early. Understand Deeply. Support Intelligently.*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/Champion-XGBoost%20(Calibrated)-EC4899.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-7C3AED.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Prediction Target](#3-prediction-target)
4. [System Architecture](#4-system-architecture)
5. [Authoritative Feature Engineering](#5-authoritative-feature-engineering)
6. [Technology Stack](#6-technology-stack)
7. [Repository Structure](#7-repository-structure)
8. [Synthetic Dataset & Validation Suite](#8-synthetic-dataset--validation-suite)
9. [Exploratory Data Analysis (EDA)](#9-exploratory-data-analysis-eda)
10. [Anti-Leakage Architecture](#10-anti-leakage-architecture)
11. [Model Comparison, Calibration & Programmatic Selection](#11-model-comparison-calibration--programmatic-selection)
12. [Dual Validation & Held-Out Test Benchmarks](#12-dual-validation--held-out-test-benchmarks)
13. [Explainable AI (SHAP) vs. Rule-Based Support](#13-explainable-ai-shap-vs-rule-based-support)
14. [FastAPI Backend Service](#14-fastapi-backend-service)
15. [Enchanted Multiverse React Frontend (Six Realms)](#15-enchanted-multiverse-react-frontend-six-realms)
16. [Streamlit Dashboard (Fallback)](#16-streamlit-dashboard-fallback)
17. [Jupyter Exploratory & Reliability Notebook](#17-jupyter-exploratory--reliability-notebook)
18. [Automated Test Suite & CI](#18-automated-test-suite--ci)
19. [Docker Packaging & Deployment Status](#19-docker-packaging--deployment-status)
20. [System Limitations & Real-World Transfer Caveats](#20-system-limitations--real-world-transfer-caveats)
21. [Responsible AI & Governance](#21-responsible-ai--governance)
22. [Future Enhancements](#22-future-enhancements)

---

## 1. Project Overview
**EduGuard AI** is a reproducible, advisory machine learning platform built to help school teachers and academic coordinators identify students who may benefit from timely academic support before an upcoming assessment takes place.

Trained solely on simulated longitudinal pre-assessment classroom indicators (`seed=42`, `2,500` school students, `10,000` prediction snapshots), EduGuard AI provides calibrated risk probabilities, game-theoretic `TreeExplainer` SHAP attributions, persisted teacher intervention tracking, downloadable CSV cohort reports, and an immersive Six-World React interface backed by FastAPI.

---

## 2. Problem Statement
In school settings, waiting until after a unit test or term exam fails reduces the window for constructive support. EduGuard AI models early warning as a cost-sensitive binary prediction problem:
- **False Negative (FN) Cost — High:** A struggling student is missed and receives no proactive mentoring or extra help prior to the next assessment.
- **False Positive (FP) Cost — Moderate:** A teacher conducts a brief, encouraging academic check-in or offers study-hall resources to a student who ultimately scores $\ge 40$.

---

## 3. Prediction Target
$$\text{Target: } \text{at\_risk} = \begin{cases} 1 & \text{if Next Assessment Score } < 40/100 \\ 0 & \text{if Next Assessment Score } \ge 40/100 \end{cases}$$

Only pre-assessment information available prior to that future evaluation enters the pipeline. The outcome column (`next_assessment_score`) is strictly isolated during target derivation and excluded from feature matrices.

---

## 4. System Architecture
```
[ Synthetic Cohort Simulation ] (seed=42, 2,500 students, 10,000 snapshots)
             │
             ▼
[ Authoritative Feature Engineering ] (post-missingness rolling 2-score mean)
             │
             ▼
[ Validation & Anti-Leakage Audit ] (reports/data_quality_report.json -> PASS)
             │
             ▼
[ Dual Validation Splits ] (Train / Val-Cal / Val-Sel / Held-Out Test)
             │
             ▼
[ Sigmoid Calibration + Composite Selection ] (Champion: XGBoost, tau* = 0.18)
             │
             ▼
[ Serialized Artifacts ] (models/eduguard_pipeline.joblib, model_metadata.json)
             │
             ▼
[ FastAPI Backend (:8000) ] ◄── REST + SHAP + Interventions + CSV Reports
      │               │
      ▼               ▼
[ React 6-World     [ Streamlit Dashboard (:8501) ]
  Multiverse UI       (7 Fallback Views)
  (:5173) ]
```

---

## 5. Authoritative Feature Engineering
All components (`src/data_generator.py`, `src/train.py`, `src/predict.py`, `api/main.py`, `dashboard/app.py`, and `frontend/src/utils/features.ts`) share a single canonical rule implemented in `src/feature_engineering.py`:
- **`previous_score_avg`**: **Rolling two-assessment mean** computed post-missingness as $(\text{previous\_score} + \text{previous\_score\_2}) / 2$, falling back cleanly to `previous_score` when `previous_score_2` is unavailable (`NaN`).
- **`score_trend`**: $\text{previous\_score} - \text{previous\_score\_2}$ (`0.0` when `previous_score_2` is missing).

Computing this rule identically after missingness injection eliminates training-serving skew across synthetic batch splits and live UI predictions (`0` mismatches verified in `reports/data_quality_report.json`).

---

## 6. Technology Stack
- **Languages:** Python 3.11.5 · TypeScript 5.7
- **Data Processing:** Pandas `2.2.3`, NumPy `2.2.1`
- **Machine Learning:** Scikit-Learn `1.7.2` (DummyClassifier, LogisticRegression, RandomForestClassifier, CalibratedClassifierCV with `FrozenEstimator`), XGBoost `3.2.0`
- **Explainable AI:** SHAP `0.51.0` (`TreeExplainer` in probability space)
- **API Backend:** FastAPI `0.162.0`, Pydantic `2.14.3`, Uvicorn `0.54.0`
- **Primary Frontend:** React 19, Vite 6, Tailwind CSS 3, Framer Motion 12, Recharts 3, Lucide Icons, Axios
- **Fallback Dashboard:** Streamlit `1.67.0`, Plotly `7.1.0`, Matplotlib `3.10.0`, Seaborn `0.13.2`
- **Testing & Notebook:** Pytest `9.2.0`, HTTPX `0.27.0`, Jupyter / nbconvert
- **Persistence & Packaging:** Joblib `1.5.2`, JSON, Docker & Docker Compose, GitHub Actions

---

## 7. Repository Structure
```
EduGuard-AI/
├── api/
│   └── main.py                         # FastAPI endpoints, strict validation, interventions & CSV reports
├── dashboard/
│   └── app.py                          # Synchronized 7-section Streamlit fallback dashboard
├── data/
│   ├── raw/student_performance.csv     # 10,000 synthetic school student snapshots
│   ├── processed/                      # Split datasets + persisted intervention_notes.json
│   └── README.md
├── frontend/
│   ├── .env.example                    # Frontend API URL configuration template
│   ├── Dockerfile                      # Multi-stage Node 20 -> Nginx SPA + proxy image
│   ├── nginx.conf                      # Reverse proxy (/api -> FastAPI) & SPA fallback
│   ├── screenshots/                    # Verified live headless Chrome captures across all realms
│   └── src/
│       ├── api/client.ts               # Typed API client with structured error parsing
│       ├── components/                 # Navigation, ParticleField, RiskGauge, SHAPChart, Backdrops
│       ├── constants/worlds.ts         # Six-World metadata & feature labels
│       ├── pages/                      # KingdomGate + World1..World6 modules
│       ├── types/index.ts              # Backend-aligned TypeScript schemas
│       └── utils/features.ts           # Canonical rolling-2 feature parity helpers
├── models/
│   ├── eduguard_pipeline.joblib        # Serialized Sigmoid-Calibrated XGBoost Pipeline
│   └── model_metadata.json             # Programmatic selection, calibration & test metrics
├── notebooks/
│   └── 01_EDA_and_Model_Analysis.ipynb # Executed EDA, validation & calibration notebook
├── reports/
│   ├── figures/                        # Figures 01..08 (EDA), 09..10 (SHAP), 11 (Calibration)
│   ├── data_quality_report.json        # Automated validation & parity report
│   ├── evaluation_metrics.json         # Full benchmark, period & error audit telemetry
│   ├── technical_report.md             # Empirical ML & engineering report
│   └── demo_video_script.md            # Presentation walkthrough script
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── leakage_audit.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── generate_eda_figures.py
│   └── generate_report.py
├── tests/                              # Comprehensive Pytest verification suite
├── .dockerignore
├── .env.example
├── .github/workflows/tests.yml
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 8. Synthetic Dataset & Validation Suite
- **Volume:** `2,500` unique simulated school students × `4` longitudinal snapshots = `10,000` records (`seed=42`).
- **Class Balance:** `80.46%` Passing (`8,046` snapshots $\ge 40$) vs. `19.54%` At-Risk (`1,954` snapshots $< 40$).
- **Realistic Missingness:** MCAR/MAR missingness in `attendance_pct` (`1.50%`), `assignment_completion_pct` (`2.01%`), and `previous_score_2` (`0.35%`) handled inside the pipeline's median imputer.
- **Validation Audit (`PASS`):** Zero duplicate `(student_id, assessment_period)` keys, zero infinite/out-of-range values, `0` derived-feature mismatches, `0` target-derivation mismatches.
```bash
python -m src.data_generator --seed 42
python -m src.data_validation
```

---

## 9. Exploratory Data Analysis (EDA)
High-resolution figures generated into `reports/figures/`:
1. `01_target_distribution.png` — Class distribution (`80.5%` vs `19.5%`)
2. `02_attendance_distribution.png` — Attendance rate distribution across risk tiers
3. `03_assignment_completion_distribution.png` — Coursework completion divergence
4. `04_previous_score_distribution.png` — Score separation across passing and at-risk groups
5. `05_score_trends_over_periods.png` — Longitudinal trajectories across Periods 1–4
6. `06_correlation_heatmap.png` — Pearson feature correlations (max target $|r| = 0.6119$)
7. `07_missing_values_analysis.png` — Missingness diagnostics
8. `08_outlier_analysis.png` — Behavioral quadrant inspection

---

## 10. Anti-Leakage Architecture
`src/leakage_audit.py` verifies:
- `next_assessment_score`, `at_risk`, and `student_id` never appear in model feature matrices (`X`).
- Student-disjoint splits guarantee zero shared `student_id` overlap across `train` / `val_cal` / `val_sel` / `test`.
- Zero sensitive or protected attributes exist in schema or data (`gender`, `race`, `ethnicity`, `caste`, `religion`, `income`, `disability`, `name`, `age`).
- Run audit: `python -m src.leakage_audit`

---

## 11. Model Comparison, Calibration & Programmatic Selection
To eliminate in-sample calibration optimism and prevent selection leakage, validation folds are split cleanly into disjoint **calibration (`val_cal`)** and **selection (`val_sel`)** partitions. Candidate models are fit on `train`, sigmoid-calibrated via `CalibratedClassifierCV(FrozenEstimator(...), method="sigmoid")` on `val_cal`, and ranked strictly on `val_sel`:

$$\text{Composite} = 0.35 \cdot \overline{\text{AP}} + 0.25 \cdot \overline{\text{ROC-AUC}} + 0.25 \cdot F_{2,\text{opt}} + 0.15 \cdot (1 - \overline{\text{Brier}})$$

| Model Architecture | Temporal Val AP | Disjoint Val AP | Temporal Val ROC | Temporal Val F2 (Opt) | Mean Brier (Post-Cal) | Composite Score | Latency / Record |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | 0.1912 | 0.2154 | 0.5000 | — | 0.1625 | Baseline | 0.004 ms |
| **LogisticRegression** (Calibrated) | 0.8420 | 0.8459 | 0.9501 | 0.8183 | 0.0696 | 0.8764 | 0.005 ms |
| **RandomForestClassifier** (Calibrated) | 0.8459 | 0.8489 | 0.9528 | 0.8141 | 0.0743 | 0.8765 | 0.108 ms |
| **XGBoost** (Calibrated · **Champion**) | **0.8515** | **0.8599** | **0.9538** | **0.8201** | **0.0701** | **0.8818** | **0.009 ms** |

---

## 12. Dual Validation & Held-Out Test Benchmarks
- **Temporal Holdout Split:** Train Periods 1–2 (`5,000`) | Val-Cal Period 3a (`1,250`) | Val-Sel Period 3b (`1,250`) | **Test Period 4 (`2,500`)**.
- **Student-Disjoint Holdout Split:** Train (`7,000`) | Val-Cal (`748`) | Val-Sel (`752`) | **Test Unseen Students (`1,500`)**.
- **Threshold Optimization:** Selected **$\tau^* = 0.18$** strictly on `t_val_sel` (`Recall 90.8%`, `Precision 59.1%`, `Specificity 85.2%`, `F2 0.8201`).

### Held-Out Test Performance ($\tau^* = 0.18$)
| Held-Out Evaluation Split | Sample $N$ | Positive Rate | Avg Precision (AP) | Trap PR-AUC | ROC-AUC | Recall | Precision | Specificity | F1 | F2 | Calibrated Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Held-Out Test (Period 4)** | 2,500 | 15.76% | **0.8064** | 0.8062 | **0.9420** | **89.8%** | 52.2% | 84.6% | 0.6604 | 0.7853 | **0.0677** *(was 0.0959)* |
| **Student-Disjoint Test (Unseen)** | 1,500 | 21.00% | **0.8384** | 0.8380 | **0.9448** | **82.9%** | 68.9% | 90.0% | 0.7522 | 0.7962 | **0.0723** |

> **Calibration & Base-Rate Transparency:** Validation-split sigmoid calibration reduced Temporal Test Brier error from `0.0959` (`ECE 0.1172`) down to `0.0677` (`ECE 0.0427`) — visualized in `reports/figures/11_calibration_curve.png`. Temporal Test precision (`52.2%`) is lower than Period 3 validation precision (`59.1%`) primarily because Period 4's underlying base risk rate decreases from `19.16%` to `15.76%`, rather than threshold overfitting. Error analysis across the `40` Period 4 false negatives (`1.60%` of test rows) shows borderline scores averaging `51.7/100` prior to a sharp unexpected drop (`mean next score 36.0`).

---

## 13. Explainable AI (SHAP) vs. Rule-Based Support
EduGuard AI separates **model-level explanations** from **pedagogical recommendations**:
1. **Statistical SHAP Explanations (`src/explain.py`):** Evaluates the unwrapped `XGBClassifier` with `shap.TreeExplainer(..., model_output="probability")` on a `150`-row cohort background. Global Mean `|SHAP|`:
   - `previous_score_avg` (`0.1284`) > `previous_score` (`0.0889`) > `previous_score_2` (`0.0589`) > `assessment_period` (`0.0130`) > `score_trend` (`0.0086`) > `assignment_completion_pct` (`0.0070`) > `late_assignment_count` (`0.0033`) > `attendance_pct` (`0.0032`).
   - *SHAP figures (`09_shap_summary.png`, `10_shap_bar.png`) show predictive feature impact, not causal mechanisms.*
2. **Rule-Based Teacher Support Plans (`src/predict.py`):** Deterministic non-punitive rules tagged `"source": "rule_based_advisory"` translate risk thresholds and low attendance/completion flags into mentoring suggestions for human teachers.

---

## 14. FastAPI Backend Service
### Start Backend Locally
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`

### Available Endpoints
- `GET /` & `GET /health` — Model health, active champion (`XGBoost`), calibration status, and threshold `0.18`.
- `GET /model-info` & `GET /audit-logs` — Validation/held-out metrics, AP/Specificity/Brier telemetry, calibration curve bins, period breakdown, and exclusion vault.
- `POST /predict` & `POST /predict/batch` — Strict Pydantic input validation (rejects booleans, non-finite `NaN`/`Inf`, out-of-range metrics, invalid periods outside `{1..4}`) + authoritative rolling-2 derivation + SHAP attribution.
- `GET /students` & `GET /students/{student_id}` — Filterable student archive & longitudinal profile.
- `GET /cohort-summary` & `GET /analytics/global` — Cohort trends, attendance/completion risk bands, and Global SHAP/constellation sample.
- `GET /interventions` & `POST /interventions` — Teacher intervention queue & persisted follow-up notes (`data/processed/intervention_notes.json`).
- `GET /reports/students` — Filterable student risk report (`format=json` or streaming `format=csv` export).

### Sample Request & Verified Model Response
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "attendance_pct": 68.0,
       "assignment_completion_pct": 55.0,
       "late_assignment_count": 3,
       "previous_score": 38.0,
       "previous_score_2": 45.0,
       "assessment_period": 3
     }'
```
Verified output from the calibrated XGBoost pipeline:
```json
{
  "prediction": 1,
  "prediction_label": "At-Risk (< 40)",
  "risk_probability": 0.3576,
  "applied_threshold": 0.18,
  "risk_band": {
    "tier": "HIGH_RISK",
    "label": "High Risk / Priority Advisory",
    "color": "#EF4444",
    "action": "Proactive Academic Support Recommended"
  },
  "model_name": "XGBoost",
  "model_version": "1.1.0",
  "is_calibrated": true
}
```

---

## 15. Enchanted Multiverse React Frontend (Six Realms)
```bash
cd frontend
npm install
npm run dev
```
Interactive Experience URL: `http://localhost:5173` (copy `frontend/.env.example` to `.env.local` if using a non-default backend host).

| Realm | Route | Academic Capabilities |
| :--- | :---: | :--- |
| **Kingdom Gate** | `/` | Entrance portal with live backend telemetry pill & retry support |
| **World 1: Enchanted Kingdom** | `/kingdom` | Cohort KPIs (total students, risk rate, attendance, completion), passing vs. at-risk trajectories |
| **World 2: Wizarding Academy** | `/academy` | Student Dossiers + Trajectory Charts, Interactive Single SHAP Oracle (live derived preview), Classroom CSV/Inline Batch Scorer, Queue-to-Intervention workflow |
| **World 3: Nature Realm** | `/nature` | Attendance & coursework completion risk tiers, longitudinal river of progress, SHAP vs. Rule-Based separator |
| **World 4: Cyberpunk Future** | `/cyberpunk` | Temporal vs. Disjoint Observatory (AP, Specificity, Brier Reliability Curve, Period 1–4 stability, Error Audit), What-If Simulator, & Teacher Intervention Center |
| **World 5: Celestial Multiverse** | `/celestial` | Student risk constellation scatter map, global `|SHAP|` rankings & Pearson correlations with non-causal framing |
| **World 6: Oceanic Crystal Realm** | `/ocean` | Responsible AI covenants, exclusion vault, evaluation ledger & downloadable CSV student/period risk reports |

Screenshots stored in `frontend/screenshots/` verify rendered desktop and interactive flows across all realms.

---

## 16. Streamlit Dashboard (Fallback)
```bash
streamlit run dashboard/app.py
```
Available at `http://localhost:8501`, synchronized to read live artifact metadata, `XGBoost` benchmarks, and canonical `compute_derived_features` transforms.

---

## 17. Jupyter Exploratory & Reliability Notebook
Interactive notebook executed and saved with cell outputs and charts:
- `notebooks/01_EDA_and_Model_Analysis.ipynb`

---

## 18. Automated Test Suite & CI
Run the full unit, ML parity, SHAP, and API test suite:
```bash
pytest -v tests/
```
CI workflow defined in `.github/workflows/tests.yml` runs data generation, data validation, leakage audit, training, and `pytest`.

---

## 19. Docker Packaging & Deployment Status
Multi-container configuration is provided across:
- `Dockerfile` (Python 3.11 FastAPI backend & Streamlit service)
- `frontend/Dockerfile` + `frontend/nginx.conf` (Node 20 Vite build served via Nginx reverse proxy)
- `docker-compose.yml` (`api` on `:8000`, `frontend` on `:5173`, `dashboard` on `:8501`)

> **Environment Limitation Note:** `docker` / `docker-compose` binaries are not installed on the authoring Windows machine, so live container builds (`docker compose up --build`) could not be executed locally and remain unverified at container runtime. All Python, Vite/TypeScript, and FastAPI services were built and verified natively.

---

## 20. System Limitations & Real-World Transfer Caveats
1. **Unverified Real-World Generalization:** All training, calibration, and test metrics come from a **100% synthetic generator**. Real school deployments must not extrapolate synthetic accuracy/recall numbers without empirical re-validation on local historical school cohorts.
2. **Associative, Not Causal:** Neither SHAP explanations nor the What-If Simulator prove causal outcomes from teacher interventions; they inspect statistical relationships learned from synthetic data.
3. **Scope Constraints:** Designed strictly for `[0, 100]` numeric assessments with a `< 40` pass benchmark across Periods 1–4.

---

## 21. Responsible AI & Governance
- **Advisory Only:** Tool for school teachers and academic coordinators; **must never** assign grades, issue disciplinary sanctions, restrict course access, or replace human conversation.
- **Zero Demographic Bias:** Protected attributes are absent by design (`src/leakage_audit.py`).
- **License:** Distributed under the [MIT License](LICENSE).

---

## 22. Future Enhancements
1. Optional district SIS / Google Classroom read-only LMS adapters with strict IRB/consent workflows.
2. Longitudinal fairness & drift monitoring dashboards across academic terms.
3. Multi-threshold early-warning profiles tailored to distinct grade levels.
