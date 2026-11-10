# EduGuard AI — 5-Minute Demonstration Script
**Target Duration:** 4 minutes 45 seconds  
**Presenter:** AI/ML & Full-Stack Engineering Intern (ALGIONS AI LABS Assessment)  
**Visual Focus:** Six-World React + TypeScript Experience (`http://localhost:5173`), FastAPI Swagger UI (`http://localhost:8000/docs`), Jupyter Notebook, and Automated `pytest` Suite.

---

### [0:00–0:35] Introduction & Responsible Advisory Mandate
**Visual:** Open **Kingdom Gate (`/`)** showcasing the animated particle backdrop and live backend health pill, then click **ENTER THE KINGDOM** to open **World 1 (`/kingdom`)**.  
**Voiceover:**
> *"Hello! In school education, teachers often discover a student needs academic support too late—right after a failed unit exam. Today I'm presenting **EduGuard AI: Academic Intelligence & Early Warning Platform**.*
>
> *EduGuard AI is an advisory ML platform that forecasts whether a school student is at risk of scoring below **40 out of 100** in their upcoming assessment using only pre-assessment classroom indicators. Crucially, the system is strictly advisory: it is designed to help teachers prioritize supportive check-ins and mentoring, and must never automatically grade, discipline, or replace teacher judgment."*

---

### [0:35–1:15] Synthetic Dataset, Derived-Feature Consistency & Anti-Leakage Safeguards
**Visual:** Display `reports/figures/06_correlation_heatmap.png` and `reports/data_quality_report.json` alongside `src/feature_engineering.py`.  
**Voiceover:**
> *"To protect student privacy and keep experiments 100% reproducible, EduGuard AI simulates 2,500 school students across four longitudinal prediction periods—10,000 snapshots seeded with random seed 42.*
>
> *Every input passes through a single authoritative feature-engineering module (`src/feature_engineering.py`), computing `previous_score_avg` as a consistent rolling two-assessment mean after missingness injection—eliminating training-serving skew across offline training, FastAPI endpoints, React forms, and batch scoring. Automated leakage and data-quality audits verify zero target contamination, zero demographic or protected attributes, and zero derived-feature mismatches."*

---

### [1:15–2:05] Dual Validation, Probability Calibration & Programmatic Champion Selection
**Visual:** Navigate to **World 4 (`/cyberpunk` — Reliability Observatory tab)** showing multi-model comparison benchmarks, calibration reliability diagram, and threshold optimization curve.  
**Voiceover:**
> *"Instead of relying on naive random splits, EduGuard AI validates across two holdout regimes: a **Temporal Split** predicting future Period 4 outcomes, and a **Student-Disjoint Split** evaluating completely unseen students.*
>
> *Each validation split is cleanly divided into a calibration fold (`val_cal`) and threshold selection fold (`val_sel`). We benchmark DummyClassifier, Logistic Regression, Random Forest, and regularized XGBoost with validation-split sigmoid calibration. Our programmatic selector chooses **XGBoost (Calibrated)** cleanly on composite validation metrics (`0.8818`), selecting threshold **tau* = 0.18** via F2-utility optimization.*
>
> *On the held-out Temporal Test set, calibrated XGBoost cuts Brier error from `0.0959` down to `0.0677`, achieving **0.806 Average Precision**, **0.942 ROC-AUC**, and **89.8% recall**, while generalizing to **82.9% recall** and **68.9% precision** on unseen students."*

---

### [2:05–3:05] Single & Batch Student Intelligence, SHAP Explainability & Non-Causal Distinction
**Visual:** Navigate to **World 2 (`/academy`)**: show a searchable student dossier, then open **Single SHAP Oracle** and adjust sliders; briefly switch to **Batch Risk Scorer**. Also peek at **World 5 (`/celestial`)** Global SHAP Constellations.  
**Voiceover:**
> *"In the **Wizarding Academy**, teachers can search any student dossier to review longitudinal scores and classroom engagement trajectories, run interactive hypothetical scores in the Single SHAP Oracle, or upload a CSV in the **Batch Risk Scorer**.*
>
> *Every prediction includes TreeExplainer SHAP attributions in probability space—highlighting risk-elevating indicators alongside protective buffers—while clearly separating model-level SHAP explanations from rule-based pedagogical suggestions, and reminding users that SHAP shows statistical association rather than causal proof."*

---

### [3:05–3:55] What-If Simulation, Teacher Intervention Center & Report Exports
**Visual:** Open **World 4 -> What-If Simulator** and **Intervention Center** tab (updating a student's status and saving a teacher note), then switch to **World 6 (`/ocean` -> Reports & CSV Exports)**.  
**Voiceover:**
> *"Teachers can explore predictive sensitivity in the **What-If Simulator**, queue priority students directly into the **Intervention Center** with persisted follow-up status and classroom notes, and export period-level summaries or filtered student CSV reports straight from the **Oceanic Crystal Realm**."*

---

### [3:55–4:45] FastAPI Swagger, Automated Tests & Ethical Summary
**Visual:** Show FastAPI `/docs` and run `pytest -v tests/` in terminal (`reports/figures/11_calibration_curve.png` briefly visible).  
**Voiceover:**
> *"Every prediction and report endpoint enforces strict Pydantic validation—rejecting invalid periods, booleans, or non-finite values—and our automated `pytest` suite verifies data invariants, feature-engineering parity, calibration bounds, SHAP fallback safety, batch inference, and intervention/report endpoints.*
>
> *Thank you for exploring **EduGuard AI**—Detect Early, Understand Deeply, Support Intelligently."*
