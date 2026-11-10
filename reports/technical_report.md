# EduGuard AI — Comprehensive Technical Report
### Academic Intelligence & Early Warning Platform
**Organization Context:** Internship Engineering Submission — ALGIONS AI LABS PRIVATE LIMITED  
**Domain:** School Student Performance Monitoring & Early Intervention Advisory  
**Evaluation Protocol:** Zero-Leakage Dual Validation (Temporal + Student-Disjoint) with Validation-Split Calibration  
**Champion Model:** XGBoost (`XGBClassifier`, Sigmoid Calibrated) | **Version:** v1.1.0 | **Global Seed:** 42

---

## Executive Summary

**EduGuard AI** is a production-style advisory machine learning system built to help teachers and academic coordinators identify school students who may score below **40/100** on their upcoming assessment (`at_risk = 1`). All model inputs are restricted strictly to pre-assessment engagement and score records. **Predictions are advisory flags only**—they must never trigger automated grades, disciplinary measures, enrollment limits, or replace human judgment.

Verified highlights from the current reproducible training run (`2026-11-10T13:35:32.183544+00:00`):
1. **Single Authoritative Feature Engineering (`src/feature_engineering.py`):** Eliminated historical training-serving skew by unifying data generation, offline evaluation, FastAPI inference, Streamlit, and React on the exact rolling two-assessment mean (`previous_score_avg`) and momentum delta (`score_trend`). Verified **0 mismatches** across 10,000 records (`PASS`).
2. **Zero-Leakage Dual Validation & Genuine Calibration:** Partitioned validation sets into disjoint calibration and selection/threshold halves (`t_val_cal`/`t_val_sel` and student-disjoint `d_val_cal`/`d_val_sel`). Sigmoid calibration reduced held-out Temporal Test Brier score from **0.0959** to **0.0677** (ECE = 0.0427).
3. **Programmatic Champion Selection:** Ranked all non-dummy candidates on disjoint validation selection metrics; **XGBoost** achieved the highest composite score (**0.8818**).
4. **Held-Out Test Generalization at Threshold τ* = 0.18:**
   - **Temporal Test (Period 4, N=2,500):** Recall = **89.8%**, Average Precision (AP) = **0.806**, ROC-AUC = **0.942**, Specificity = **84.6%**, Precision = **52.2%**, F1 = **0.660**.
   - **Student-Disjoint Test (Unseen Students, N=1,500):** Recall = **82.9%**, Average Precision (AP) = **0.838**, ROC-AUC = **0.945**, Specificity = **90.0%**, Precision = **68.9%**, F1 = **0.752**.

---

## 1. Problem Formulation & Anti-Leakage Architecture

### 1.1 Asymmetric Cost Framing for School Support
Missing a struggling student before a key exam carries high educational cost, whereas a brief, supportive teacher check-in with a borderline passing student carries low non-punitive cost:
- **False Negative (FN):** A student destined to score < 40 goes unflagged and misses early peer tutoring or teacher extra help.
- **False Positive (FP):** A student who scores >= 40 receives a preventive check-in or study resource suggestion.

### 1.2 Formal Target & Pre-Assessment Feature Schema
$$\text{Target: } y_{i, t} = \begin{cases} 1 & \text{if } \text{next\_assessment\_score}_{i, t+1} < 40.0 \\ 0 & \text{otherwise} \end{cases}$$

Modeled inputs (all strictly observed prior to assessment $t+1$):
1. `attendance_pct`: Student classroom attendance percentage in period $t$ $[0, 100]$.
2. `assignment_completion_pct`: Coursework completion percentage in period $t$ $[0, 100]$.
3. `late_assignment_count`: Count of late homework/assignment submissions $[0, 30]$.
4. `previous_score`: Most recent assessment score in period $t$ $[0, 100]$.
5. `previous_score_2`: Prior assessment score in period $t-1$ $[0, 100]$ (nullable for Period 1 transfer students).
6. `previous_score_avg`: **Canonical rolling mean** of `previous_score` and `previous_score_2`, falling back to `previous_score` when `previous_score_2` is missing. Computed post-missingness so hidden prior scores cannot leak through the average.
7. `score_trend`: `previous_score - previous_score_2` (`0.0` if `previous_score_2` is missing).
8. `assessment_period`: Period index $t \in \{1, 2, 3, 4\}$.

### 1.3 Anti-Leakage Safeguards (`src/leakage_audit.py` & `src/data_validation.py`)
- `next_assessment_score`, `at_risk`, and `student_id` are dropped via `ColumnTransformer(..., remainder="drop")` and asserted absent from $X$.
- Zero protected or socio-demographic attributes (`gender`, `race`, `religion`, `caste`, `disability`, `income`, `name`, `zipcode`, etc.) exist in any dataset or model input.
- Train, calibration, selection/threshold, and held-out test partitions undergo programmatic mutual-exclusivity and temporal-directionality checks.

---

## 2. Synthetic Data Generation & Quality Verification

- **Cohort Size & Horizon:** 2,500 synthetic school students across 5 curriculum cycles generating **10,000 snapshots** (Periods 1–4 predicting Periods 2–5), seed `42`.
- **Target Balance:** `8,046` Not At-Risk (`80.46%`) vs. `1,954` At-Risk (`19.54%`). Period-wise positive rates: P1 `18.92%`, P2 `24.32%`, P3 `19.16%`, P4 `15.76%`.
- **Controlled Missingness (MCAR/MAR):** `attendance_pct` `1.50%`, `assignment_completion_pct` `2.01%`, `previous_score_2` `0.35%`.
- **Quality Report (`reports/data_quality_report.json`):** Status `PASS` — `0` duplicate keys, `0` infinite/out-of-range values, `0` derived-feature mismatches, `0` target-label mismatches.
- **Visualizations Generated (`reports/figures/`):** Figures `01` through `08` cover class balance, attendance/completion/score distributions, longitudinal period means, correlation heatmap (max $|r| = 0.6119$), missingness rates, and late-submission vs. score behavioral scatter; Figures `09`–`11` capture global SHAP beeswarm/bar plots and the calibration curve.

---

## 3. Experimental Protocol, Calibration & Model Selection

### 3.1 Partition Architecture (Zero Leakage Across Fit, Calibration & Tuning)
| Split Regime | Training Fit | Validation Calibration (`val_cal`) | Validation Selection/Threshold (`val_sel`) | Held-Out Test (Single Eval) |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal** | 5,000 rows (P1–P2) | 1,250 rows (P3 half) | 1,250 rows (P3 half) | 2,500 rows (Period 4) |
| **Student-Disjoint** | 7,000 rows (70% students) | 748 rows (half val students) | 752 rows (other half val students) | 1,500 rows (15% unseen students) |

### 3.2 Candidate Validation Comparison (`val_sel` Folds Only)
> **Metric Note:** Table reports **`Average Precision (AP)`** (`sklearn.metrics.average_precision_score`, step-wise area under Precision-Recall curve) rather than inflated trapezoidal PR-AUC (`0.596` on balanced-unaware baselines). Calibrated Brier scores reflect validation sigmoid calibration tested on `val_sel`.

| Candidate Model | Temp Val AP | Disjoint Val AP | Temp ROC-AUC | Temp Brier (Before -> After) | Latency / Sample | Composite Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | 0.1912 | 0.2154 | 0.5000 | 0.1553 -> 0.1553 | 0.0040 ms | Baseline |
| **LogisticRegression** | 0.8420 | 0.8459 | 0.9501 | 0.2029 -> 0.0653 | 0.0046 ms | 0.8764 |
| **RandomForestClassifier** | 0.8459 | 0.8489 | 0.9528 | 0.1267 -> 0.0742 | 0.1078 ms | 0.8765 |
| **XGBoost** **(Selected)** | 0.8515 | 0.8599 | 0.9538 | 0.0884 -> 0.0676 | 0.0090 ms | 0.8818 |

**Programmatic Selection Rule:** XGBoost was selected programmatically as champion via composite validation score (0.8818 = 0.35*MeanAP + 0.25*MeanROCAUC + 0.25*ValOptF2 + 0.15*(1-MeanBrier)) over non-dummy candidates evaluated exclusively on validation selection folds (Temp AP=0.852, Disjoint AP=0.860, Mean Brier=0.0701).

### 3.3 Probability Calibration Summary
- **Method:** `CalibratedClassifierCV(FrozenEstimator(base_pipeline), method="sigmoid")` fit exclusively on validation calibration halves.
- **Temporal Validation Brier:** 0.0884 -> **0.0676**
- **Held-Out Temporal Test Brier:** 0.0959 -> **0.0677** (Expected Calibration Error = **0.0427**, Figure `11_calibration_curve.png`).
- **Disjoint Validation Brier:** 0.0704 -> 0.0727 (Disjoint Test Brier = **0.0723**).

---

## 4. Threshold Optimization & Held-Out Test Performance

### 4.1 Validation Threshold Tuning
Threshold 0.18 was selected strictly on the validation selection fold (N=1250) to maximize F2 safety-net score (0.820) subject to precision >= 0.55. It achieved 90.8% recall, 59.1% precision, and 85.2% specificity.

- **Validation Selection (`t_val_sel`, N=1,250):** Recall `90.8%`, Precision `59.1%`, Specificity `85.2%`, F1 `0.716`, F2 `0.820`, AP `0.852` (trapezoidal PR-AUC `0.851`).

### 4.2 Held-Out Test Metrics (Single Final Evaluation at τ* = 0.18)

| Held-Out Split | N | Pos Rate | AP | Trap PR-AUC | ROC-AUC | Recall | Precision | Specificity | F1 | F2 | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Test (P4)** | 2,500 | 15.8% | 0.806 | 0.806 | 0.942 | 89.8% | 52.2% | 84.6% | 0.660 | 0.785 | 0.0677 |
| **Disjoint Test (Unseen Students)** | 1,500 | 21.0% | 0.838 | 0.838 | 0.945 | 82.9% | 68.9% | 90.0% | 0.752 | 0.796 | 0.0723 |

**Temporal Test Confusion Matrix:** TN = `1,782`, FP = `324`, FN = `40`, TP = `354` (Total = `2,500`).  
**Disjoint Test Confusion Matrix:** TN = `1,067`, FP = `118`, FN = `54`, TP = `261` (Total = `1,500`).

### 4.3 Per-Period Generalization on Unseen Disjoint Test Students
| Period | N | Positive Rate | AP | Recall | Precision | Specificity | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Period 1 | 375 | 19.2% | 0.8429 | 93.1% | 63.8% | 87.5% | 0.757 |
| Period 2 | 375 | 27.2% | 0.8866 | 83.3% | 75.9% | 90.1% | 0.794 |
| Period 3 | 375 | 20.5% | 0.8190 | 81.8% | 68.5% | 90.3% | 0.746 |
| Period 4 | 375 | 17.1% | 0.8095 | 71.9% | 65.7% | 92.3% | 0.687 |

---

## 5. Error Analysis (Held-Out Temporal Test, N=2,500)

Inspecting error cohorts reveals how synthetic outcome dynamics and threshold selection interact:

| Outcome Group | Count | Share | Mean Pred Prob | Mean Prior Score | Mean Attendance (%) | Mean Completion (%) | Mean Next Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 354 | 14.16% | 0.735 | 30.55 | 54.64% | 54.64% | 30.48 |
| **False Positives (FP)** | 324 | 12.96% | 0.513 | 39.87 | 55.20% | 55.76% | 46.34 |
| **False Negatives (FN)** | 40 | 1.60% | 0.060 | 51.72 | 53.84% | 53.40% | 36.04 |
| **True Negatives (TN)** | 1,782 | 71.28% | 0.028 | 62.04 | 59.85% | 59.82% | 63.66 |

**Key Findings:**
1. **Borderline False Positives:** Students flagged falsely in Period 4 averaged `previous_score = 39.9` and scored `46.3` next cycle—hovering directly near the 40-mark boundary. Additionally, Period 4 exhibits a lower natural base at-risk rate (`15.8%` vs. `19.1%` in Period 3 validation) due to lower test difficulty and cumulative learning progression, which mechanically lowers Period 4 precision from `59.1%` to `52.2%` at fixed threshold `τ* = 0.18`.
2. **Unexpected Drop False Negatives:** Only `40` students (`1.60%` of Period 4) were missed; these students averaged `51.7` in Period 4 (`49.7` in Period 3) before experiencing an unexpected noise drop to `36.0`. Teachers should always supplement model flags with qualitative classroom observation for sudden behavioral changes.

---

## 6. Explainable AI (SHAP) & Support Recommendations Separation

Using `shap.TreeExplainer` (`model_output="probability"` on background reference sample N=150) over the base `XGBClassifier` ensemble prior to sigmoid calibration, global mean `|SHAP|` feature impact (`sample_size = 500`) ranks as follows:

| Rank | Feature | Mean \|SHAP\| Impact | Dataset Mean | Missing Count |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `previous_score_avg` | 0.1284 | 54.267 | 0 |
| 2 | `previous_score` | 0.0889 | 54.202 | 0 |
| 3 | `previous_score_2` | 0.0589 | 54.316 | 35 |
| 4 | `assessment_period` | 0.0130 | 2.5 | 0 |
| 5 | `score_trend` | 0.0086 | -0.131 | 0 |
| 6 | `assignment_completion_pct` | 0.0070 | 58.486 | 201 |
| 7 | `late_assignment_count` | 0.0033 | 2.08 | 0 |
| 8 | `attendance_pct` | 0.0032 | 58.569 | 150 |

- **Statistical vs. Causal Clarification:** High `previous_score_avg`, `previous_score`, and `previous_score_2` attributions quantify statistical association with next-assessment scores in synthetic trajectories. Lower attendance or late assignments often covary with latent student discipline and aptitude; **SHAP values do not establish causal effect sizes**.
- **Rule-Based Support Separation:** Pedagogical suggestions returned by `predict.py` (`"source": "rule_based_advisory"`) are explicit teacher-facing heuristics triggered by observable thresholds (e.g., attendance < 75%, completion < 70%, score drop <= -8 marks), kept architecturally separate from SHAP attribution scores.

---

## 7. Responsible AI Limitations & Reproducibility

- **Synthetic Data Boundary & Unverified Real-World Generalization:** Every data point originates from a controlled synthetic simulator (`N=2,500`, `seed=42`). **Model calibration, error bounds, and feature weights have NOT been validated on real school student records** and should never be deployed in live academic settings without local institutional retraining, teacher co-design, and fairness auditing.
- **Advisory Governance:** EduGuard AI prohibits automated grading adjustments, course tracking, disciplinary flagging, or zero-oversight decisions.

### Execution Environment
- **Python:** `3.11.5`
- **scikit-learn:** `1.7.2` | **xgboost:** `3.2.0` | **shap:** `0.51.0`
- **pandas:** `2.2.3` | **numpy:** `2.2.1` | **joblib:** `1.5.2`
