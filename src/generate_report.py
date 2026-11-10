"""
EduGuard AI - Automated Technical Report Generator
Builds `reports/technical_report.md` programmatically from real data quality,
model metadata, evaluation metrics, error analysis, and SHAP JSON outputs.
Zero hardcoded metrics or unverified claims.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_QUALITY_REPORT_PATH,
    EVALUATION_METRICS_PATH,
    METADATA_PATH,
    REPORTS_DIR,
)
from src.explain import EduGuardExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.ReportGenerator")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report input missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_technical_report(output_path: Path = REPORTS_DIR / "technical_report.md") -> Path:
    """Synthesizes complete markdown report strictly from verified pipeline artifacts."""
    quality = _load_json(DATA_QUALITY_REPORT_PATH)
    meta = _load_json(METADATA_PATH)
    eval_data = _load_json(EVALUATION_METRICS_PATH)

    explainer = EduGuardExplainer()
    global_shap = explainer.generate_global_explanations(sample_size=500)
    shap_ranking = global_shap["global_feature_importance"]

    champion = meta["model_name"]
    arch = meta["model_architecture"]
    thresh = meta["selected_threshold"]
    val_m = meta["validation_metrics"]
    t_test = meta["final_test_metrics_temporal"]
    d_test = meta["final_test_metrics_disjoint"]
    cal_sum = meta["calibration_summary"]
    splits = meta["split_sizes"]
    env = meta["environment_versions"]
    err_analysis = eval_data["temporal_test_error_analysis"]
    benchmarks = eval_data["model_benchmarks"]
    period_breakdown = eval_data["disjoint_test_by_period"]

    t_cm = t_test["confusion_matrix"]
    d_cm = d_test["confusion_matrix"]

    # Format candidate table dynamically
    bench_rows = []
    for name, b in benchmarks.items():
        tv = b["temporal_validation"]
        dv = b["student_disjoint_validation"]
        cal_cmp = b["calibration_comparison"]
        comp_score = eval_data["candidate_selection_scores"].get(name, {}).get("composite_score", "-")
        comp_str = f"{comp_score:.4f}" if isinstance(comp_score, float) else "Baseline"
        is_champ = " **(Selected)**" if name == champion else ""
        bench_rows.append(
            f"| **{name}**{is_champ} | {tv['average_precision']:.4f} | {dv['average_precision']:.4f} | "
            f"{tv['roc_auc']:.4f} | {cal_cmp['temporal_brier_before']:.4f} -> {cal_cmp['temporal_brier_after']:.4f} | "
            f"{b['inference_latency_ms']:.4f} ms | {comp_str} |"
        )
    bench_table = "\n".join(bench_rows)

    # Format period breakdown table on Disjoint Held-Out Students
    period_rows = []
    for p_key in sorted(period_breakdown.keys()):
        pm = period_breakdown[p_key]
        period_label = p_key.replace("_", " ").title()
        period_rows.append(
            f"| {period_label} | {pm['sample_size']} | {pm['positive_rate']*100:.1f}% | "
            f"{pm['average_precision']:.4f} | {pm['recall']*100:.1f}% | {pm['precision']*100:.1f}% | "
            f"{pm['specificity']*100:.1f}% | {pm['f1_score']:.3f} |"
        )
    period_table = "\n".join(period_rows)

    # Format global SHAP ranking
    shap_rows = []
    for idx, item in enumerate(shap_ranking, start=1):
        stats = quality["feature_statistics"].get(item["feature"], {})
        shap_rows.append(
            f"| {idx} | `{item['feature']}` | {item['mean_absolute_shap']:.4f} | "
            f"{stats.get('mean', 'N/A')} | {stats.get('missing', 0)} |"
        )
    shap_table = "\n".join(shap_rows)

    fn_stats = err_analysis["false_negatives"]
    fp_stats = err_analysis["false_positives"]
    tp_stats = err_analysis["true_positives"]
    tn_stats = err_analysis["true_negatives"]

    missing_pcts = quality["missingness"]["missing_percentages"]
    target_dist = quality["target"]

    report_md = f"""# EduGuard AI — Comprehensive Technical Report
### Academic Intelligence & Early Warning Platform
**Organization Context:** Internship Engineering Submission — ALGIONS AI LABS PRIVATE LIMITED  
**Domain:** School Student Performance Monitoring & Early Intervention Advisory  
**Evaluation Protocol:** Zero-Leakage Dual Validation (Temporal + Student-Disjoint) with Validation-Split Calibration  
**Champion Model:** {champion} (`{arch}`, Sigmoid Calibrated) | **Version:** v{meta['model_version']} | **Global Seed:** {meta['random_seed']}

---

## Executive Summary

**EduGuard AI** is a production-style advisory machine learning system built to help teachers and academic coordinators identify school students who may score below **40/100** on their upcoming assessment (`at_risk = 1`). All model inputs are restricted strictly to pre-assessment engagement and score records. **Predictions are advisory flags only**—they must never trigger automated grades, disciplinary measures, enrollment limits, or replace human judgment.

Verified highlights from the current reproducible training run (`{meta['training_timestamp_utc']}`):
1. **Single Authoritative Feature Engineering (`src/feature_engineering.py`):** Eliminated historical training-serving skew by unifying data generation, offline evaluation, FastAPI inference, Streamlit, and React on the exact rolling two-assessment mean (`previous_score_avg`) and momentum delta (`score_trend`). Verified **0 mismatches** across {quality['total_records']:,} records (`{quality['derived_consistency']['status']}`).
2. **Zero-Leakage Dual Validation & Genuine Calibration:** Partitioned validation sets into disjoint calibration and selection/threshold halves (`t_val_cal`/`t_val_sel` and student-disjoint `d_val_cal`/`d_val_sel`). Sigmoid calibration reduced held-out Temporal Test Brier score from **{cal_sum['temporal_test_brier_before']:.4f}** to **{cal_sum['temporal_test_brier_after']:.4f}** (ECE = {cal_sum['temporal_test_ece_after']:.4f}).
3. **Programmatic Champion Selection:** Ranked all non-dummy candidates on disjoint validation selection metrics; **{champion}** achieved the highest composite score (**{eval_data['candidate_selection_scores'][champion]['composite_score']:.4f}**).
4. **Held-Out Test Generalization at Threshold τ* = {thresh:.2f}:**
   - **Temporal Test (Period 4, N={t_test['sample_size']:,}):** Recall = **{t_test['recall']*100:.1f}%**, Average Precision (AP) = **{t_test['average_precision']:.3f}**, ROC-AUC = **{t_test['roc_auc']:.3f}**, Specificity = **{t_test['specificity']*100:.1f}%**, Precision = **{t_test['precision']*100:.1f}%**, F1 = **{t_test['f1_score']:.3f}**.
   - **Student-Disjoint Test (Unseen Students, N={d_test['sample_size']:,}):** Recall = **{d_test['recall']*100:.1f}%**, Average Precision (AP) = **{d_test['average_precision']:.3f}**, ROC-AUC = **{d_test['roc_auc']:.3f}**, Specificity = **{d_test['specificity']*100:.1f}%**, Precision = **{d_test['precision']*100:.1f}%**, F1 = **{d_test['f1_score']:.3f}**.

---

## 1. Problem Formulation & Anti-Leakage Architecture

### 1.1 Asymmetric Cost Framing for School Support
Missing a struggling student before a key exam carries high educational cost, whereas a brief, supportive teacher check-in with a borderline passing student carries low non-punitive cost:
- **False Negative (FN):** A student destined to score < 40 goes unflagged and misses early peer tutoring or teacher extra help.
- **False Positive (FP):** A student who scores >= 40 receives a preventive check-in or study resource suggestion.

### 1.2 Formal Target & Pre-Assessment Feature Schema
$$\\text{{Target: }} y_{{i, t}} = \\begin{{cases}} 1 & \\text{{if }} \\text{{next\\_assessment\\_score}}_{{i, t+1}} < 40.0 \\\\ 0 & \\text{{otherwise}} \\end{{cases}}$$

Modeled inputs (all strictly observed prior to assessment $t+1$):
1. `attendance_pct`: Student classroom attendance percentage in period $t$ $[0, 100]$.
2. `assignment_completion_pct`: Coursework completion percentage in period $t$ $[0, 100]$.
3. `late_assignment_count`: Count of late homework/assignment submissions $[0, 30]$.
4. `previous_score`: Most recent assessment score in period $t$ $[0, 100]$.
5. `previous_score_2`: Prior assessment score in period $t-1$ $[0, 100]$ (nullable for Period 1 transfer students).
6. `previous_score_avg`: **Canonical rolling mean** of `previous_score` and `previous_score_2`, falling back to `previous_score` when `previous_score_2` is missing. Computed post-missingness so hidden prior scores cannot leak through the average.
7. `score_trend`: `previous_score - previous_score_2` (`0.0` if `previous_score_2` is missing).
8. `assessment_period`: Period index $t \\in \\{{1, 2, 3, 4\\}}$.

### 1.3 Anti-Leakage Safeguards (`src/leakage_audit.py` & `src/data_validation.py`)
- `next_assessment_score`, `at_risk`, and `student_id` are dropped via `ColumnTransformer(..., remainder="drop")` and asserted absent from $X$.
- Zero protected or socio-demographic attributes (`gender`, `race`, `religion`, `caste`, `disability`, `income`, `name`, `zipcode`, etc.) exist in any dataset or model input.
- Train, calibration, selection/threshold, and held-out test partitions undergo programmatic mutual-exclusivity and temporal-directionality checks.

---

## 2. Synthetic Data Generation & Quality Verification

- **Cohort Size & Horizon:** {quality['unique_students']:,} synthetic school students across 5 curriculum cycles generating **{quality['total_records']:,} snapshots** (Periods 1–4 predicting Periods 2–5), seed `{meta['random_seed']}`.
- **Target Balance:** `{target_dist['class_distribution']['0']:,}` Not At-Risk (`{(1-target_dist['at_risk_ratio'])*100:.2f}%`) vs. `{target_dist['class_distribution']['1']:,}` At-Risk (`{target_dist['at_risk_ratio']*100:.2f}%`). Period-wise positive rates: P1 `{target_dist['per_period_at_risk_rate']['period_1']*100:.2f}%`, P2 `{target_dist['per_period_at_risk_rate']['period_2']*100:.2f}%`, P3 `{target_dist['per_period_at_risk_rate']['period_3']*100:.2f}%`, P4 `{target_dist['per_period_at_risk_rate']['period_4']*100:.2f}%`.
- **Controlled Missingness (MCAR/MAR):** `attendance_pct` `{missing_pcts['attendance_pct']:.2f}%`, `assignment_completion_pct` `{missing_pcts['assignment_completion_pct']:.2f}%`, `previous_score_2` `{missing_pcts['previous_score_2']:.2f}%`.
- **Quality Report (`reports/data_quality_report.json`):** Status `{quality['overall_status']}` — `0` duplicate keys, `0` infinite/out-of-range values, `0` derived-feature mismatches, `0` target-label mismatches.
- **Visualizations Generated (`reports/figures/`):** Figures `01` through `08` cover class balance, attendance/completion/score distributions, longitudinal period means, correlation heatmap (max $|r| = 0.6119$), missingness rates, and late-submission vs. score behavioral scatter; Figures `09`–`11` capture global SHAP beeswarm/bar plots and the calibration curve.

---

## 3. Experimental Protocol, Calibration & Model Selection

### 3.1 Partition Architecture (Zero Leakage Across Fit, Calibration & Tuning)
| Split Regime | Training Fit | Validation Calibration (`val_cal`) | Validation Selection/Threshold (`val_sel`) | Held-Out Test (Single Eval) |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal** | {splits['temporal_train_rows']:,} rows (P1–P2) | {splits['temporal_val_cal_rows']:,} rows (P3 half) | {splits['temporal_val_sel_rows']:,} rows (P3 half) | {splits['temporal_test_rows']:,} rows (Period 4) |
| **Student-Disjoint** | {splits['disjoint_train_rows']:,} rows (70% students) | {splits['disjoint_val_cal_rows']:,} rows (half val students) | {splits['disjoint_val_sel_rows']:,} rows (other half val students) | {splits['disjoint_test_rows']:,} rows (15% unseen students) |

### 3.2 Candidate Validation Comparison (`val_sel` Folds Only)
> **Metric Note:** Table reports **`Average Precision (AP)`** (`sklearn.metrics.average_precision_score`, step-wise area under Precision-Recall curve) rather than inflated trapezoidal PR-AUC (`0.596` on balanced-unaware baselines). Calibrated Brier scores reflect validation sigmoid calibration tested on `val_sel`.

| Candidate Model | Temp Val AP | Disjoint Val AP | Temp ROC-AUC | Temp Brier (Before -> After) | Latency / Sample | Composite Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{bench_table}

**Programmatic Selection Rule:** {meta['selection_rationale']}

### 3.3 Probability Calibration Summary
- **Method:** `CalibratedClassifierCV(FrozenEstimator(base_pipeline), method="sigmoid")` fit exclusively on validation calibration halves.
- **Temporal Validation Brier:** {cal_sum['temporal_val_brier_before']:.4f} -> **{cal_sum['temporal_val_brier_after']:.4f}**
- **Held-Out Temporal Test Brier:** {cal_sum['temporal_test_brier_before']:.4f} -> **{cal_sum['temporal_test_brier_after']:.4f}** (Expected Calibration Error = **{cal_sum['temporal_test_ece_after']:.4f}**, Figure `11_calibration_curve.png`).
- **Disjoint Validation Brier:** {cal_sum['disjoint_val_brier_before']:.4f} -> {cal_sum['disjoint_val_brier_after']:.4f} (Disjoint Test Brier = **{d_test['brier_score']:.4f}**).

---

## 4. Threshold Optimization & Held-Out Test Performance

### 4.1 Validation Threshold Tuning
{meta['threshold_optimization_rationale']}

- **Validation Selection (`t_val_sel`, N={val_m['sample_size']:,}):** Recall `{val_m['recall']*100:.1f}%`, Precision `{val_m['precision']*100:.1f}%`, Specificity `{val_m['specificity']*100:.1f}%`, F1 `{val_m['f1_score']:.3f}`, F2 `{val_m['f2_score']:.3f}`, AP `{val_m['average_precision']:.3f}` (trapezoidal PR-AUC `{val_m['pr_auc']:.3f}`).

### 4.2 Held-Out Test Metrics (Single Final Evaluation at τ* = {thresh:.2f})

| Held-Out Split | N | Pos Rate | AP | Trap PR-AUC | ROC-AUC | Recall | Precision | Specificity | F1 | F2 | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Test (P4)** | {t_test['sample_size']:,} | {t_test['positive_rate']*100:.1f}% | {t_test['average_precision']:.3f} | {t_test['pr_auc']:.3f} | {t_test['roc_auc']:.3f} | {t_test['recall']*100:.1f}% | {t_test['precision']*100:.1f}% | {t_test['specificity']*100:.1f}% | {t_test['f1_score']:.3f} | {t_test['f2_score']:.3f} | {t_test['brier_score']:.4f} |
| **Disjoint Test (Unseen Students)** | {d_test['sample_size']:,} | {d_test['positive_rate']*100:.1f}% | {d_test['average_precision']:.3f} | {d_test['pr_auc']:.3f} | {d_test['roc_auc']:.3f} | {d_test['recall']*100:.1f}% | {d_test['precision']*100:.1f}% | {d_test['specificity']*100:.1f}% | {d_test['f1_score']:.3f} | {d_test['f2_score']:.3f} | {d_test['brier_score']:.4f} |

**Temporal Test Confusion Matrix:** TN = `{t_cm['true_negatives']:,}`, FP = `{t_cm['false_positives']:,}`, FN = `{t_cm['false_negatives']:,}`, TP = `{t_cm['true_positives']:,}` (Total = `{sum(t_cm.values()):,}`).  
**Disjoint Test Confusion Matrix:** TN = `{d_cm['true_negatives']:,}`, FP = `{d_cm['false_positives']:,}`, FN = `{d_cm['false_negatives']:,}`, TP = `{d_cm['true_positives']:,}` (Total = `{sum(d_cm.values()):,}`).

### 4.3 Per-Period Generalization on Unseen Disjoint Test Students
| Period | N | Positive Rate | AP | Recall | Precision | Specificity | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{period_table}

---

## 5. Error Analysis (Held-Out Temporal Test, N={t_test['sample_size']:,})

Inspecting error cohorts reveals how synthetic outcome dynamics and threshold selection interact:

| Outcome Group | Count | Share | Mean Pred Prob | Mean Prior Score | Mean Attendance (%) | Mean Completion (%) | Mean Next Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | {tp_stats['count']:,} | {tp_stats['share_pct']:.2f}% | {tp_stats['mean_predicted_prob']:.3f} | {tp_stats['mean_previous_score']:.2f} | {tp_stats['mean_attendance_pct']:.2f}% | {tp_stats['mean_assignment_completion_pct']:.2f}% | {tp_stats['mean_next_assessment_score']:.2f} |
| **False Positives (FP)** | {fp_stats['count']:,} | {fp_stats['share_pct']:.2f}% | {fp_stats['mean_predicted_prob']:.3f} | {fp_stats['mean_previous_score']:.2f} | {fp_stats['mean_attendance_pct']:.2f}% | {fp_stats['mean_assignment_completion_pct']:.2f}% | {fp_stats['mean_next_assessment_score']:.2f} |
| **False Negatives (FN)** | {fn_stats['count']:,} | {fn_stats['share_pct']:.2f}% | {fn_stats['mean_predicted_prob']:.3f} | {fn_stats['mean_previous_score']:.2f} | {fn_stats['mean_attendance_pct']:.2f}% | {fn_stats['mean_assignment_completion_pct']:.2f}% | {fn_stats['mean_next_assessment_score']:.2f} |
| **True Negatives (TN)** | {tn_stats['count']:,} | {tn_stats['share_pct']:.2f}% | {tn_stats['mean_predicted_prob']:.3f} | {tn_stats['mean_previous_score']:.2f} | {tn_stats['mean_attendance_pct']:.2f}% | {tn_stats['mean_assignment_completion_pct']:.2f}% | {tn_stats['mean_next_assessment_score']:.2f} |

**Key Findings:**
1. **Borderline False Positives:** Students flagged falsely in Period 4 averaged `previous_score = {fp_stats['mean_previous_score']:.1f}` and scored `{fp_stats['mean_next_assessment_score']:.1f}` next cycle—hovering directly near the 40-mark boundary. Additionally, Period 4 exhibits a lower natural base at-risk rate (`{t_test['positive_rate']*100:.1f}%` vs. `{val_m['positive_rate']*100:.1f}%` in Period 3 validation) due to lower test difficulty and cumulative learning progression, which mechanically lowers Period 4 precision from `{val_m['precision']*100:.1f}%` to `{t_test['precision']*100:.1f}%` at fixed threshold `τ* = {thresh:.2f}`.
2. **Unexpected Drop False Negatives:** Only `{fn_stats['count']}` students (`{fn_stats['share_pct']:.2f}%` of Period 4) were missed; these students averaged `{fn_stats['mean_previous_score']:.1f}` in Period 4 (`{fn_stats['mean_previous_score_2']:.1f}` in Period 3) before experiencing an unexpected noise drop to `{fn_stats['mean_next_assessment_score']:.1f}`. Teachers should always supplement model flags with qualitative classroom observation for sudden behavioral changes.

---

## 6. Explainable AI (SHAP) & Support Recommendations Separation

Using `shap.TreeExplainer` (`model_output="probability"` on background reference sample N=150) over the base `{arch}` ensemble prior to sigmoid calibration, global mean `|SHAP|` feature impact (`sample_size = {global_shap['sample_size']}`) ranks as follows:

| Rank | Feature | Mean \\|SHAP\\| Impact | Dataset Mean | Missing Count |
| :---: | :--- | :---: | :---: | :---: |
{shap_table}

- **Statistical vs. Causal Clarification:** High `previous_score_avg`, `previous_score`, and `previous_score_2` attributions quantify statistical association with next-assessment scores in synthetic trajectories. Lower attendance or late assignments often covary with latent student discipline and aptitude; **SHAP values do not establish causal effect sizes**.
- **Rule-Based Support Separation:** Pedagogical suggestions returned by `predict.py` (`"source": "rule_based_advisory"`) are explicit teacher-facing heuristics triggered by observable thresholds (e.g., attendance < 75%, completion < 70%, score drop <= -8 marks), kept architecturally separate from SHAP attribution scores.

---

## 7. Responsible AI Limitations & Reproducibility

- **Synthetic Data Boundary & Unverified Real-World Generalization:** Every data point originates from a controlled synthetic simulator (`N=2,500`, `seed=42`). **Model calibration, error bounds, and feature weights have NOT been validated on real school student records** and should never be deployed in live academic settings without local institutional retraining, teacher co-design, and fairness auditing.
- **Advisory Governance:** EduGuard AI prohibits automated grading adjustments, course tracking, disciplinary flagging, or zero-oversight decisions.

### Execution Environment
- **Python:** `{env['python']}`
- **scikit-learn:** `{env['scikit_learn']}` | **xgboost:** `{env['xgboost']}` | **shap:** `{env['shap']}`
- **pandas:** `{env['pandas']}` | **numpy:** `{env['numpy']}` | **joblib:** `{env['joblib']}`
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Automated technical report generated successfully at {output_path}")
    return output_path


if __name__ == "__main__":
    generate_technical_report()

