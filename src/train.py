"""
EduGuard AI - Model Training, Calibration, Selection & Experiment Benchmarking Pipeline
Trains, calibrates, selects, benchmarks, and persists:
1. DummyClassifier (Prior Baseline)
2. LogisticRegression (Scaled Linear Benchmark)
3. RandomForestClassifier (Nonlinear Bagged Ensemble)
4. XGBoost (Regularized Gradient Boosted Decision Trees)

Implements strict zero-leakage Dual Validation (Temporal + Student-Disjoint) with
disjoint calibration/selection validation folds, programmatic composite champion selection,
F2 safety-net threshold optimization, held-out test error analysis, and calibration plotting.
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib cache safety
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_METHOD,
    DATA_QUALITY_REPORT_PATH,
    DISCLAIMER_TEXT,
    EVALUATION_METRICS_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    ID_COLUMN,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_VERSION,
    NUM_PERIODS,
    NUM_STUDENTS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    RISK_CATEGORIES,
    TARGET_COLUMN,
)
from src.data_validation import StudentDataValidator
from src.evaluate import (
    compute_calibration_curve_stats,
    compute_comprehensive_metrics,
    optimize_classification_threshold,
    split_student_disjoint,
    split_temporal,
    split_validation_for_calibration,
)
from src.leakage_audit import LeakageAuditor
from src.preprocessing import build_full_pipeline, extract_features_and_target, unwrap_base_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.Train")


def initialize_candidate_models(random_seed: int = RANDOM_SEED) -> Dict[str, Tuple[Any, bool]]:
    """
    Initializes candidate classifiers and indicates whether feature scaling is applied.
    Returns: Dict[name, (estimator_instance, scale_features_flag)]
    """
    return {
        "DummyClassifier": (
            DummyClassifier(strategy="prior"),
            False,
        ),
        "LogisticRegression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=random_seed,
            ),
            True,
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                scale_pos_weight=1.0,
                random_state=random_seed,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            False,
        ),
    }


def _compute_error_analysis(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes genuine error-cohort breakdown across TP, FP, FN, and TN groups."""
    y_pred = (y_prob >= threshold).astype(int)
    analysis_df = df_test.copy()
    analysis_df["_prob"] = y_prob

    buckets = {
        "true_positives": (y_true == 1) & (y_pred == 1),
        "false_positives": (y_true == 0) & (y_pred == 1),
        "false_negatives": (y_true == 1) & (y_pred == 0),
        "true_negatives": (y_true == 0) & (y_pred == 0),
    }

    summary: Dict[str, Any] = {}
    summary_cols = [
        "attendance_pct",
        "assignment_completion_pct",
        "late_assignment_count",
        "previous_score",
        "previous_score_2",
        "score_trend",
        "next_assessment_score",
    ]
    for name, mask in buckets.items():
        sub = analysis_df[mask]
        entry: Dict[str, Any] = {
            "count": int(len(sub)),
            "share_pct": round(float(len(sub) / len(analysis_df) * 100.0), 2) if len(analysis_df) else 0.0,
            "mean_predicted_prob": round(float(sub["_prob"].mean()), 4) if len(sub) else None,
        }
        for col in summary_cols:
            if col in sub.columns:
                valid = sub[col].dropna()
                entry[f"mean_{col}"] = round(float(valid.mean()), 2) if len(valid) else None
        summary[name] = entry

    return summary


def _compute_period_breakdown(
    df_eval: pd.DataFrame,
    model_pipe: Any,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluates per-period generalization across unseen evaluation rows."""
    breakdown: Dict[str, Any] = {}
    for p in sorted(df_eval["assessment_period"].unique()):
        sub = df_eval[df_eval["assessment_period"] == p]
        X_sub, y_sub, _ = extract_features_and_target(sub)
        probs = model_pipe.predict_proba(X_sub)[:, 1]
        metrics = compute_comprehensive_metrics(y_sub.to_numpy(), probs, threshold=threshold)
        breakdown[f"period_{int(p)}"] = metrics
    return breakdown


def _save_calibration_figure(
    cal_before: Dict[str, Any],
    cal_after: Dict[str, Any],
    champion_name: str,
) -> Path:
    """Generates Figure 11: Reliability Diagram comparing uncalibrated vs calibrated probabilities."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x_b = [b["mean_predicted_prob"] for b in cal_before["bins"]]
    y_b = [b["empirical_positive_rate"] for b in cal_before["bins"]]
    x_a = [b["mean_predicted_prob"] for b in cal_after["bins"]]
    y_a = [b["empirical_positive_rate"] for b in cal_after["bins"]]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration (y = x)")
    ax.plot(
        x_b,
        y_b,
        marker="s",
        linewidth=2.0,
        color="#F59E0B",
        label=f"Uncalibrated {champion_name} (Brier={cal_before['brier_score']:.4f})",
    )
    ax.plot(
        x_a,
        y_a,
        marker="o",
        linewidth=2.2,
        color="#10B981",
        label=f"Sigmoid Calibrated {champion_name} (Brier={cal_after['brier_score']:.4f})",
    )

    ax.set_title(f"Probability Calibration Curve ({champion_name} — Held-Out Temporal Test)")
    ax.set_xlabel("Mean Predicted At-Risk Probability")
    ax.set_ylabel("Observed At-Risk Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path = FIGURES_DIR / "11_calibration_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Calibration reliability diagram saved to {out_path}")
    return out_path


def train_and_evaluate_all(random_seed: int = RANDOM_SEED):
    """Execute complete benchmarking, calibration, selection, and evaluation suite."""
    logger.info("Starting EduGuard AI Zero-Leakage Training & Calibration Pipeline...")

    # 1. Load, Validate & Audit Dataset
    validator = StudentDataValidator(RAW_DATA_PATH)
    raw_df = validator.load_data()
    data_quality_report = validator.run_all_validations(raw_df, save_report=True, report_path=DATA_QUALITY_REPORT_PATH)

    auditor = LeakageAuditor(RAW_DATA_PATH)
    auditor.run_full_audit(raw_df)

    # 2. Generate Dual Validation Splits & Save Processed Partitions
    logger.info("Partitioning data into Temporal and Student-Disjoint splits...")
    t_train_df, t_val_df, t_test_df = split_temporal(raw_df)
    d_train_df, d_val_df, d_test_df = split_student_disjoint(raw_df, random_seed=random_seed)

    # Enforce Zero-Leakage Split Audits
    auditor.audit_temporal_split(
        set(t_train_df["assessment_period"].unique()),
        set(t_val_df["assessment_period"].unique()),
        set(t_test_df["assessment_period"].unique()),
    )
    auditor.audit_split_disjointness(
        set(d_train_df[ID_COLUMN].unique()),
        set(d_val_df[ID_COLUMN].unique()),
        set(d_test_df[ID_COLUMN].unique()),
    )

    # Persist processed splits cleanly to data/processed/
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_train_df.to_csv(PROCESSED_DATA_DIR / "train_temporal.csv", index=False)
    t_val_df.to_csv(PROCESSED_DATA_DIR / "val_temporal.csv", index=False)
    t_test_df.to_csv(PROCESSED_DATA_DIR / "test_temporal.csv", index=False)
    d_train_df.to_csv(PROCESSED_DATA_DIR / "train_disjoint.csv", index=False)
    d_val_df.to_csv(PROCESSED_DATA_DIR / "val_disjoint.csv", index=False)
    d_test_df.to_csv(PROCESSED_DATA_DIR / "test_disjoint.csv", index=False)
    logger.info(f"Saved processed temporal & disjoint split CSVs to {PROCESSED_DATA_DIR}")

    # Partition Validation Sets strictly into disjoint Calibration vs Selection/Threshold halves
    t_val_cal_df, t_val_sel_df = split_validation_for_calibration(
        t_val_df, group_by_student=False, cal_fraction=0.50, random_seed=random_seed
    )
    d_val_cal_df, d_val_sel_df = split_validation_for_calibration(
        d_val_df, group_by_student=True, cal_fraction=0.50, random_seed=random_seed
    )

    # Extract feature matrices and targets
    X_t_train, y_t_train, _ = extract_features_and_target(t_train_df)
    X_t_val_cal, y_t_val_cal, _ = extract_features_and_target(t_val_cal_df)
    X_t_val_sel, y_t_val_sel, _ = extract_features_and_target(t_val_sel_df)
    X_t_test, y_t_test, _ = extract_features_and_target(t_test_df)

    X_d_train, y_d_train, _ = extract_features_and_target(d_train_df)
    X_d_val_cal, y_d_val_cal, _ = extract_features_and_target(d_val_cal_df)
    X_d_val_sel, y_d_val_sel, _ = extract_features_and_target(d_val_sel_df)
    X_d_test, y_d_test, _ = extract_features_and_target(d_test_df)

    # 3. Candidate Training, Calibration & Validation Comparison
    candidate_models = initialize_candidate_models(random_seed)
    benchmark_results: Dict[str, Any] = {}
    selection_scores: Dict[str, Any] = {}
    fitted_t_calibrated: Dict[str, Any] = {}
    fitted_t_uncalibrated: Dict[str, Any] = {}
    fitted_d_calibrated: Dict[str, Any] = {}
    fitted_d_uncalibrated: Dict[str, Any] = {}

    for name, (estimator_template, scale_features) in candidate_models.items():
        logger.info(f"--- Benchmarking Candidate: {name} ---")

        # A. Temporal Branch (Fit on P1-P2 -> Calibrate on P3 Cal -> Evaluate on P3 Sel)
        t0 = time.time()
        base_pipe_t = build_full_pipeline(clone(estimator_template), scale_features=scale_features)
        base_pipe_t.fit(X_t_train, y_t_train)

        uncal_probs_t_sel = base_pipe_t.predict_proba(X_t_val_sel)[:, 1]
        uncal_metrics_t_sel = compute_comprehensive_metrics(
            y_t_val_sel.to_numpy(), uncal_probs_t_sel, threshold=0.50
        )

        if name == "DummyClassifier":
            cal_pipe_t = base_pipe_t
            is_calibrated = False
        else:
            cal_pipe_t = CalibratedClassifierCV(FrozenEstimator(base_pipe_t), method=CALIBRATION_METHOD)
            cal_pipe_t.fit(X_t_val_cal, y_t_val_cal)
            is_calibrated = True
        train_duration = time.time() - t0

        # Inference speed benchmark on validation selection batch
        lat_slice = X_t_val_sel.iloc[: min(1000, len(X_t_val_sel))]
        t_inf_0 = time.time()
        _ = cal_pipe_t.predict_proba(lat_slice)
        inference_latency_ms = ((time.time() - t_inf_0) / max(1, len(lat_slice))) * 1000.0

        cal_probs_t_sel = cal_pipe_t.predict_proba(X_t_val_sel)[:, 1]
        val_metrics_t_sel = compute_comprehensive_metrics(
            y_t_val_sel.to_numpy(), cal_probs_t_sel, threshold=0.50
        )
        cand_thresh_opt = optimize_classification_threshold(y_t_val_sel.to_numpy(), cal_probs_t_sel)

        # B. Student-Disjoint Branch (Fit on 70% Students -> Calibrate on Val_Cal Students -> Evaluate on Val_Sel Students)
        base_pipe_d = build_full_pipeline(clone(estimator_template), scale_features=scale_features)
        base_pipe_d.fit(X_d_train, y_d_train)
        uncal_probs_d_sel = base_pipe_d.predict_proba(X_d_val_sel)[:, 1]
        uncal_metrics_d_sel = compute_comprehensive_metrics(
            y_d_val_sel.to_numpy(), uncal_probs_d_sel, threshold=0.50
        )

        if name == "DummyClassifier":
            cal_pipe_d = base_pipe_d
        else:
            cal_pipe_d = CalibratedClassifierCV(FrozenEstimator(base_pipe_d), method=CALIBRATION_METHOD)
            cal_pipe_d.fit(X_d_val_cal, y_d_val_cal)

        cal_probs_d_sel = cal_pipe_d.predict_proba(X_d_val_sel)[:, 1]
        val_metrics_d_sel = compute_comprehensive_metrics(
            y_d_val_sel.to_numpy(), cal_probs_d_sel, threshold=0.50
        )

        benchmark_results[name] = {
            "is_calibrated": is_calibrated,
            "calibration_method": CALIBRATION_METHOD if is_calibrated else None,
            "training_time_sec": round(train_duration, 4),
            "inference_latency_ms": round(inference_latency_ms, 4),
            "temporal_validation": val_metrics_t_sel,
            "temporal_validation_optimized": cand_thresh_opt["validation_metrics"],
            "student_disjoint_validation": val_metrics_d_sel,
            "calibration_comparison": {
                "temporal_brier_before": uncal_metrics_t_sel["brier_score"],
                "temporal_brier_after": val_metrics_t_sel["brier_score"],
                "disjoint_brier_before": uncal_metrics_d_sel["brier_score"],
                "disjoint_brier_after": val_metrics_d_sel["brier_score"],
            },
        }

        fitted_t_calibrated[name] = cal_pipe_t
        fitted_t_uncalibrated[name] = base_pipe_t
        fitted_d_calibrated[name] = cal_pipe_d
        fitted_d_uncalibrated[name] = base_pipe_d

        if name != "DummyClassifier":
            mean_ap = 0.5 * (val_metrics_t_sel["average_precision"] + val_metrics_d_sel["average_precision"])
            mean_roc = 0.5 * (val_metrics_t_sel["roc_auc"] + val_metrics_d_sel["roc_auc"])
            val_opt_f2 = cand_thresh_opt["validation_metrics"]["f2_score"]
            mean_brier = 0.5 * (val_metrics_t_sel["brier_score"] + val_metrics_d_sel["brier_score"])
            composite = (0.35 * mean_ap) + (0.25 * mean_roc) + (0.25 * val_opt_f2) + (0.15 * (1.0 - mean_brier))
            selection_scores[name] = {
                "mean_average_precision": round(mean_ap, 4),
                "mean_roc_auc": round(mean_roc, 4),
                "temporal_val_optimized_f2": round(val_opt_f2, 4),
                "mean_brier_score": round(mean_brier, 4),
                "composite_score": round(composite, 4),
            }

        logger.info(
            f"{name} -> Temp Val AP: {val_metrics_t_sel['average_precision']:.3f}, "
            f"Brier: {uncal_metrics_t_sel['brier_score']:.4f}->{val_metrics_t_sel['brier_score']:.4f} | "
            f"Disj Val AP: {val_metrics_d_sel['average_precision']:.3f}, "
            f"Brier: {uncal_metrics_d_sel['brier_score']:.4f}->{val_metrics_d_sel['brier_score']:.4f}"
        )

    # 4. Programmatic Non-Hardcoded Champion Selection Strictly on Validation Selection Folds
    ranked_candidates = sorted(
        selection_scores.items(),
        key=lambda kv: (kv[1]["composite_score"], kv[1]["mean_average_precision"]),
        reverse=True,
    )
    champion_name = ranked_candidates[0][0]
    champion_score_detail = ranked_candidates[0][1]

    selection_rationale = (
        f"{champion_name} was selected programmatically as champion via composite validation score "
        f"({champion_score_detail['composite_score']:.4f} = 0.35*MeanAP + 0.25*MeanROCAUC + 0.25*ValOptF2 + 0.15*(1-MeanBrier)) "
        f"over non-dummy candidates evaluated exclusively on validation selection folds "
        f"(Temp AP={benchmark_results[champion_name]['temporal_validation']['average_precision']:.3f}, "
        f"Disjoint AP={benchmark_results[champion_name]['student_disjoint_validation']['average_precision']:.3f}, "
        f"Mean Brier={champion_score_detail['mean_brier_score']:.4f})."
    )
    logger.info(f"CHAMPION SELECTED PROGRAMMATICALLY: {champion_name} | Score={champion_score_detail['composite_score']:.4f}")

    champion_pipeline = fitted_t_calibrated[champion_name]
    champion_uncal_t = fitted_t_uncalibrated[champion_name]
    champion_disjoint_pipeline = fitted_d_calibrated[champion_name]
    champion_base_pipeline, champion_is_calibrated = unwrap_base_pipeline(champion_pipeline)
    champion_arch = champion_base_pipeline.named_steps["classifier"].__class__.__name__

    # 5. Optimize Decision Threshold on Temporal Validation Selection Fold ONLY
    logger.info(f"Optimizing safety-net decision threshold on Validation Selection fold for {champion_name}...")
    val_sel_probs = champion_pipeline.predict_proba(X_t_val_sel)[:, 1]
    threshold_optimization = optimize_classification_threshold(y_t_val_sel.to_numpy(), val_sel_probs)
    selected_threshold = threshold_optimization["selected_threshold"]

    # 6. Single Final Evaluation on Strictly Held-Out Test Sets
    test_probs_t_cal = champion_pipeline.predict_proba(X_t_test)[:, 1]
    test_probs_t_uncal = champion_uncal_t.predict_proba(X_t_test)[:, 1]
    test_metrics_temporal = compute_comprehensive_metrics(
        y_t_test.to_numpy(), test_probs_t_cal, threshold=selected_threshold
    )

    test_probs_d_cal = champion_disjoint_pipeline.predict_proba(X_d_test)[:, 1]
    test_metrics_disjoint = compute_comprehensive_metrics(
        y_d_test.to_numpy(), test_probs_d_cal, threshold=selected_threshold
    )

    # Calibration Reliability Analysis & Figure
    cal_curve_before = compute_calibration_curve_stats(y_t_test.to_numpy(), test_probs_t_uncal, n_bins=10)
    cal_curve_after = compute_calibration_curve_stats(y_t_test.to_numpy(), test_probs_t_cal, n_bins=10)
    cal_plot_path = _save_calibration_figure(cal_curve_before, cal_curve_after, champion_name)

    calibration_summary = {
        "method": CALIBRATION_METHOD,
        "temporal_val_brier_before": benchmark_results[champion_name]["calibration_comparison"]["temporal_brier_before"],
        "temporal_val_brier_after": benchmark_results[champion_name]["calibration_comparison"]["temporal_brier_after"],
        "disjoint_val_brier_before": benchmark_results[champion_name]["calibration_comparison"]["disjoint_brier_before"],
        "disjoint_val_brier_after": benchmark_results[champion_name]["calibration_comparison"]["disjoint_brier_after"],
        "temporal_test_before": cal_curve_before,
        "temporal_test_after": cal_curve_after,
        "calibration_plot_path": str(cal_plot_path),
    }

    # Per-Period Breakdown & Held-Out Temporal Test Error Analysis
    disjoint_test_by_period = _compute_period_breakdown(d_test_df, champion_disjoint_pipeline, selected_threshold)
    temporal_test_error_analysis = _compute_error_analysis(
        t_test_df, y_t_test.to_numpy(), test_probs_t_cal, selected_threshold
    )

    logger.info(
        f"Final Held-Out Test Performance ({champion_name}, Calibrated={champion_is_calibrated}, Threshold={selected_threshold:.2f}):\n"
        f"  Temporal Test   -> AP: {test_metrics_temporal['average_precision']:.3f}, Recall: {test_metrics_temporal['recall']:.3f}, "
        f"Precision: {test_metrics_temporal['precision']:.3f}, Specificity: {test_metrics_temporal['specificity']:.3f}, "
        f"F1: {test_metrics_temporal['f1_score']:.3f}, ROC-AUC: {test_metrics_temporal['roc_auc']:.3f}, Brier: {test_metrics_temporal['brier_score']:.4f}\n"
        f"  Disjoint Test   -> AP: {test_metrics_disjoint['average_precision']:.3f}, Recall: {test_metrics_disjoint['recall']:.3f}, "
        f"Precision: {test_metrics_disjoint['precision']:.3f}, Specificity: {test_metrics_disjoint['specificity']:.3f}, "
        f"F1: {test_metrics_disjoint['f1_score']:.3f}, ROC-AUC: {test_metrics_disjoint['roc_auc']:.3f}, Brier: {test_metrics_disjoint['brier_score']:.4f}"
    )

    # 7. Persist Production Calibrated Pipeline Artifact
    logger.info(f"Serializing calibrated champion pipeline to {MODEL_ARTIFACT_PATH}...")
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, MODEL_ARTIFACT_PATH)

    # 8. Export Complete Metadata & Evaluation JSONs
    env_versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }

    model_metadata = {
        "model_version": MODEL_VERSION,
        "model_name": champion_name,
        "model_architecture": champion_arch,
        "is_calibrated": champion_is_calibrated,
        "calibration_method": CALIBRATION_METHOD if champion_is_calibrated else None,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "feature_names": FEATURE_COLUMNS,
        "derived_feature_rule": (
            "previous_score_avg = rolling mean of (previous_score, previous_score_2) with single-score fallback to previous_score; "
            "score_trend = previous_score - previous_score_2 (0.0 if previous_score_2 is missing)."
        ),
        "target_definition": "1 if next assessment score < 40 else 0",
        "selected_threshold": selected_threshold,
        "threshold_optimization_rationale": threshold_optimization["rationale"],
        "selection_rationale": selection_rationale,
        "selection_formula": "0.35 * Mean_AP + 0.25 * Mean_ROC_AUC + 0.25 * Temporal_Val_Opt_F2 + 0.15 * (1 - Mean_Brier)",
        "candidate_selection_scores": selection_scores,
        "validation_metrics": threshold_optimization["validation_metrics"],
        "final_test_metrics_temporal": test_metrics_temporal,
        "final_test_metrics_disjoint": test_metrics_disjoint,
        "calibration_summary": {
            "method": CALIBRATION_METHOD,
            "temporal_val_brier_before": calibration_summary["temporal_val_brier_before"],
            "temporal_val_brier_after": calibration_summary["temporal_val_brier_after"],
            "disjoint_val_brier_before": calibration_summary["disjoint_val_brier_before"],
            "disjoint_val_brier_after": calibration_summary["disjoint_val_brier_after"],
            "temporal_test_brier_before": cal_curve_before["brier_score"],
            "temporal_test_brier_after": cal_curve_after["brier_score"],
            "temporal_test_ece_after": cal_curve_after["expected_calibration_error"],
        },
        "split_sizes": {
            "temporal_train_rows": int(len(t_train_df)),
            "temporal_val_cal_rows": int(len(t_val_cal_df)),
            "temporal_val_sel_rows": int(len(t_val_sel_df)),
            "temporal_test_rows": int(len(t_test_df)),
            "disjoint_train_rows": int(len(d_train_df)),
            "disjoint_val_cal_rows": int(len(d_val_cal_df)),
            "disjoint_val_sel_rows": int(len(d_val_sel_df)),
            "disjoint_test_rows": int(len(d_test_df)),
        },
        "data_generator_parameters": {
            "num_students": NUM_STUDENTS,
            "num_periods": NUM_PERIODS,
            "total_snapshots": int(data_quality_report["total_records"]),
            "random_seed": random_seed,
        },
        "environment_versions": env_versions,
        "risk_categories": RISK_CATEGORIES,
        "disclaimer": DISCLAIMER_TEXT,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"Model metadata exported to {METADATA_PATH}")

    evaluation_export = {
        "champion_model": champion_name,
        "champion_architecture": champion_arch,
        "is_calibrated": champion_is_calibrated,
        "calibration_method": CALIBRATION_METHOD if champion_is_calibrated else None,
        "selected_threshold": selected_threshold,
        "selection_rationale": selection_rationale,
        "candidate_selection_scores": selection_scores,
        "model_benchmarks": benchmark_results,
        "temporal_test_evaluation": test_metrics_temporal,
        "disjoint_test_evaluation": test_metrics_disjoint,
        "disjoint_test_by_period": disjoint_test_by_period,
        "calibration_analysis": calibration_summary,
        "temporal_test_error_analysis": temporal_test_error_analysis,
        "threshold_optimization_sweep": threshold_optimization["sweep_grid"],
    }
    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_export, f, indent=2)
    logger.info(f"Comprehensive evaluation benchmarks exported to {EVALUATION_METRICS_PATH}")

    return champion_pipeline, model_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, calibrate, select, and evaluate EduGuard AI models.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for splits and models.")
    args = parser.parse_args()
    train_and_evaluate_all(random_seed=args.seed)
