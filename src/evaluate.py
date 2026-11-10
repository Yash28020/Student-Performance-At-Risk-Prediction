"""
EduGuard AI - Evaluation & Split Strategy Module
Implements Dual Validation with Zero-Leakage Calibration Partitions:
A. Temporal Evaluation (Periods 1, 2 -> Train; Period 3 -> Val [Cal + Sel]; Period 4 -> Test)
B. Student-Disjoint Evaluation (70% Students -> Train; 15% -> Val [Cal + Sel]; 15% -> Test)
Plus rigorous metric computation (Average Precision, Trapezoidal PR-AUC, ROC-AUC,
Specificity, Precision, Recall, F1, F2, Brier Score, ECE) and Validation Threshold Optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import ID_COLUMN, RANDOM_SEED, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.Evaluate")


def split_temporal(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits longitudinal snapshots temporally:
    - Train: Earlier periods (Periods 1 and 2) -> 5,000 snapshots
    - Validation: Intermediate period (Period 3) -> 2,500 snapshots
    - Test: Latest prediction cycle (Period 4) -> 2,500 snapshots

    Measures out-of-time forecasting performance as syllabus difficulty rises.
    """
    train_df = df[df["assessment_period"].isin([1, 2])].copy()
    val_df = df[df["assessment_period"] == 3].copy()
    test_df = df[df["assessment_period"] == 4].copy()

    logger.info(
        f"Temporal Split generated: Train={len(train_df)} rows (P1-P2), "
        f"Val={len(val_df)} rows (P3), Test={len(test_df)} rows (P4)"
    )
    return train_df, val_df, test_df


def split_student_disjoint(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset along unique students so NO student appears across multiple splits.
    - Train Students: 70% (1,750 students, 7,000 snapshots)
    - Val Students: 15% (375 students, 1,500 snapshots)
    - Test Students: 15% (375 students, 1,500 snapshots)

    Measures generalization to completely unseen student cohorts.
    """
    unique_students = np.array(sorted(df[ID_COLUMN].unique()))
    rng = np.random.RandomState(random_seed)
    rng.shuffle(unique_students)

    n_total = len(unique_students)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(unique_students[:n_train])
    val_ids = set(unique_students[n_train : n_train + n_val])
    test_ids = set(unique_students[n_train + n_val :])

    assert len(train_ids.intersection(val_ids)) == 0, "Train-Val student overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train-Test student overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val-Test student overlap detected!"

    train_df = df[df[ID_COLUMN].isin(train_ids)].copy()
    val_df = df[df[ID_COLUMN].isin(val_ids)].copy()
    test_df = df[df[ID_COLUMN].isin(test_ids)].copy()

    logger.info(
        f"Student-Disjoint Split generated: Train={len(train_df)} rows ({len(train_ids)} students), "
        f"Val={len(val_df)} rows ({len(val_ids)} students), "
        f"Test={len(test_df)} rows ({len(test_ids)} students)"
    )
    return train_df, val_df, test_df


def split_validation_for_calibration(
    val_df: pd.DataFrame,
    group_by_student: bool = False,
    cal_fraction: float = 0.50,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions validation data into two strictly disjoint halves:
    - val_cal_df: Used solely to fit post-hoc probability calibration (Platt/sigmoid).
    - val_sel_df: Used strictly for candidate model comparison and threshold optimization.

    This prevents calibration overfitting from biasing threshold selection or leaking into test sets.
    """
    if len(val_df) < 4:
        raise ValueError("Validation slice must contain at least 4 samples to partition calibration/selection.")

    if group_by_student:
        val_students = np.array(sorted(val_df[ID_COLUMN].unique()))
        rng = np.random.RandomState(random_seed)
        rng.shuffle(val_students)
        n_cal = max(1, int(len(val_students) * cal_fraction))
        cal_ids = set(val_students[:n_cal])
        sel_ids = set(val_students[n_cal:])
        assert len(cal_ids.intersection(sel_ids)) == 0, "Calibration/Selection student overlap detected!"
        val_cal_df = val_df[val_df[ID_COLUMN].isin(cal_ids)].copy()
        val_sel_df = val_df[val_df[ID_COLUMN].isin(sel_ids)].copy()
        return val_cal_df, val_sel_df

    y_val = val_df[TARGET_COLUMN].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - cal_fraction), random_state=random_seed)
    cal_idx, sel_idx = next(splitter.split(val_df, y_val))
    return val_df.iloc[cal_idx].copy(), val_df.iloc[sel_idx].copy()


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Calculates full classification, ranking, and calibration metrics with zero-denominator guards.

    Explicitly separates:
    - `average_precision`: Exact step-wise area under precision-recall curve (`sklearn.metrics.average_precision_score`).
    - `pr_auc`: Trapezoidal integration of precision-recall curve (`sklearn.metrics.auc(recall, precision)`).
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)

    if len(y_true_arr) == 0:
        raise ValueError("y_true cannot be empty when computing metrics.")
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(f"Mismatched lengths: y_true={len(y_true_arr)}, y_prob={len(y_prob_arr)}")

    y_pred = (y_prob_arr >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    assert int(tn + fp + fn + tp) == len(y_true_arr), "Confusion matrix cells must sum to sample_size!"

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    p = float(precision_score(y_true_arr, y_pred, zero_division=0))
    r = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    f2 = float((5.0 * p * r) / ((4.0 * p) + r)) if ((4.0 * p) + r) > 0.0 else 0.0
    acc = float(accuracy_score(y_true_arr, y_pred))

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) >= 2:
        avg_precision = float(average_precision_score(y_true_arr, y_prob_arr))
        precisions, recalls, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        trapezoidal_pr_auc = float(auc(recalls, precisions))
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        pos_fallback = float(np.mean(y_true_arr))
        avg_precision = pos_fallback
        trapezoidal_pr_auc = pos_fallback
        roc_auc = 0.5

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "average_precision": round(avg_precision, 4),
        "pr_auc": round(trapezoidal_pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "sample_size": int(len(y_true_arr)),
        "positive_rate": round(float(np.mean(y_true_arr)), 4),
    }


def compute_calibration_curve_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes empirical calibration reliability curve bins and Expected Calibration Error (ECE).
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: List[Dict[str, Any]] = []
    ece = 0.0
    total = len(y_true_arr)

    for i in range(n_bins):
        low, high = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob_arr >= low) & (y_prob_arr <= high)
        else:
            mask = (y_prob_arr >= low) & (y_prob_arr < high)

        count = int(mask.sum())
        if count > 0:
            mean_pred = float(y_prob_arr[mask].mean())
            frac_pos = float(y_true_arr[mask].mean())
            ece += (count / total) * abs(frac_pos - mean_pred)
            bins_out.append({
                "bin_range": f"{low:.1f}-{high:.1f}",
                "count": count,
                "mean_predicted_prob": round(mean_pred, 4),
                "empirical_positive_rate": round(frac_pos, 4),
            })

    return {
        "brier_score": round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4),
        "expected_calibration_error": round(float(ece), 4),
        "bins": bins_out,
    }


def optimize_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.55,
    candidate_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Optimizes decision threshold strictly on held-out VALIDATION SELECTION DATA ONLY.

    Pedagogical trade-off rationale:
    - False Negatives (FN): Missing a student who scores < 40 means lost early mentoring support.
    - False Positives (FP): Alerting on a student who scores >= 40 causes minor teacher check-in
      overhead without student penalties.
    - Objective: Maximize F2 score (weighing recall 2x precision) subject to precision >= min_precision.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.round(np.arange(0.10, 0.86, 0.02), 3)

    sweep_results = []
    best_threshold = 0.25
    best_f2 = -1.0
    best_metrics = None

    for thresh in candidate_thresholds:
        metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=float(thresh))
        sweep_results.append(metrics)

        if metrics["precision"] >= min_precision:
            if metrics["f2_score"] > best_f2:
                best_f2 = metrics["f2_score"]
                best_threshold = float(thresh)
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = max(sweep_results, key=lambda m: (m["f1_score"], m["recall"]))
        best_threshold = best_metrics["threshold"]

    logger.info(
        f"Validation Threshold Optimization selected: {best_threshold:.2f} "
        f"(Recall: {best_metrics['recall']:.3f}, Precision: {best_metrics['precision']:.3f}, "
        f"F2: {best_metrics['f2_score']:.3f}, Specificity: {best_metrics['specificity']:.3f})"
    )

    return {
        "selected_threshold": round(float(best_threshold), 3),
        "rationale": (
            f"Threshold {best_threshold:.2f} was selected strictly on the validation selection fold "
            f"(N={best_metrics['sample_size']}) to maximize F2 safety-net score ({best_metrics['f2_score']:.3f}) "
            f"subject to precision >= {min_precision:.2f}. It achieved {best_metrics['recall']*100:.1f}% recall, "
            f"{best_metrics['precision']*100:.1f}% precision, and {best_metrics['specificity']*100:.1f}% specificity."
        ),
        "validation_metrics": best_metrics,
        "sweep_grid": sweep_results,
    }
