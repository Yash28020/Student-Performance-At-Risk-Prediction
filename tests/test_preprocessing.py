"""
EduGuard AI - Test Suite: Preprocessing & Feature Engineering
Validates imputation, scaling, pipeline integrity, canonical ordering, and leakage prevention.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, ID_COLUMN, OUTCOME_SCORE_COLUMN, TARGET_COLUMN
from src.evaluate import compute_comprehensive_metrics, split_validation_for_calibration
from src.preprocessing import (
    create_preprocessor,
    extract_features_and_target,
    build_full_pipeline,
    unwrap_base_pipeline,
)
from src.feature_engineering import (
    apply_feature_engineering,
    compute_derived_features,
    prepare_inference_features,
    validate_inference_input,
)


def test_extract_features_and_target_leakage_safeguard():
    """Verify that next_assessment_score and student_id are excluded from feature matrix X."""
    raw_df = pd.DataFrame({
        ID_COLUMN: ["STU_1", "STU_2"],
        OUTCOME_SCORE_COLUMN: [35.0, 78.0],
        TARGET_COLUMN: [1, 0],
        "attendance_pct": [70.0, 95.0],
        "assignment_completion_pct": [65.0, 90.0],
        "late_assignment_count": [3, 0],
        "previous_score": [45.0, 80.0],
        "previous_score_2": [50.0, 75.0],
        "previous_score_avg": [47.5, 77.5],
        "score_trend": [-5.0, 5.0],
        "assessment_period": [2, 2],
    })

    X, y, student_ids = extract_features_and_target(raw_df)

    assert OUTCOME_SCORE_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert ID_COLUMN not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(y) == 2
    assert len(student_ids) == 2


def test_preprocessor_handles_missing_values():
    """Verify median imputer handles missing values without errors."""
    df_with_nan = pd.DataFrame({
        "attendance_pct": [np.nan, 80.0, 90.0],
        "assignment_completion_pct": [70.0, np.nan, 85.0],
        "late_assignment_count": [1, 2, 0],
        "previous_score": [50.0, 60.0, 70.0],
        "previous_score_2": [np.nan, 55.0, 65.0],
        "previous_score_avg": [50.0, 57.5, 67.5],
        "score_trend": [0.0, 5.0, 5.0],
        "assessment_period": [1, 2, 3],
    })

    preprocessor = create_preprocessor(scale_features=True)
    transformed = preprocessor.fit_transform(df_with_nan)

    assert not np.isnan(transformed).any(), "NaN values found after preprocessing!"
    assert transformed.shape == (3, len(FEATURE_COLUMNS))


def test_full_pipeline_fitting_and_prediction():
    """Verify end-to-end Pipeline trains and infers probabilities."""
    X = pd.DataFrame(np.random.uniform(20, 90, size=(100, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = np.random.choice([0, 1], size=100)

    pipeline = build_full_pipeline(LogisticRegression(random_state=42), scale_features=True)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == 100
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_compute_derived_features():
    """Verify authoritative rolling-2 score mean, momentum, and Period-0 fallback parity."""
    res = compute_derived_features(
        previous_score=60.0,
        previous_score_2=50.0,
        historical_scores=[45.0, 50.0, 60.0],
    )
    assert res["score_trend"] == 10.0
    assert res["previous_score_avg"] == 55.0

    # Fallback when prior score t-1 is missing
    res_missing = compute_derived_features(previous_score=62.5, previous_score_2=None)
    assert res_missing["previous_score_avg"] == 62.5
    assert res_missing["score_trend"] == 0.0


def test_training_serving_feature_parity():
    """Verify batch dataframe engineering and live inference preparation match identically."""
    batch_df = pd.DataFrame({
        "attendance_pct": [75.0, 82.0],
        "assignment_completion_pct": [68.0, 90.0],
        "late_assignment_count": [2, 0],
        "previous_score": [38.0, 70.0],
        "previous_score_2": [46.0, np.nan],
        "assessment_period": [3, 1],
    })
    engineered_df = apply_feature_engineering(batch_df)

    for idx, row in batch_df.iterrows():
        live = prepare_inference_features(
            {
                "attendance_pct": row["attendance_pct"],
                "assignment_completion_pct": row["assignment_completion_pct"],
                "late_assignment_count": int(row["late_assignment_count"]),
                "previous_score": row["previous_score"],
                "previous_score_2": None if pd.isna(row["previous_score_2"]) else float(row["previous_score_2"]),
                "previous_score_avg": 99.9,  # stale client value must be overwritten
                "score_trend": 99.9,
                "assessment_period": int(row["assessment_period"]),
            },
            recompute_derived=True,
        )
        assert live["previous_score_avg"] == engineered_df.loc[idx, "previous_score_avg"]
        assert live["score_trend"] == engineered_df.loc[idx, "score_trend"]


def test_validate_inference_input_rejects_out_of_bounds():
    """Ensure invalid inference payloads, invalid periods, and NaNs raise ValueError."""
    bad_payload = {
        "attendance_pct": 125.0,
        "assignment_completion_pct": 80.0,
        "late_assignment_count": 0,
        "previous_score": 60.0,
        "previous_score_2": 60.0,
        "previous_score_avg": 60.0,
        "score_trend": 0.0,
        "assessment_period": 2,
    }
    with pytest.raises(ValueError, match="out of valid range"):
        validate_inference_input(bad_payload)

    bad_period = {**bad_payload, "attendance_pct": 80.0, "assessment_period": 5}
    with pytest.raises(ValueError, match="out of valid range|assessment_period"):
        validate_inference_input(bad_period)


def test_evaluate_metrics_separation_and_specificity():
    """Verify average_precision is stored distinct from trapezoidal pr_auc and specificity is accurate."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.05])
    metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.5)
    assert "average_precision" in metrics
    assert "pr_auc" in metrics
    assert "specificity" in metrics
    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] + cm["false_negatives"] == 2
    assert cm["true_negatives"] + cm["false_positives"] == 4
    assert metrics["specificity"] == round(3 / 4, 4)


def test_calibration_split_disjointness_and_unwrap():
    """Verify calibration/selection split is student-disjoint and unwrap extracts inner pipeline."""
    val_df = pd.DataFrame({
        ID_COLUMN: [f"S_{i // 2}" for i in range(20)],
        "assessment_period": [1, 2] * 10,
        "attendance_pct": np.linspace(50, 95, 20),
        "assignment_completion_pct": np.linspace(50, 95, 20),
        "late_assignment_count": [1] * 20,
        "previous_score": np.linspace(30, 85, 20),
        "previous_score_2": np.linspace(32, 82, 20),
        "previous_score_avg": np.linspace(31, 83.5, 20),
        "score_trend": np.zeros(20),
        TARGET_COLUMN: [1 if i < 6 else 0 for i in range(20)],
    })
    cal_df, sel_df = split_validation_for_calibration(val_df, group_by_student=True)
    assert len(set(cal_df[ID_COLUMN]).intersection(set(sel_df[ID_COLUMN]))) == 0

    X = val_df[FEATURE_COLUMNS]
    y = val_df[TARGET_COLUMN]
    pipe = build_full_pipeline(LogisticRegression(random_state=42), scale_features=True)
    pipe.fit(X, y)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
    cal.fit(X, y)
    unwrapped, is_cal = unwrap_base_pipeline(cal)
    assert is_cal is True
    assert hasattr(unwrapped, "named_steps")
    assert "classifier" in unwrapped.named_steps

