"""
EduGuard AI - Test Suite: Prediction Engine & Advisory Support Planner
Validates persistent pipeline loading, threshold consistency, real inference, and risk categorization.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import EduGuardPredictor, generate_support_recommendations
from src.config import MODEL_ARTIFACT_PATH, METADATA_PATH


@pytest.fixture(scope="module")
def predictor_instance():
    """Module-level fixture to load predictor once for tests."""
    return EduGuardPredictor(enable_shap=False)


def test_predictor_initialization(predictor_instance):
    """Verify persistent pipeline loads cleanly with metadata."""
    assert predictor_instance.pipeline is not None
    assert 0.0 < predictor_instance.threshold < 1.0
    assert predictor_instance.version is not None


def test_real_inference_produces_valid_probabilities(predictor_instance):
    """Ensure prediction probabilities are genuine floats strictly bounded in [0, 1]."""
    sample = {
        "attendance_pct": 82.0,
        "assignment_completion_pct": 78.0,
        "late_assignment_count": 1,
        "previous_score": 62.0,
        "previous_score_2": 58.0,
        "previous_score_avg": 60.0,
        "score_trend": 4.0,
        "assessment_period": 2,
    }
    result = predictor_instance.predict_single(sample, include_shap=False)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["applied_threshold"] == round(predictor_instance.threshold, 3)
    assert result["risk_band"]["tier"] in ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK")
    assert "human_review_disclaimer" in result


def test_threshold_consistency(predictor_instance):
    """Ensure prediction strictly reflects probability >= threshold."""
    low_risk_sample = {
        "attendance_pct": 95.0,
        "assignment_completion_pct": 98.0,
        "late_assignment_count": 0,
        "previous_score": 88.0,
        "previous_score_2": 85.0,
        "previous_score_avg": 86.5,
        "score_trend": 3.0,
        "assessment_period": 3,
    }
    res_low = predictor_instance.predict_single(low_risk_sample, include_shap=False)
    expected_pred = int(res_low["risk_probability"] >= predictor_instance.threshold)
    assert res_low["prediction"] == expected_pred


def test_support_planner_generates_advisory_actions():
    """Verify support planner suggests tailored rule-based advisory actions."""
    features = {
        "attendance_pct": 60.0,  # Below 75
        "assignment_completion_pct": 45.0,  # Below 70
        "late_assignment_count": 4,  # >= 2
        "previous_score": 35.0,  # Below 42
        "score_trend": -12.0,  # <= -8
    }
    plans = generate_support_recommendations(features, risk_prob=0.85, threshold=0.18)

    assert len(plans) >= 4
    categories = [p["category"] for p in plans]
    assert "Attendance & Engagement Check-in" in categories
    assert "Assignment Workflow Support" in categories
    assert "Foundational Mastery Support" in categories
    assert all(p.get("source") == "rule_based_advisory" for p in plans)


def test_shap_local_explanation_and_graceful_fallback():
    """Verify local SHAP generation works cleanly and prediction degrades safely if explainer fails."""
    shap_predictor = EduGuardPredictor(enable_shap=True)
    sample = {
        "attendance_pct": 65.0,
        "assignment_completion_pct": 55.0,
        "late_assignment_count": 3,
        "previous_score": 37.0,
        "previous_score_2": 44.0,
        "assessment_period": 3,
    }
    res = shap_predictor.predict_single(sample, include_shap=True)
    exp = res["explanation"]
    assert exp is not None
    assert exp.get("shap_available") is True
    assert len(exp["feature_contributions"]) == 8
    assert "probability" in exp.get("explanation_space", "")

    # Simulate explainer exception and confirm predict_single still returns valid prediction
    orig_explainer = shap_predictor._explainer
    class BrokenExplainer:
        def explain_instance(self, *args, **kwargs):
            raise RuntimeError("Simulated SHAP failure")
    shap_predictor._explainer = BrokenExplainer()
    try:
        fallback_res = shap_predictor.predict_single(sample, include_shap=True)
        assert 0.0 <= fallback_res["risk_probability"] <= 1.0
        assert fallback_res["explanation"]["shap_available"] is False
    finally:
        shap_predictor._explainer = orig_explainer


def test_batch_prediction_matches_single(predictor_instance):
    """Verify batch scoring aligns with single inference probabilities."""
    rows = [
        {
            "attendance_pct": 90.0,
            "assignment_completion_pct": 92.0,
            "late_assignment_count": 0,
            "previous_score": 82.0,
            "previous_score_2": 78.0,
            "assessment_period": 3,
        },
        {
            "attendance_pct": 52.0,
            "assignment_completion_pct": 48.0,
            "late_assignment_count": 4,
            "previous_score": 34.0,
            "previous_score_2": 40.0,
            "assessment_period": 3,
        },
    ]
    batch = predictor_instance.predict_batch(rows)
    assert len(batch) == 2
    single_0 = predictor_instance.predict_single(rows[0], include_shap=False)
    single_1 = predictor_instance.predict_single(rows[1], include_shap=False)
    assert batch[0]["risk_probability"] == single_0["risk_probability"]
    assert batch[1]["risk_probability"] == single_1["risk_probability"]
    assert batch[0]["model_name"] == predictor_instance.model_name

