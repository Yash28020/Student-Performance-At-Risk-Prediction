"""
EduGuard AI - Test Suite: FastAPI Backend Service
Tests REST endpoint logic, health monitoring, Pydantic boundary validation, and real inference.
Directly exercises endpoint callables and Pydantic schemas for reliable execution across environments.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import (
    app,
    get_or_create_predictor,
    read_root,
    health_check,
    get_model_info,
    predict_single_student,
    predict_batch_students,
    StudentFeatureInput,
    BatchPredictInput,
    PredictionResponse,
)
from src.config import INTERVENTIONS_PATH


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def initialize_service():
    """Ensure predictor is loaded into memory before tests."""
    pred = get_or_create_predictor()
    assert pred is not None


def test_root_endpoint():
    """Verify read_root returns 200 equivalent and operational declaration."""
    data = read_root()
    assert data["status"] == "OPERATIONAL"
    assert "EduGuard AI" in data["platform"]
    assert "advisory_directive" in data
    assert "api_documentation" in data


def test_health_endpoint():
    """Verify health_check reports healthy status and model artifact verification."""
    data = health_check()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "model_artifact" in data


def test_model_info_endpoint():
    """Verify get_model_info returns governance metadata, metrics, and threshold."""
    data = get_model_info()
    assert "champion_architecture" in data
    assert "selected_threshold" in data
    assert 0.0 < data["selected_threshold"] < 1.0
    assert "validation_metrics" in data
    assert "feature_schema" in data
    assert "risk_bands" in data


def test_predict_single_valid_student():
    """Verify predict_single_student generates genuine probabilities, risk band, and support plan."""
    payload = StudentFeatureInput(
        attendance_pct=72.0,
        assignment_completion_pct=65.0,
        late_assignment_count=2,
        previous_score=48.0,
        previous_score_2=52.0,
        previous_score_avg=50.0,
        score_trend=-4.0,
        assessment_period=3,
    )
    res = predict_single_student(payload)

    assert isinstance(res, PredictionResponse)
    assert res.prediction in (0, 1)
    assert 0.0 <= res.risk_probability <= 1.0
    assert res.applied_threshold > 0
    assert res.risk_band.tier in ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK")
    assert len(res.support_plan) > 0
    assert len(res.human_review_disclaimer) > 0


def test_predict_single_rejects_invalid_ranges():
    """Verify Pydantic rejects out-of-bounds features with ValidationError."""
    with pytest.raises(ValidationError):
        StudentFeatureInput(
            attendance_pct=145.0,  # Invalid: > 100
            assignment_completion_pct=85.0,
            late_assignment_count=0,
            previous_score=75.0,
            assessment_period=2,
        )

    with pytest.raises(ValidationError):
        StudentFeatureInput(
            attendance_pct=85.0,
            assignment_completion_pct=-10.0,  # Invalid: < 0
            late_assignment_count=0,
            previous_score=75.0,
            assessment_period=2,
        )


def test_predict_batch_endpoint():
    """Verify predict_batch_students processes cohort records and preserves relative risks."""
    batch_input = BatchPredictInput(
        students=[
            StudentFeatureInput(
                attendance_pct=92.0,
                assignment_completion_pct=95.0,
                late_assignment_count=0,
                previous_score=85.0,
                previous_score_2=80.0,
                previous_score_avg=82.5,
                score_trend=5.0,
                assessment_period=2,
            ),
            StudentFeatureInput(
                attendance_pct=55.0,
                assignment_completion_pct=40.0,
                late_assignment_count=4,
                previous_score=32.0,
                previous_score_2=38.0,
                previous_score_avg=35.0,
                score_trend=-6.0,
                assessment_period=2,
            ),
        ]
    )
    batch_res = predict_batch_students(batch_input)

    assert batch_res["total_records"] == 2
    assert len(batch_res["predictions"]) == 2
    assert batch_res["predictions"][0]["row_index"] == 0
    assert batch_res["predictions"][1]["row_index"] == 1

    prob_student1 = batch_res["predictions"][0]["risk_probability"]
    prob_student2 = batch_res["predictions"][1]["risk_probability"]
    assert prob_student1 < prob_student2, "Expected high-engagement student to have lower risk than struggling student"


def test_api_http_rejects_invalid_period_booleans_and_nan(client):
    """Verify HTTPX TestClient enforces 422 validation against invalid period, bool, and non-finite inputs."""
    base = {
        "attendance_pct": 80.0,
        "assignment_completion_pct": 80.0,
        "late_assignment_count": 1,
        "previous_score": 65.0,
        "previous_score_2": 60.0,
        "assessment_period": 2,
    }
    resp_period = client.post("/predict", json={**base, "assessment_period": 5})
    assert resp_period.status_code == 422

    resp_bool = client.post("/predict", json={**base, "attendance_pct": True})
    assert resp_bool.status_code == 422

    resp_nan = client.post(
        "/predict",
        content='{"attendance_pct": NaN, "assignment_completion_pct": 80, "late_assignment_count": 1, "previous_score": 65, "assessment_period": 2}',
        headers={"Content-Type": "application/json"},
    )
    assert resp_nan.status_code in (400, 422)


def test_api_recomputes_skewed_client_derived_features(client):
    """Verify /predict overwrites skewed client previous_score_avg/score_trend authoritatively."""
    resp = client.post(
        "/predict",
        json={
            "attendance_pct": 70.0,
            "assignment_completion_pct": 65.0,
            "late_assignment_count": 2,
            "previous_score": 40.0,
            "previous_score_2": 50.0,
            "previous_score_avg": 99.0,  # Intentionally stale/wrong
            "score_trend": 99.0,
            "assessment_period": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_features"]["previous_score_avg"] == 45.0
    assert body["input_features"]["score_trend"] == -10.0


def test_api_model_info_exposes_calibration_ap_and_specificity(client):
    """Verify /model-info exposes real champion name, calibration state, AP, and specificity."""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_calibrated"] is True
    assert "model_name" in body
    temp = body["held_out_test_metrics_temporal"]
    assert "average_precision" in temp
    assert "specificity" in temp
    assert "calibration_analysis" in body


def test_api_interventions_and_reports_endpoints(client):
    """Verify /interventions persistence roundtrip (with cleanup) and /reports/students JSON & CSV exports."""
    orig_content = INTERVENTIONS_PATH.read_text(encoding="utf-8") if INTERVENTIONS_PATH.exists() else None
    try:
        r_get = client.get("/interventions")
        assert r_get.status_code == 200
        assert r_get.json()["total"] >= 1

        r_post = client.post(
            "/interventions",
            json={
                "student_id": "STU_TEST_001",
                "status": "Outreach Scheduled",
                "priority": "High",
                "teacher_notes": "Automated test follow-up note",
                "assessment_period": 4,
            },
        )
        assert r_post.status_code == 200
        assert r_post.json()["saved"] is True
    finally:
        if orig_content is not None:
            INTERVENTIONS_PATH.write_text(orig_content, encoding="utf-8")

    r_rep_json = client.get("/reports/students?period=4&limit=10&format=json")
    assert r_rep_json.status_code == 200
    rep_body = r_rep_json.json()
    assert rep_body["returned_records"] == 10
    assert len(rep_body["period_reports"]) == 4

    r_rep_csv = client.get("/reports/students?period=4&risk_tier=HIGH_RISK&limit=5&format=csv")
    assert r_rep_csv.status_code == 200
    assert "text/csv" in r_rep_csv.headers["content-type"]
    lines = [line for line in r_rep_csv.text.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert "student_id,assessment_period" in lines[0]
