"""
EduGuard AI - Configuration Module
Centralized settings, file paths, feature schemas, hyperparameters, and thresholds.
"""

from pathlib import Path
import os

# Base directory: points to the EduGuard-AI project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directory Hierarchy
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure essential directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "student_performance.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "eduguard_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.json"
INTERVENTIONS_PATH = PROCESSED_DATA_DIR / "intervention_notes.json"

# Random Seed for Global Reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Synthetic Data Generation Parameters
NUM_STUDENTS = 2500
NUM_PERIODS = 5  # Longitudinal assessment cycles 0..5 generating prediction snapshots for periods 1..4
SUPPORTED_ASSESSMENT_PERIODS = (1, 2, 3, 4)
AT_RISK_THRESHOLD_SCORE = 40.0  # Scores strictly below this are considered "at risk"

# Canonical Model Feature Schema
# CRITICAL: next_assessment_score and student_id are strictly excluded from FEATURES
FEATURE_COLUMNS = [
    "attendance_pct",
    "assignment_completion_pct",
    "late_assignment_count",
    "previous_score",
    "previous_score_2",
    "previous_score_avg",
    "score_trend",
    "assessment_period",
]

TARGET_COLUMN = "at_risk"
OUTCOME_SCORE_COLUMN = "next_assessment_score"
ID_COLUMN = "student_id"

# Allowed Valid Ranges for Input Validation
FEATURE_RANGES = {
    "attendance_pct": (0.0, 100.0),
    "assignment_completion_pct": (0.0, 100.0),
    "late_assignment_count": (0.0, 30.0),
    "previous_score": (0.0, 100.0),
    "previous_score_2": (0.0, 100.0),
    "previous_score_avg": (0.0, 100.0),
    "score_trend": (-100.0, 100.0),
    "assessment_period": (1, 4),
}

# Calibration & Default Decision Threshold
CALIBRATION_METHOD = "sigmoid"
DEFAULT_THRESHOLD = 0.25  # Default fallback calibrated probability threshold if metadata unavailable

# Explicit CORS Origins for API Security
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

# Model Version
MODEL_VERSION = "1.1.0"

# Risk Categories and Explanations
RISK_CATEGORIES = {
    "LOW_RISK": {
        "label": "Low Risk",
        "description": "Student trajectory suggests strong likelihood of reaching assessment benchmarks (>= 40). Continue standard classroom engagement.",
        "color": "#10B981",  # Emerald Green
    },
    "MODERATE_RISK": {
        "label": "Moderate Risk",
        "description": "Student trajectory demonstrates emerging instability (e.g., lower assignment completion or declining trend). Recommend light teacher check-in.",
        "color": "#F59E0B",  # Amber
    },
    "HIGH_RISK": {
        "label": "High Risk / Priority Advisory",
        "description": "Student trajectory indicates elevated risk of scoring below 40. Timely teacher review, mentoring check-in, or study support recommended.",
        "color": "#EF4444",  # Crimson Red
    },
}

# Responsible AI & Advisory Guidance Notice
DISCLAIMER_TEXT = (
    "EduGuard AI is strictly an advisory school intelligence tool designed to help teachers and academic coordinators identify students who may benefit from support. "
    "Model probabilities and SHAP attributions represent statistical associations observed in synthetic historical cohorts, NOT certainties, causal proof, or real-world verified guarantees. "
    "EduGuard AI outputs MUST NEVER be used to automatically assign grades, penalize students, restrict enrollment, "
    "or replace professional teacher judgment."
)
