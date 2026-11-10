"""
EduGuard AI - Test Suite: Data Validation & Anti-Leakage Audit
Tests schema enforcement, boundary checks, duplicate prevention, and leakage tripwires.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    OUTCOME_SCORE_COLUMN,
    TARGET_COLUMN,
    RAW_DATA_PATH,
)
from src.data_validation import StudentDataValidator, DataValidationError
from src.leakage_audit import LeakageAuditor, LeakageViolationError


@pytest.fixture
def sample_valid_df():
    """Generates a minimal valid mock dataframe obeying rolling-2 derived feature rules."""
    return pd.DataFrame({
        ID_COLUMN: ["STU_001", "STU_001", "STU_002", "STU_002"],
        "assessment_period": [1, 2, 1, 2],
        "attendance_pct": [85.0, 90.0, 75.0, 80.0],
        "assignment_completion_pct": [88.0, 92.0, 70.0, 78.0],
        "late_assignment_count": [1, 0, 2, 1],
        "previous_score": [65.0, 70.0, 50.0, 55.0],
        "previous_score_2": [60.0, 65.0, 48.0, 50.0],
        "previous_score_avg": [62.5, 67.5, 49.0, 52.5],
        "score_trend": [5.0, 5.0, 2.0, 5.0],
        OUTCOME_SCORE_COLUMN: [72.0, 68.0, 52.0, 38.0],
        TARGET_COLUMN: [0, 0, 0, 1],
    })


def test_validation_passes_on_valid_data(sample_valid_df):
    """Ensure valid data passes all validation checks."""
    validator = StudentDataValidator()
    report = validator.run_all_validations(sample_valid_df)
    assert report["schema"]["status"] == "PASS"
    assert report["duplicates"]["status"] == "PASS"
    assert report["ranges"]["status"] == "PASS"
    assert report["temporal"]["status"] == "PASS"


def test_validation_fails_on_missing_column(sample_valid_df):
    """Ensure missing columns trigger DataValidationError."""
    invalid_df = sample_valid_df.drop(columns=["attendance_pct"])
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="missing mandatory columns"):
        validator.validate_schema(invalid_df)


def test_validation_fails_on_duplicates(sample_valid_df):
    """Ensure duplicate student-period entries trigger DataValidationError."""
    dup_df = pd.concat([sample_valid_df, sample_valid_df.iloc[[0]]], ignore_index=True)
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="Duplicate record violation"):
        validator.validate_duplicates(dup_df)


def test_validation_fails_on_out_of_bounds_features(sample_valid_df):
    """Ensure boundary breaches (e.g. attendance > 100) trigger DataValidationError."""
    invalid_df = sample_valid_df.copy()
    invalid_df.loc[0, "attendance_pct"] = 150.0  # Invalid attendance percentage
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="Boundary validation failed"):
        validator.validate_feature_ranges(invalid_df)


def test_leakage_audit_catches_target_in_features():
    """Ensure leakage auditor catches outcome/target in feature list."""
    auditor = LeakageAuditor()
    with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE"):
        auditor.audit_feature_matrix_isolation(FEATURE_COLUMNS + [OUTCOME_SCORE_COLUMN])

    with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE"):
        auditor.audit_feature_matrix_isolation(FEATURE_COLUMNS + [TARGET_COLUMN])

    with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE"):
        auditor.audit_feature_matrix_isolation(FEATURE_COLUMNS + [ID_COLUMN])


def test_leakage_audit_catches_sensitive_attributes():
    """Ensure demographic/sensitive columns are flagged."""
    auditor = LeakageAuditor()
    with pytest.raises(LeakageViolationError, match="Sensitive attributes detected"):
        auditor.audit_forbidden_attributes(["attendance_pct", "gender", "previous_score"])


def test_split_disjointness_catches_overlap():
    """Ensure student overlap between train and test raises LeakageViolationError."""
    auditor = LeakageAuditor()
    train_ids = {"STU_001", "STU_002", "STU_003"}
    val_ids = {"STU_004", "STU_005"}
    test_ids = {"STU_003", "STU_006"}  # STU_003 leaked into test

    with pytest.raises(LeakageViolationError, match="STUDENT SPLIT CONTAMINATION"):
        auditor.audit_split_disjointness(train_ids, val_ids, test_ids)


def test_validation_detects_derived_feature_skew(sample_valid_df):
    """Ensure corrupted previous_score_avg triggers DataValidationError."""
    skewed = sample_valid_df.copy()
    skewed.loc[1, "previous_score_avg"] = 99.0
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="Derived feature consistency failure"):
        validator.validate_derived_feature_consistency(skewed)


def test_validation_rejects_infinite_values(sample_valid_df):
    """Ensure +/- inf numeric entries trigger DataValidationError."""
    inf_df = sample_valid_df.copy()
    inf_df.loc[0, "previous_score"] = np.inf
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="Infinite numeric values detected"):
        validator.validate_feature_ranges(inf_df)


def test_validation_rejects_target_definition_mismatch(sample_valid_df):
    """Ensure target column mismatch against next_assessment_score < 40 fails validation."""
    bad_target_df = sample_valid_df.copy()
    bad_target_df.loc[0, TARGET_COLUMN] = 1  # Score is 72.0 >= 40, must be 0
    validator = StudentDataValidator()
    with pytest.raises(DataValidationError, match="Target derivation mismatch"):
        validator.validate_target_integrity(bad_target_df)

