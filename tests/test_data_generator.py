"""
EduGuard AI - Test Suite: Synthetic Data Generator
Validates reproducibility, dataset dimensions, schema integrity, and anti-leakage requirements.
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
    RANDOM_SEED,
    AT_RISK_THRESHOLD_SCORE,
)
from src.data_generator import generate_synthetic_dataset, save_synthetic_dataset
from src.data_validation import StudentDataValidator


def test_generator_reproducibility():
    """Verify that using the fixed random seed produces strictly identical datasets."""
    df1 = generate_synthetic_dataset(num_students=100, num_periods=5, random_seed=42)
    df2 = generate_synthetic_dataset(num_students=100, num_periods=5, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_dataset_dimensions_and_counts():
    """Verify standard simulation produces 2,500 unique students and 10,000 snapshots."""
    df = generate_synthetic_dataset(num_students=2500, num_periods=5, random_seed=RANDOM_SEED)

    assert len(df) == 10000, f"Expected 10,000 snapshot records, got {len(df)}"
    assert df[ID_COLUMN].nunique() == 2500, f"Expected 2,500 unique students, got {df[ID_COLUMN].nunique()}"
    assert set(df["assessment_period"].unique()) == {1, 2, 3, 4}


def test_schema_completeness():
    """Ensure all required feature columns, identifiers, and targets are present."""
    df = generate_synthetic_dataset(num_students=50, num_periods=5, random_seed=42)

    expected_cols = {ID_COLUMN, OUTCOME_SCORE_COLUMN, TARGET_COLUMN} | set(FEATURE_COLUMNS)
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"


def test_target_definition_alignment():
    """Ensure target at_risk strictly mirrors next_assessment_score < 40."""
    df = generate_synthetic_dataset(num_students=100, num_periods=5, random_seed=42)

    expected_target = (df[OUTCOME_SCORE_COLUMN] < AT_RISK_THRESHOLD_SCORE).astype(int)
    np.testing.assert_array_equal(df[TARGET_COLUMN].values, expected_target.values)


def test_no_forbidden_attributes():
    """Ensure protected, demographic, and sensitive attributes are strictly absent."""
    df = generate_synthetic_dataset(num_students=50, num_periods=5, random_seed=42)
    forbidden_tokens = ["gender", "sex", "race", "ethnicity", "religion", "caste", "disability", "income", "age", "name"]

    for col in df.columns:
        col_lower = col.lower()
        for token in forbidden_tokens:
            assert token not in col_lower, f"Forbidden attribute token '{token}' detected in column '{col}'"


def test_save_synthetic_dataset(tmp_path):
    """Test saving functionality to an explicit path."""
    dest = tmp_path / "test_students.csv"
    saved_path = save_synthetic_dataset(output_path=dest, random_seed=42)

    assert saved_path.exists()
    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 10000


def test_derived_feature_consistency_in_generated_dataset():
    """Verify generated rows maintain exact rolling-2 derived feature consistency post-missingness."""
    df = generate_synthetic_dataset(num_students=300, num_periods=5, random_seed=42)
    consistency = StudentDataValidator().validate_derived_feature_consistency(df)
    assert consistency["status"] == "PASS"
    assert consistency["avg_mismatches"] == 0
    assert consistency["trend_mismatches"] == 0
