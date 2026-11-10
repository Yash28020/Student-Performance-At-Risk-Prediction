"""
EduGuard AI - Data Preprocessing & Scikit-learn Pipeline Module
Builds reusable, leak-free preprocessing pipelines utilizing SimpleImputer,
StandardScaler, and ColumnTransformer for both linear and tree-based estimators.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    OUTCOME_SCORE_COLUMN,
    TARGET_COLUMN,
)
from src.feature_engineering import enforce_feature_order

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.Preprocessing")


def create_preprocessor(scale_features: bool = True) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for preprocessing input features.

    Implements:
    - Median imputation for missing numeric engagement signals (attendance, completion, scores).
    - Optional standard scaling (z-score standardization) required for linear classifiers
      and distance metrics, or pass-through for tree-based models.

    Args:
        scale_features: If True, applies StandardScaler after imputation.

    Returns:
        Configured scikit-learn ColumnTransformer.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
        ],
        remainder="drop",  # Strictly drop any non-feature columns (e.g. ID, target, future scores)
        verbose_feature_names_out=False,
    )
    return preprocessor


def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Safely separates the feature matrix X from target y and student identifiers.

    Enforces:
    - Anti-leakage: OUTCOME_SCORE_COLUMN is permanently discarded from X.
    - Identification isolation: ID_COLUMN is returned separately for group-aware splitting only.
    - Strict feature order: X is indexed solely by canonical FEATURE_COLUMNS.

    Args:
        df: Raw or snapshot DataFrame.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series, student_ids: Optional[pd.Series]).
    """
    # Verify mandatory feature presence
    missing_feats = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_feats:
        raise ValueError(f"DataFrame is missing modeled feature columns: {missing_feats}")

    # Canonical ordered feature matrix
    X = df[FEATURE_COLUMNS].copy()

    # Target series (if present)
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None

    # Student IDs (for group splitting)
    student_ids = df[ID_COLUMN].copy() if ID_COLUMN in df.columns else None

    # CRITICAL LEAKAGE ASSERTIONS
    assert OUTCOME_SCORE_COLUMN not in X.columns, f"CRITICAL LEAKAGE: {OUTCOME_SCORE_COLUMN} found in X!"
    assert TARGET_COLUMN not in X.columns, f"CRITICAL LEAKAGE: {TARGET_COLUMN} found in X!"
    assert ID_COLUMN not in X.columns, f"CRITICAL LEAKAGE: {ID_COLUMN} found in X!"

    return X, y, student_ids


def build_full_pipeline(estimator, scale_features: bool = True) -> Pipeline:
    """
    Constructs an end-to-end composite Pipeline coupling preprocessing and estimator.

    Args:
        estimator: Initialized scikit-learn classifier or XGBoost model.
        scale_features: Whether to include feature scaling in the preprocessor.

    Returns:
        Complete fitted or unfitted scikit-learn Pipeline.
    """
    preprocessor = create_preprocessor(scale_features=scale_features)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
    return pipeline


def unwrap_base_pipeline(model_artifact: Any) -> Tuple[Pipeline, bool]:
    """
    Unwraps either a direct sklearn Pipeline or a CalibratedClassifierCV wrapping
    a FrozenEstimator(Pipeline) into `(base_pipeline, is_calibrated)`.
    """
    if isinstance(model_artifact, Pipeline):
        return model_artifact, False

    candidate = getattr(model_artifact, "estimator", None)
    if candidate is not None:
        inner = getattr(candidate, "estimator", candidate)
        if isinstance(inner, Pipeline):
            return inner, True

    cal_list = getattr(model_artifact, "calibrated_classifiers_", None)
    if cal_list:
        first_est = getattr(cal_list[0], "estimator", None)
        inner = getattr(first_est, "estimator", first_est)
        if isinstance(inner, Pipeline):
            return inner, True

    raise TypeError(
        f"Unsupported model artifact type {type(model_artifact).__name__}: expected Pipeline or CalibratedClassifierCV(Pipeline)."
    )
