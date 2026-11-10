"""
EduGuard AI - Data Leakage & Integrity Audit Suite
Enforces strict verification against target leakage, feature contamination,
identifier leakage, demographic feature intrusion, and train/test cross-contamination.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure EduGuard-AI project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    OUTCOME_SCORE_COLUMN,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.LeakageAudit")


class LeakageViolationError(Exception):
    """Raised when data leakage or governance violations are detected."""
    pass


class LeakageAuditor:
    """Rigorous audit engine enforcing zero data leakage in training and inference."""

    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "name", "student_name", "gender", "sex", "race", "ethnicity",
        "religion", "caste", "disability", "income", "socioeconomic",
        "ses", "zipcode", "postal_code", "address", "phone", "email"
    }

    def __init__(self, raw_data_path: Optional[Path] = None):
        self.raw_data_path = raw_data_path or RAW_DATA_PATH

    def load_data(self) -> pd.DataFrame:
        """Load dataset from disk."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.raw_data_path}")
        return pd.read_csv(self.raw_data_path)

    def audit_feature_matrix_isolation(self, feature_list: List[str]) -> Dict[str, Any]:
        """
        Verify that future outcome, target, and identifier never appear in feature columns.
        """
        leaked_cols = []
        if OUTCOME_SCORE_COLUMN in feature_list:
            leaked_cols.append(OUTCOME_SCORE_COLUMN)
        if TARGET_COLUMN in feature_list:
            leaked_cols.append(TARGET_COLUMN)
        if ID_COLUMN in feature_list:
            leaked_cols.append(ID_COLUMN)

        if leaked_cols:
            raise LeakageViolationError(
                f"CRITICAL LEAKAGE: Prohibited columns detected in feature matrix: {leaked_cols}"
            )

        return {
            "status": "PASS",
            "feature_count": len(feature_list),
            "prohibited_columns_detected": 0,
            "inspected_features": feature_list,
        }

    def audit_forbidden_attributes(self, columns: List[str]) -> Dict[str, Any]:
        """
        Scan all column names for sensitive/protected demographic attributes.
        Splits column names by underscores/delimiters to avoid sub-word false positives.
        """
        detected = []
        for col in columns:
            col_tokens = set(col.strip().lower().split("_"))
            for forbidden in self.FORBIDDEN_ATTRIBUTES:
                if forbidden in col_tokens:
                    detected.append((col, forbidden))

        if detected:
            raise LeakageViolationError(
                f"RESPONSIBLE AI VIOLATION: Sensitive attributes detected: {detected}"
            )

        return {
            "status": "PASS",
            "forbidden_attributes_found": 0,
            "checked_tokens": list(self.FORBIDDEN_ATTRIBUTES),
        }

    def audit_target_correlations(self, df: pd.DataFrame, max_allowed_corr: float = 0.90) -> Dict[str, Any]:
        """
        Ensure no feature is trivially correlated with the target (|r| >= max_allowed_corr),
        which would indicate an unrealistic synthetic shortcut or leaked target information.
        """
        correlations = {}
        excessive = {}

        for col in FEATURE_COLUMNS:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Drop NAs for correlation calculation
                valid_mask = df[col].notna() & df[TARGET_COLUMN].notna()
                if valid_mask.sum() > 10:
                    r = float(np.corrcoef(df.loc[valid_mask, col], df.loc[valid_mask, TARGET_COLUMN])[0, 1])
                    correlations[col] = round(r, 4)
                    if abs(r) >= max_allowed_corr:
                        excessive[col] = round(r, 4)

        if excessive:
            raise LeakageViolationError(
                f"POTENTIAL TARGET LEAKAGE: Features with suspicious near-perfect correlation (|r| >= {max_allowed_corr}): {excessive}"
            )

        return {
            "status": "PASS",
            "correlations": correlations,
            "max_observed_abs_r": max([abs(v) for v in correlations.values()]) if correlations else 0.0,
        }

    def audit_split_disjointness(
        self,
        train_students: Set[str],
        val_students: Set[str],
        test_students: Set[str],
    ) -> Dict[str, Any]:
        """
        Verify complete student-disjoint isolation across train, validation, and test sets.
        """
        train_val_overlap = train_students.intersection(val_students)
        train_test_overlap = train_students.intersection(test_students)
        val_test_overlap = val_students.intersection(test_students)

        total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
        if total_overlap > 0:
            raise LeakageViolationError(
                f"STUDENT SPLIT CONTAMINATION: Overlaps detected! "
                f"Train/Val: {len(train_val_overlap)}, Train/Test: {len(train_test_overlap)}, Val/Test: {len(val_test_overlap)}"
            )

        return {
            "status": "PASS",
            "train_students": len(train_students),
            "val_students": len(val_students),
            "test_students": len(test_students),
            "overlap_count": 0,
        }

    def audit_temporal_split(
        self,
        train_periods: Set[int],
        val_periods: Set[int],
        test_periods: Set[int],
    ) -> Dict[str, Any]:
        """
        Verify temporal directionality: train periods < val periods < test periods.
        """
        max_train = max(train_periods)
        min_val = min(val_periods)
        max_val = max(val_periods)
        min_test = min(test_periods)

        if max_train >= min_val:
            raise LeakageViolationError(
                f"TEMPORAL LEAKAGE: Train periods {train_periods} overlap with or succeed Val periods {val_periods}"
            )
        if max_val >= min_test:
            raise LeakageViolationError(
                f"TEMPORAL LEAKAGE: Val periods {val_periods} overlap with or succeed Test periods {test_periods}"
            )

        return {
            "status": "PASS",
            "train_periods": sorted(list(train_periods)),
            "val_periods": sorted(list(val_periods)),
            "test_periods": sorted(list(test_periods)),
        }

    def run_full_audit(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute full anti-leakage audit on configured dataset."""
        if df is None:
            df = self.load_data()

        logger.info("Executing comprehensive Anti-Leakage & Governance Audit...")
        report = {
            "feature_isolation": self.audit_feature_matrix_isolation(FEATURE_COLUMNS),
            "forbidden_attributes": self.audit_forbidden_attributes(list(df.columns)),
            "target_correlations": self.audit_target_correlations(df),
        }
        logger.info("Anti-leakage audit passed: zero leakage detected across all checks.")
        return report


if __name__ == "__main__":
    auditor = LeakageAuditor()
    report = auditor.run_full_audit()
    print("\n=== EduGuard Anti-Leakage Audit Summary ===")
    for check, res in report.items():
        print(f"[{check.upper()}]: {res}")
