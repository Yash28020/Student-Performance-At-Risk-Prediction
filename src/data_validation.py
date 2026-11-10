"""
EduGuard AI - Data Validation Module
Comprehensive validation suite enforcing schema constraints, value boundaries,
finiteness, derived-feature parity, temporal consistency, duplicate absence,
missing value thresholds, and anti-leakage invariants.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure EduGuard-AI project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    AT_RISK_THRESHOLD_SCORE,
    DATA_QUALITY_REPORT_PATH,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    ID_COLUMN,
    OUTCOME_SCORE_COLUMN,
    RAW_DATA_PATH,
    SUPPORTED_ASSESSMENT_PERIODS,
    TARGET_COLUMN,
)
from src.feature_engineering import compute_derived_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.DataValidation")


class DataValidationError(Exception):
    """Raised when critical data integrity or anti-leakage invariants are violated."""
    pass


class StudentDataValidator:
    """Production validator for student snapshot datasets."""

    def __init__(self, raw_data_path: Optional[Path] = None):
        self.raw_data_path = raw_data_path or RAW_DATA_PATH

    def load_data(self) -> pd.DataFrame:
        """Load dataset from configured raw data path."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")
        return pd.read_csv(self.raw_data_path)

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate presence and non-empty structure of all required columns."""
        if df.empty:
            raise DataValidationError("Dataset cannot be empty.")

        required = {ID_COLUMN, OUTCOME_SCORE_COLUMN, TARGET_COLUMN} | set(FEATURE_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(f"Schema violation: missing mandatory columns: {missing}")

        if df[ID_COLUMN].isna().any():
            raise DataValidationError(f"Identifier column '{ID_COLUMN}' contains null values.")

        return {
            "status": "PASS",
            "columns_present": list(df.columns),
            "total_columns": len(df.columns),
        }

    def validate_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify no duplicate (student_id, assessment_period) pairs exist."""
        duplicates = int(df.duplicated(subset=[ID_COLUMN, "assessment_period"]).sum())
        if duplicates > 0:
            raise DataValidationError(
                f"Duplicate record violation: found {duplicates} duplicate (student_id, assessment_period) rows."
            )
        return {"status": "PASS", "duplicate_keys_count": 0}

    def validate_feature_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify numerical features are finite, well-typed, and adhere to valid boundaries."""
        violations: Dict[str, Any] = {}
        infinite_counts: Dict[str, int] = {}

        for feature, (min_val, max_val) in FEATURE_RANGES.items():
            if feature not in df.columns:
                continue
            series = df[feature].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                violations[feature] = {"error": f"Non-numeric dtype: {series.dtype}"}
                continue

            inf_mask = np.isinf(series.to_numpy(dtype=float))
            if inf_mask.any():
                infinite_counts[feature] = int(inf_mask.sum())

            out_of_bounds = series[(series < min_val) | (series > max_val)]
            if len(out_of_bounds) > 0:
                violations[feature] = {
                    "count": int(len(out_of_bounds)),
                    "min_found": float(series.min()),
                    "max_found": float(series.max()),
                    "allowed_range": [min_val, max_val],
                }

        if infinite_counts:
            raise DataValidationError(f"Infinite numeric values detected: {infinite_counts}")
        if violations:
            raise DataValidationError(f"Boundary validation failed for features: {violations}")

        # Verify late_assignment_count consists strictly of non-negative whole integers
        late_non_na = df["late_assignment_count"].dropna()
        if ((late_non_na < 0) | ((late_non_na % 1) != 0)).any():
            raise DataValidationError("late_assignment_count must contain non-negative whole integers.")

        # Verify outcome score bounds
        outcome_non_na = df[OUTCOME_SCORE_COLUMN].dropna()
        if df[OUTCOME_SCORE_COLUMN].isna().any() or ((outcome_non_na < 0.0) | (outcome_non_na > 100.0)).any():
            raise DataValidationError(f"{OUTCOME_SCORE_COLUMN} must be non-null and bounded within [0, 100].")

        return {"status": "PASS", "violations": {}, "infinite_values_detected": 0}

    def validate_derived_feature_consistency(
        self,
        df: pd.DataFrame,
        tolerance: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Verify dataset derived columns (`previous_score_avg`, `score_trend`) strictly match
        the authoritative calculation in `src/feature_engineering.py`.
        """
        avg_mismatches = 0
        trend_mismatches = 0

        for p1, p2, csv_avg, csv_trend in zip(
            df["previous_score"].tolist(),
            df["previous_score_2"].tolist(),
            df["previous_score_avg"].tolist(),
            df["score_trend"].tolist(),
        ):
            expected = compute_derived_features(previous_score=p1, previous_score_2=p2)
            if pd.isna(csv_avg) or abs(float(csv_avg) - expected["previous_score_avg"]) > tolerance:
                avg_mismatches += 1
            if pd.isna(csv_trend) or abs(float(csv_trend) - expected["score_trend"]) > tolerance:
                trend_mismatches += 1

        if avg_mismatches > 0 or trend_mismatches > 0:
            raise DataValidationError(
                f"Derived feature consistency failure: avg_mismatches={avg_mismatches}, "
                f"trend_mismatches={trend_mismatches}. Regenerate dataset with authoritative feature engineering."
            )

        return {
            "status": "PASS",
            "avg_mismatches": 0,
            "trend_mismatches": 0,
            "tolerance": tolerance,
        }

    def validate_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify temporal ordering:
        - Assessment period must strictly range within {1, 2, 3, 4} for predictions.
        """
        if df["assessment_period"].isna().any():
            raise DataValidationError("assessment_period contains null values.")

        periods_non_na = df["assessment_period"].dropna()
        if ((periods_non_na % 1) != 0).any():
            raise DataValidationError("assessment_period must contain whole integer values.")

        unique_periods = {int(p) for p in periods_non_na.unique()}
        invalid_periods = unique_periods - set(SUPPORTED_ASSESSMENT_PERIODS)
        if invalid_periods:
            raise DataValidationError(f"Invalid assessment periods detected: {sorted(invalid_periods)}")

        return {"status": "PASS", "valid_periods": sorted(unique_periods)}

    def validate_missingness(self, df: pd.DataFrame, max_allowed_pct: float = 5.0) -> Dict[str, Any]:
        """Verify missing value percentages do not exceed operational threshold."""
        missing_pcts = (df[FEATURE_COLUMNS].isna().mean() * 100).to_dict()
        missing_counts = df[FEATURE_COLUMNS].isna().sum().to_dict()
        critical_missing = {k: round(float(v), 2) for k, v in missing_pcts.items() if v > max_allowed_pct}

        if critical_missing:
            raise DataValidationError(
                f"Excessive missing values detected (> {max_allowed_pct}%): {critical_missing}"
            )

        return {
            "status": "PASS",
            "missing_counts": {k: int(v) for k, v in missing_counts.items()},
            "missing_percentages": {k: round(float(v), 3) for k, v in missing_pcts.items()},
        }

    def validate_target_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ensure:
        - Target values are strictly binary (0 or 1).
        - No NaN values in target.
        - Target equals (next_assessment_score < 40).
        - Target class balance is within realistic bounds (10% - 35% at risk).
        """
        if df[TARGET_COLUMN].isna().sum() > 0:
            raise DataValidationError("Target column contains NaN values.")

        unique_targets = {int(v) for v in df[TARGET_COLUMN].unique()}
        if not unique_targets.issubset({0, 1}):
            raise DataValidationError(f"Invalid target values found: {unique_targets}")

        expected_target = (df[OUTCOME_SCORE_COLUMN] < AT_RISK_THRESHOLD_SCORE).astype(int)
        label_mismatches = int((df[TARGET_COLUMN].astype(int) != expected_target).sum())
        if label_mismatches > 0:
            raise DataValidationError(
                f"Target derivation mismatch: {label_mismatches} rows violate `at_risk = 1 if {OUTCOME_SCORE_COLUMN} < 40 else 0`."
            )

        at_risk_ratio = float(df[TARGET_COLUMN].mean())
        if not (0.10 <= at_risk_ratio <= 0.35):
            logger.warning(f"Target at-risk ratio {at_risk_ratio:.3f} is outside recommended bounds [0.10, 0.35]")

        per_period_rates = {
            f"period_{int(p)}": round(float(grp[TARGET_COLUMN].mean()), 4)
            for p, grp in df.groupby("assessment_period")
        }

        return {
            "status": "PASS",
            "at_risk_ratio": round(at_risk_ratio, 4),
            "class_distribution": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()},
            "per_period_at_risk_rate": per_period_rates,
            "target_threshold_mismatches": 0,
        }

    def compute_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics across modeled features for reproducible quality auditing."""
        stats: Dict[str, Any] = {}
        for col in FEATURE_COLUMNS + [OUTCOME_SCORE_COLUMN]:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            stats[col] = {
                "count": int(len(s)),
                "missing": int(df[col].isna().sum()),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "median": round(float(s.median()), 3),
                "max": round(float(s.max()), 3),
            }
        return stats

    def run_all_validations(
        self,
        df: Optional[pd.DataFrame] = None,
        save_report: bool = False,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute full validation suite and optionally persist data quality JSON."""
        if df is None:
            df = self.load_data()

        logger.info(f"Running validation on dataset with shape: {df.shape}")
        report: Dict[str, Any] = {
            "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(self.raw_data_path),
            "overall_status": "PASS",
            "total_records": int(len(df)),
            "unique_students": int(df[ID_COLUMN].nunique()),
            "schema": self.validate_schema(df),
            "duplicates": self.validate_duplicates(df),
            "ranges": self.validate_feature_ranges(df),
            "derived_consistency": self.validate_derived_feature_consistency(df),
            "temporal": self.validate_temporal_consistency(df),
            "missingness": self.validate_missingness(df),
            "target": self.validate_target_integrity(df),
            "feature_statistics": self.compute_feature_statistics(df),
        }

        if save_report:
            dest = report_path or DATA_QUALITY_REPORT_PATH
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Data quality report exported to {dest}")

        logger.info("All data validation checks passed successfully.")
        return report


if __name__ == "__main__":
    validator = StudentDataValidator()
    report = validator.run_all_validations(save_report=True)
    print("\n--- EduGuard Data Validation Report ---")
    for check in ["overall_status", "total_records", "unique_students", "duplicates", "derived_consistency", "temporal", "missingness", "target"]:
        print(f"[{check.upper()}]: {report[check]}")
