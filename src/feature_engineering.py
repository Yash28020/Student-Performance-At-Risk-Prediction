"""
EduGuard AI - Authoritative Feature Engineering Module
Single source of truth shared across synthetic data generation, training, validation,
testing, FastAPI inference, Streamlit dashboard, and automated tests.

Canonical Derived Feature Definitions (Zero Training-Serving Skew):
- previous_score_avg: Rolling mean of the latest two assessments (previous_score and
  previous_score_2). When previous_score_2 is missing/NaN (single-score record or
  Period 1 transfer), falls back deterministically to previous_score.
- score_trend: Score delta (previous_score - previous_score_2). When previous_score_2
  is missing/NaN, evaluates deterministically to 0.0.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURE_RANGES, SUPPORTED_ASSESSMENT_PERIODS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.FeatureEngineering")


def _is_missing(value: Any) -> bool:
    """Return True if a value represents null/missing (None, NaN, pd.NA)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_bounded_score(score: Any, field_name: str = "score") -> float:
    """Validate that an assessment score is a finite number in [0.0, 100.0]."""
    if _is_missing(score) or isinstance(score, bool):
        raise ValueError(f"Invalid {field_name}: value cannot be missing or boolean.")
    try:
        num = float(score)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Feature '{field_name}' must be numeric, got: {score!r}") from exc

    if not math.isfinite(num):
        raise ValueError(f"Feature '{field_name}' must be finite, got: {score!r}")

    min_v, max_v = FEATURE_RANGES["previous_score"]
    if not (min_v <= num <= max_v):
        raise ValueError(
            f"Feature '{field_name}' value {num} out of valid range [{min_v}, {max_v}]"
        )
    return num


def compute_derived_features(
    previous_score: float,
    previous_score_2: Optional[float] = None,
    historical_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Authoritative calculation for derived longitudinal features ('score_trend' and
    'previous_score_avg') using the canonical rolling two-assessment rule.

    If `historical_scores` is supplied, validates every item cleanly (rejecting empty
    lists, nulls, infinities, or out-of-range values). When `previous_score_2` is
    omitted/NaN and `historical_scores` contains >= 2 assessments, the prior assessment
    score is extracted from the second-to-last entry in `historical_scores`.

    Args:
        previous_score: Most recent assessment score (period t), in [0.0, 100.0].
        previous_score_2: Prior assessment score (period t-1), in [0.0, 100.0] or None/NaN.
        historical_scores: Optional chronological score sequence ending with period t.

    Returns:
        Dict with 'previous_score_2_resolved' (optional/float), 'score_trend',
        and 'previous_score_avg' rounded to 2 decimal places.
    """
    curr = _validate_bounded_score(previous_score, "previous_score")

    validated_history: Optional[List[float]] = None
    if historical_scores is not None:
        if not isinstance(historical_scores, (list, tuple)):
            raise ValueError("historical_scores must be a list or tuple of numeric scores.")
        if len(historical_scores) == 0:
            raise ValueError("historical_scores cannot be empty when explicitly provided.")
        validated_history = [
            _validate_bounded_score(s, f"historical_scores[{idx}]")
            for idx, s in enumerate(historical_scores)
        ]

    resolved_prev2: Optional[float] = None
    if not _is_missing(previous_score_2):
        resolved_prev2 = _validate_bounded_score(previous_score_2, "previous_score_2")
    elif validated_history is not None and len(validated_history) >= 2:
        resolved_prev2 = validated_history[-2]

    if resolved_prev2 is None:
        score_trend = 0.0
        score_avg = curr
    else:
        score_trend = float(curr - resolved_prev2)
        score_avg = float((curr + resolved_prev2) / 2.0)

    return {
        "score_trend": round(score_trend, 2),
        "previous_score_avg": round(score_avg, 2),
    }


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the canonical derived-feature calculation deterministically across every row
    of a DataFrame (after any missingness masks have been applied).

    Guarantees 100% mathematical parity between generated/training datasets and live inference.
    """
    if "previous_score" not in df.columns:
        raise ValueError("DataFrame must contain 'previous_score' to compute derived features.")

    out = df.copy()
    has_prev2 = "previous_score_2" in out.columns

    trends = np.zeros(len(out), dtype=float)
    avgs = np.zeros(len(out), dtype=float)

    prev_vals = out["previous_score"].tolist()
    prev2_vals = out["previous_score_2"].tolist() if has_prev2 else [None] * len(out)

    for idx, (p1, p2) in enumerate(zip(prev_vals, prev2_vals)):
        derived = compute_derived_features(previous_score=p1, previous_score_2=p2)
        trends[idx] = derived["score_trend"]
        avgs[idx] = derived["previous_score_avg"]

    out["score_trend"] = trends
    out["previous_score_avg"] = avgs
    return out


def prepare_inference_features(
    payload: Dict[str, Any],
    recompute_derived: bool = True,
) -> Dict[str, Any]:
    """
    Validates raw prediction payload and ensures derived features match the canonical
    authoritative formula cleanly (accepting optional `historical_scores` if passed).

    Args:
        payload: Input feature dictionary.
        recompute_derived: When True (default), computes `previous_score_avg` and
            `score_trend` deterministically via `compute_derived_features` if omitted
            or whenever base scores are supplied.

    Returns:
        Validated dictionary with canonical FEATURE_COLUMNS.
    """
    work = dict(payload)

    if "previous_score" not in work or _is_missing(work.get("previous_score")):
        raise ValueError("Missing required input feature: 'previous_score'")

    # Validate explicit optional derived bounds before deriving/overriding if caller supplied them
    if "previous_score_avg" in work and not _is_missing(work["previous_score_avg"]):
        _ = _validate_bounded_score(work["previous_score_avg"], "previous_score_avg")

    if "score_trend" in work and not _is_missing(work["score_trend"]):
        raw_trend = work["score_trend"]
        if isinstance(raw_trend, bool):
            raise ValueError("Feature 'score_trend' cannot be boolean.")
        try:
            trend_num = float(raw_trend)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature 'score_trend' must be numeric, got: {raw_trend!r}") from exc
        if not math.isfinite(trend_num) or not (-100.0 <= trend_num <= 100.0):
            raise ValueError(f"Feature 'score_trend' value {trend_num} out of valid range [-100.0, 100.0]")

    hist = work.get("historical_scores")
    prev2_raw = work.get("previous_score_2")

    # Resolve previous_score_2 from historical_scores if provided and previous_score_2 missing
    if _is_missing(prev2_raw) and hist is not None:
        if not isinstance(hist, (list, tuple)) or len(hist) == 0:
            raise ValueError("historical_scores must be a non-empty list when provided.")
        valid_hist = [_validate_bounded_score(s, f"historical_scores[{i}]") for i, s in enumerate(hist)]
        if len(valid_hist) >= 2:
            work["previous_score_2"] = valid_hist[-2]

    if recompute_derived or _is_missing(work.get("score_trend")) or _is_missing(work.get("previous_score_avg")):
        derived = compute_derived_features(
            previous_score=work["previous_score"],
            previous_score_2=work.get("previous_score_2"),
            historical_scores=hist,
        )
        work["score_trend"] = derived["score_trend"]
        work["previous_score_avg"] = derived["previous_score_avg"]

    return validate_inference_input(work)


def enforce_feature_order(df: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Ensures input features strictly adhere to the canonical FEATURE_COLUMNS order.

    Args:
        df: Input DataFrame or single dictionary of feature values.

    Returns:
        pd.DataFrame containing exactly FEATURE_COLUMNS in canonical order.
    """
    if isinstance(df, dict):
        df = pd.DataFrame([df])

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing mandatory feature columns: {missing_cols}")

    return df[FEATURE_COLUMNS].copy()


def validate_inference_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a single student inference record against valid bounds, supported
    assessment periods, finiteness, and numeric types.

    Args:
        payload: Dict of feature inputs.

    Returns:
        Sanitized and validated feature dictionary containing all FEATURE_COLUMNS.
    """
    validated: Dict[str, Any] = {}
    optional_nullable = {"attendance_pct", "assignment_completion_pct", "previous_score_2"}

    for col in FEATURE_COLUMNS:
        if col not in payload:
            raise ValueError(f"Missing required input feature: '{col}'")

        val = payload[col]
        if _is_missing(val):
            if col in optional_nullable:
                validated[col] = np.nan
                continue
            raise ValueError(f"Required input feature '{col}' cannot be null or NaN.")

        if isinstance(val, bool):
            raise ValueError(f"Feature '{col}' must be numeric, got boolean: {val}")

        try:
            val_num = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature '{col}' must be numeric, got: {val}") from exc

        if not math.isfinite(val_num):
            raise ValueError(f"Feature '{col}' must be finite, got: {val}")

        min_bound, max_bound = FEATURE_RANGES[col]
        if not (min_bound <= val_num <= max_bound):
            raise ValueError(
                f"Feature '{col}' value {val_num} out of valid range [{min_bound}, {max_bound}]"
            )

        if col == "assessment_period":
            if not float(val_num).is_integer() or int(val_num) not in SUPPORTED_ASSESSMENT_PERIODS:
                raise ValueError(
                    f"Feature 'assessment_period' must be an integer in {SUPPORTED_ASSESSMENT_PERIODS}, got: {val}"
                )
            validated[col] = int(val_num)
        elif col == "late_assignment_count":
            if not float(val_num).is_integer():
                raise ValueError(f"Feature 'late_assignment_count' must be a whole integer, got: {val}")
            validated[col] = int(val_num)
        else:
            validated[col] = val_num

    return validated
