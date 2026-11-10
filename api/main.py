"""
EduGuard AI - Production FastAPI Backend Service
Serves RESTful endpoints for real-time and batch student risk prediction,
SHAP explainability, model metadata & calibration introspection, cohort reports,
CSV exports, and persisted teacher intervention tracking.
"""

from contextlib import asynccontextmanager
import csv
import io
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.config import (
    DATA_QUALITY_REPORT_PATH,
    DEFAULT_CORS_ORIGINS,
    DEFAULT_THRESHOLD,
    DISCLAIMER_TEXT,
    EVALUATION_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    INTERVENTIONS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_VERSION,
    RAW_DATA_PATH,
    RISK_CATEGORIES,
    SUPPORTED_ASSESSMENT_PERIODS,
)
from src.feature_engineering import enforce_feature_order, prepare_inference_features
from src.predict import EduGuardPredictor, generate_support_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.API")

# Global Predictor Instance
predictor: Optional[EduGuardPredictor] = None


def get_or_create_predictor() -> EduGuardPredictor:
    """Lazy loader ensuring predictor is available."""
    global predictor
    if predictor is None:
        logger.info("Initializing EduGuard Predictor...")
        predictor = EduGuardPredictor(enable_shap=True)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    get_or_create_predictor()
    yield


# Determine explicit allowed CORS origins (never '*' with allow_credentials=True)
_env_origins = os.getenv("CORS_ORIGINS", "").strip()
ALLOWED_CORS_ORIGINS = (
    [origin.strip() for origin in _env_origins.split(",") if origin.strip()]
    if _env_origins
    else list(DEFAULT_CORS_ORIGINS)
)

# Initialize FastAPI App
app = FastAPI(
    title="EduGuard AI — Academic Intelligence & Early Warning Platform",
    description=(
        "Production-style advisory ML backend forecasting school student assessment risk (score < 40/100). "
        "Strictly advisory; never automated or punitive. Incorporates pre-assessment indicators, "
        "calibrated risk probabilities, SHAP explainability, and non-punitive teacher support plans."
    ),
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


def _sanitize_json_value(value: Any) -> Any:
    """Sanitizes non-finite floats (NaN/Inf) and Exception values for strict RFC 8259 JSON error responses."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, Exception):
        return str(value)
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Returns HTTP 422 cleanly even when rejected JSON payloads contain non-finite NaN/Inf floats."""
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_json_value(exc.errors())},
    )


# --- Pydantic Request & Response Schemas ---

def _ensure_not_bool_and_finite(value: Any, field_name: str) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} cannot be boolean.")
    if value is not None:
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric.") from exc
        if not math.isfinite(num):
            raise ValueError(f"{field_name} must be a finite numeric value.")
    return value


class StudentFeatureInput(BaseModel):
    attendance_pct: Optional[float] = Field(
        ...,
        description="Classroom attendance percentage [0.0 - 100.0] (nullable if transfer record missing)",
        json_schema_extra={"example": 85.5},
    )
    assignment_completion_pct: Optional[float] = Field(
        ...,
        description="Percentage of assigned coursework completed [0.0 - 100.0]",
        json_schema_extra={"example": 90.0},
    )
    late_assignment_count: int = Field(
        ...,
        description="Total number of assignments submitted past deadline [0 - 30]",
        json_schema_extra={"example": 1},
    )
    previous_score: float = Field(
        ...,
        description="Assessment score in current cycle period t [0.0 - 100.0]",
        json_schema_extra={"example": 62.0},
    )
    previous_score_2: Optional[float] = Field(
        None,
        description="Assessment score in prior cycle period t-1 [0.0 - 100.0]",
        json_schema_extra={"example": 58.0},
    )
    previous_score_avg: Optional[float] = Field(
        None,
        description="Rolling two-assessment average score. Authoritatively recomputed in serving pipeline.",
        json_schema_extra={"example": 60.0},
    )
    score_trend: Optional[float] = Field(
        None,
        description="Score delta (previous_score - previous_score_2). Authoritatively recomputed.",
        json_schema_extra={"example": 4.0},
    )
    historical_scores: Optional[List[float]] = Field(
        None,
        description="Optional chronological score sequence up to period t.",
    )
    assessment_period: int = Field(
        2,
        description="Current assessment period index [1 - 4]",
        json_schema_extra={"example": 2},
    )

    @field_validator("attendance_pct", "assignment_completion_pct", "previous_score", "previous_score_2", "previous_score_avg", "score_trend", mode="before")
    @classmethod
    def check_finite_numeric(cls, v, info):
        return _ensure_not_bool_and_finite(v, info.field_name)

    @field_validator("late_assignment_count", "assessment_period", mode="before")
    @classmethod
    def check_int_not_bool(cls, v, info):
        if isinstance(v, bool):
            raise ValueError(f"{info.field_name} cannot be boolean.")
        return v

    @field_validator("attendance_pct")
    @classmethod
    def validate_attendance(cls, v):
        if v is None:
            return v
        min_v, max_v = FEATURE_RANGES["attendance_pct"]
        if not (min_v <= v <= max_v):
            raise ValueError(f"attendance_pct must be within [{min_v}, {max_v}], got {v}")
        return v

    @field_validator("assignment_completion_pct")
    @classmethod
    def validate_completion(cls, v):
        if v is None:
            return v
        min_v, max_v = FEATURE_RANGES["assignment_completion_pct"]
        if not (min_v <= v <= max_v):
            raise ValueError(f"assignment_completion_pct must be within [{min_v}, {max_v}], got {v}")
        return v

    @field_validator("late_assignment_count")
    @classmethod
    def validate_late_count(cls, v):
        min_v, max_v = FEATURE_RANGES["late_assignment_count"]
        if not (min_v <= v <= max_v):
            raise ValueError(f"late_assignment_count must be within [{int(min_v)}, {int(max_v)}], got {v}")
        return v

    @field_validator("previous_score")
    @classmethod
    def validate_prev_score(cls, v):
        min_v, max_v = FEATURE_RANGES["previous_score"]
        if not (min_v <= v <= max_v):
            raise ValueError(f"previous_score must be within [{min_v}, {max_v}], got {v}")
        return v

    @field_validator("previous_score_2", "previous_score_avg")
    @classmethod
    def validate_optional_scores(cls, v, info):
        if v is None:
            return v
        min_v, max_v = FEATURE_RANGES[info.field_name]
        if not (min_v <= v <= max_v):
            raise ValueError(f"{info.field_name} must be within [{min_v}, {max_v}], got {v}")
        return v

    @field_validator("score_trend")
    @classmethod
    def validate_score_trend(cls, v):
        if v is None:
            return v
        min_v, max_v = FEATURE_RANGES["score_trend"]
        if not (min_v <= v <= max_v):
            raise ValueError(f"score_trend must be within [{min_v}, {max_v}], got {v}")
        return v

    @field_validator("historical_scores")
    @classmethod
    def validate_historical_scores(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("historical_scores cannot be empty when supplied.")
        for idx, s in enumerate(v):
            if isinstance(s, bool) or not math.isfinite(float(s)) or not (0.0 <= float(s) <= 100.0):
                raise ValueError(f"historical_scores[{idx}] must be finite in [0, 100], got {s}")
        return [float(s) for s in v]

    @field_validator("assessment_period")
    @classmethod
    def validate_period(cls, v):
        if int(v) not in SUPPORTED_ASSESSMENT_PERIODS:
            raise ValueError(
                f"assessment_period must be in {SUPPORTED_ASSESSMENT_PERIODS}, got {v}"
            )
        return int(v)


class BatchPredictInput(BaseModel):
    students: List[StudentFeatureInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Batch list of student snapshot feature records (1 to 500 items)",
    )


class RiskBandDetail(BaseModel):
    tier: str
    label: str
    color: str
    action: str


class SupportPlanItem(BaseModel):
    category: str
    priority: str
    source: str = "rule_based_advisory"
    advisory_action: str


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    risk_probability: float
    applied_threshold: float
    risk_band: RiskBandDetail
    input_features: Dict[str, Any]
    influential_signals: Optional[Dict[str, Any]] = None
    support_plan: List[SupportPlanItem]
    model_name: Optional[str] = None
    model_version: str
    is_calibrated: Optional[bool] = None
    human_review_disclaimer: str


class InterventionUpsertRequest(BaseModel):
    student_id: str = Field(..., min_length=3, max_length=64)
    status: Literal["Pending Review", "Outreach Scheduled", "Mentoring Active", "Resolved"] = "Pending Review"
    priority: Literal["Urgent", "High", "Medium", "Low", "Routine"] = "High"
    teacher_notes: str = Field("", max_length=2000)
    recommended_support: str = Field("", max_length=1000)
    assessment_period: Optional[int] = Field(None, ge=1, le=4)
    risk_probability: Optional[float] = Field(None, ge=0.0, le=1.0)


# --- Helper Utilities ---

def _sanitize_nans(obj: Any) -> Any:
    """Recursively converts float('nan') / np.nan to None for strict JSON compliance."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


_cached_eval_metrics: Optional[Dict[str, Any]] = None
_cached_quality_report: Optional[Dict[str, Any]] = None
_cached_cohort_df: Optional[Any] = None
_cached_global_analytics: Optional[Dict[str, Any]] = None


def get_evaluation_metrics() -> Dict[str, Any]:
    """Lazy loads and caches evaluation benchmarks from reports/evaluation_metrics.json."""
    global _cached_eval_metrics
    if _cached_eval_metrics is None and EVALUATION_METRICS_PATH.exists():
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            _cached_eval_metrics = json.load(f)
    return _cached_eval_metrics or {}


def get_data_quality_report() -> Dict[str, Any]:
    """Lazy loads data quality audit JSON."""
    global _cached_quality_report
    if _cached_quality_report is None and DATA_QUALITY_REPORT_PATH.exists():
        with open(DATA_QUALITY_REPORT_PATH, "r", encoding="utf-8") as f:
            _cached_quality_report = json.load(f)
    return _cached_quality_report or {}


def get_cohort_dataframe():
    """Lazy loads and caches raw student records for fast querying."""
    global _cached_cohort_df
    if _cached_cohort_df is None and RAW_DATA_PATH.exists():
        import pandas as pd
        _cached_cohort_df = pd.read_csv(RAW_DATA_PATH)
    return _cached_cohort_df


def _load_or_seed_interventions() -> Dict[str, Dict[str, Any]]:
    """Loads persisted intervention records, seeding un-annotated initial queue if absent."""
    if INTERVENTIONS_PATH.exists():
        try:
            with open(INTERVENTIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"Failed reading intervention store, re-seeding: {exc}")

    # Seed initial advisory priority queue deterministically from Period 4 model flags
    seeded: Dict[str, Dict[str, Any]] = {}
    df = get_cohort_dataframe()
    pred = get_or_create_predictor()
    if df is not None and pred is not None:
        latest_df = df[df["assessment_period"] == 4].copy()
        if not latest_df.empty:
            X_latest = enforce_feature_order(latest_df)
            probs = pred.pipeline.predict_proba(X_latest)[:, 1]
            latest_df["_prob"] = probs
            flagged = latest_df.sort_values("_prob", ascending=False).head(15)
            now_iso = datetime.now(timezone.utc).isoformat()
            for _, row in flagged.iterrows():
                feat_dict = {col: row[col] for col in FEATURE_COLUMNS}
                prob_val = round(float(row["_prob"]), 4)
                plans = generate_support_recommendations(feat_dict, prob_val, threshold=pred.threshold)
                top_plan = plans[0]
                sid = str(row["student_id"])
                seeded[sid] = {
                    "student_id": sid,
                    "assessment_period": int(row["assessment_period"]),
                    "risk_probability": prob_val,
                    "priority": top_plan["priority"],
                    "recommended_support": f"{top_plan['category']}: {top_plan['advisory_action']}",
                    "status": "Pending Review",
                    "teacher_notes": "",
                    "source": "rule_based_advisory",
                    "updated_at_utc": now_iso,
                }

    INTERVENTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERVENTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(seeded, f, indent=2)
    return seeded


def _save_interventions(store: Dict[str, Dict[str, Any]]) -> None:
    INTERVENTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERVENTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


# --- Core API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root platform status and advisory declaration."""
    pred = get_or_create_predictor()
    return {
        "platform": "EduGuard AI — Academic Intelligence & Early Warning Platform",
        "tagline": "Detect Early. Understand Deeply. Support Intelligently.",
        "status": "OPERATIONAL",
        "model_name": pred.model_name,
        "model_version": pred.version,
        "advisory_directive": (
            "EduGuard AI predictions are statistical decision aids for school teachers and coordinators. "
            "They must NEVER be used to automatically assign grades, penalize students, "
            "or replace human academic mentoring."
        ),
        "api_documentation": "/docs",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Healthcheck endpoint reporting model artifact readiness and calibration state."""
    pred = get_or_create_predictor()
    if pred is None or pred.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor pipeline is not loaded or unavailable.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": pred.model_name,
        "champion_architecture": pred.model_architecture,
        "is_calibrated": pred.is_calibrated,
        "selected_threshold": pred.threshold,
        "model_artifact": str(MODEL_ARTIFACT_PATH.name),
        "model_version": pred.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Governance"])
def get_model_info():
    """Returns architecture, validation benchmarks, calibration summary, error analysis, and governance metadata."""
    pred = get_or_create_predictor()
    eval_metrics = get_evaluation_metrics()
    quality = get_data_quality_report()
    return {
        "model_version": pred.version,
        "model_name": pred.model_name,
        "champion_architecture": pred.metadata.get("model_architecture", pred.model_architecture),
        "is_calibrated": pred.metadata.get("is_calibrated", pred.is_calibrated),
        "calibration_method": pred.metadata.get("calibration_method"),
        "calibration_summary": pred.metadata.get("calibration_summary"),
        "selection_rationale": pred.metadata.get("selection_rationale"),
        "selection_formula": pred.metadata.get("selection_formula"),
        "candidate_selection_scores": pred.metadata.get("candidate_selection_scores", {}),
        "target_definition": "Next Assessment Score < 40.0 (At-Risk = 1)",
        "derived_feature_rule": pred.metadata.get("derived_feature_rule"),
        "selected_threshold": pred.threshold,
        "threshold_rationale": pred.metadata.get("threshold_optimization_rationale"),
        "validation_metrics": pred.metadata.get("validation_metrics"),
        "held_out_test_metrics_temporal": pred.metadata.get("final_test_metrics_temporal"),
        "held_out_test_metrics_disjoint": pred.metadata.get("final_test_metrics_disjoint"),
        "feature_schema": pred.metadata.get("feature_names", FEATURE_COLUMNS),
        "risk_bands": RISK_CATEGORIES,
        "responsible_ai_disclaimer": DISCLAIMER_TEXT,
        "training_timestamp_utc": pred.metadata.get("training_timestamp_utc"),
        "split_sizes": pred.metadata.get("split_sizes"),
        "data_generator_parameters": pred.metadata.get("data_generator_parameters"),
        "environment_versions": pred.metadata.get("environment_versions"),
        "model_benchmarks": eval_metrics.get("model_benchmarks", {}),
        "disjoint_test_by_period": eval_metrics.get("disjoint_test_by_period", {}),
        "calibration_analysis": eval_metrics.get("calibration_analysis", {}),
        "temporal_test_error_analysis": eval_metrics.get("temporal_test_error_analysis", {}),
        "threshold_optimization_sweep": eval_metrics.get("threshold_optimization_sweep", []),
        "data_quality_status": quality.get("overall_status"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_student(payload: StudentFeatureInput):
    """
    Real-time assessment risk prediction using authoritative derived-feature calculation,
    calibrated risk probability, local SHAP signals, and clearly labeled rule-based support plans.
    """
    pred = get_or_create_predictor()
    raw_dict = payload.model_dump()

    try:
        canonical_features = prepare_inference_features(raw_dict, recompute_derived=True)
        inference_res = pred.predict_single(canonical_features, include_shap=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction inference failure: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return PredictionResponse(
        prediction=inference_res["prediction"],
        prediction_label=inference_res["prediction_label"],
        risk_probability=inference_res["risk_probability"],
        applied_threshold=inference_res["applied_threshold"],
        risk_band=RiskBandDetail(**inference_res["risk_band"]),
        input_features=_sanitize_nans(inference_res["input_features"]),
        influential_signals=_sanitize_nans(inference_res["explanation"]),
        support_plan=[SupportPlanItem(**sp) for sp in inference_res["support_plan"]],
        model_name=inference_res.get("model_name"),
        model_version=inference_res["model_version"],
        is_calibrated=inference_res.get("is_calibrated"),
        human_review_disclaimer=inference_res["human_review_disclaimer"],
    )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch_students(payload: BatchPredictInput):
    """Batch student snapshot inference using authoritative feature engineering."""
    pred = get_or_create_predictor()

    try:
        records = [
            prepare_inference_features(item.model_dump(), recompute_derived=True)
            for item in payload.students
        ]
        batch_results = pred.predict_batch(records)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failure: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    flagged_count = sum(1 for r in batch_results if r["prediction"] == 1)
    indexed_results = [{"row_index": idx, **row} for idx, row in enumerate(batch_results)]
    return _sanitize_nans({
        "total_records": len(indexed_results),
        "flagged_at_risk_count": flagged_count,
        "applied_threshold": pred.threshold,
        "model_name": pred.model_name,
        "model_version": pred.version,
        "is_calibrated": pred.is_calibrated,
        "predictions": indexed_results,
    })


# --- Cohort, Student Search, Analytics & Interventions Endpoints ---

@app.get("/students", tags=["Cohort Data"])
def get_students(
    search: Optional[str] = None,
    period: Optional[int] = None,
    risk_status: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Search and filter students from simulated longitudinal records."""
    if period is not None and period not in SUPPORTED_ASSESSMENT_PERIODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"period must be within {SUPPORTED_ASSESSMENT_PERIODS}, got {period}",
        )
    if risk_status is not None and risk_status not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="risk_status must be 0 or 1.",
        )
    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))

    df = get_cohort_dataframe()
    if df is None:
        return {"total": 0, "limit": limit, "offset": offset, "students": []}

    filtered = df.copy()
    if search:
        search_upper = search.strip().upper()
        filtered = filtered[filtered["student_id"].str.contains(search_upper, na=False)]
    if period is not None:
        filtered = filtered[filtered["assessment_period"] == period]
    if risk_status is not None:
        filtered = filtered[filtered["at_risk"] == risk_status]

    total_count = int(len(filtered))
    page = filtered.iloc[offset : offset + limit]

    records = _sanitize_nans(page.to_dict(orient="records"))
    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "students": records,
    }


@app.get("/students/{student_id}", tags=["Cohort Data"])
def get_student_detail(student_id: str):
    """
    Get full multi-period history for a specific student, plus real-time ML & SHAP appraisal
    for their latest snapshot.
    """
    df = get_cohort_dataframe()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cohort data unavailable.")

    sid_clean = student_id.strip().upper()
    match = df[df["student_id"] == sid_clean].sort_values("assessment_period")
    if match.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student {student_id} not found.")

    records = match.where(match.notna(), None).to_dict(orient="records")
    latest = records[-1] if records else None

    latest_prediction = None
    if latest is not None:
        pred = get_or_create_predictor()
        feature_payload = {col: latest.get(col) for col in FEATURE_COLUMNS}
        latest_prediction = _sanitize_nans(pred.predict_single(feature_payload, include_shap=True))

    return _sanitize_nans({
        "student_id": sid_clean,
        "total_snapshots": len(records),
        "history": records,
        "latest_snapshot": latest,
        "latest_prediction": latest_prediction,
    })


@app.get("/cohort-summary", tags=["Cohort Data"])
def get_cohort_summary():
    """Returns aggregated metrics, engagement bands, and longitudinal period trends."""
    df = get_cohort_dataframe()
    if df is None:
        return {}

    total_students = int(df["student_id"].nunique())
    total_snapshots = int(len(df))
    at_risk_count = int(df["at_risk"].sum())
    at_risk_rate = round(float(at_risk_count / total_snapshots) * 100, 2)

    period_trends = []
    for p in sorted(df["assessment_period"].unique()):
        sub = df[df["assessment_period"] == p]
        passing_sub = sub[sub["at_risk"] == 0]
        risk_sub = sub[sub["at_risk"] == 1]
        period_trends.append({
            "period": int(p),
            "avg_attendance": round(float(sub["attendance_pct"].mean()), 2),
            "avg_completion": round(float(sub["assignment_completion_pct"].mean()), 2),
            "avg_score": round(float(sub["previous_score"].mean()), 2),
            "passing_avg_score": round(float(passing_sub["previous_score"].mean()), 2) if len(passing_sub) else None,
            "at_risk_avg_score": round(float(risk_sub["previous_score"].mean()), 2) if len(risk_sub) else None,
            "at_risk_count": int(sub["at_risk"].sum()),
            "at_risk_rate": round(float(sub["at_risk"].mean()) * 100, 2),
        })

    att_bands = [
        {"band": "< 60% Critical", "count": int((df["attendance_pct"] < 60).sum()), "risk_rate": round(float(df[df["attendance_pct"] < 60]["at_risk"].mean() * 100), 1)},
        {"band": "60-75% Low", "count": int(((df["attendance_pct"] >= 60) & (df["attendance_pct"] < 75)).sum()), "risk_rate": round(float(df[(df["attendance_pct"] >= 60) & (df["attendance_pct"] < 75)]["at_risk"].mean() * 100), 1)},
        {"band": "75-85% Moderate", "count": int(((df["attendance_pct"] >= 75) & (df["attendance_pct"] < 85)).sum()), "risk_rate": round(float(df[(df["attendance_pct"] >= 75) & (df["attendance_pct"] < 85)]["at_risk"].mean() * 100), 1)},
        {"band": "85-100% High", "count": int((df["attendance_pct"] >= 85).sum()), "risk_rate": round(float(df[df["attendance_pct"] >= 85]["at_risk"].mean() * 100), 1)},
    ]

    comp_bands = [
        {"band": "< 50% Lagging", "count": int((df["assignment_completion_pct"] < 50).sum()), "risk_rate": round(float(df[df["assignment_completion_pct"] < 50]["at_risk"].mean() * 100), 1)},
        {"band": "50-70% Partial", "count": int(((df["assignment_completion_pct"] >= 50) & (df["assignment_completion_pct"] < 70)).sum()), "risk_rate": round(float(df[(df["assignment_completion_pct"] >= 50) & (df["assignment_completion_pct"] < 70)]["at_risk"].mean() * 100), 1)},
        {"band": "70-85% Substantial", "count": int(((df["assignment_completion_pct"] >= 70) & (df["assignment_completion_pct"] < 85)).sum()), "risk_rate": round(float(df[(df["assignment_completion_pct"] >= 70) & (df["assignment_completion_pct"] < 85)]["at_risk"].mean() * 100), 1)},
        {"band": "85-100% Complete", "count": int((df["assignment_completion_pct"] >= 85).sum()), "risk_rate": round(float(df[df["assignment_completion_pct"] >= 85]["at_risk"].mean() * 100), 1)},
    ]

    latest_df = df.sort_values("assessment_period").groupby("student_id").tail(1)
    latest_at_risk = int(latest_df["at_risk"].sum())

    return {
        "total_unique_students": total_students,
        "total_snapshots": total_snapshots,
        "at_risk_count": at_risk_count,
        "passing_count": total_snapshots - at_risk_count,
        "at_risk_rate_pct": at_risk_rate,
        "latest_period_at_risk_students": latest_at_risk,
        "avg_attendance": round(float(df["attendance_pct"].mean()), 1),
        "avg_completion": round(float(df["assignment_completion_pct"].mean()), 1),
        "avg_late_assignments": round(float(df["late_assignment_count"].mean()), 2),
        "avg_score": round(float(df["previous_score"].mean()), 1),
        "period_trends": period_trends,
        "attendance_bands": att_bands,
        "completion_bands": comp_bands,
    }


@app.get("/analytics/global", tags=["Cohort Data"])
def get_global_analytics():
    """
    Computes and caches global SHAP feature importance, feature correlations with at-risk status,
    and a representative cohort sample for constellation and risk cluster visualization.
    """
    global _cached_global_analytics
    if _cached_global_analytics is not None:
        return _cached_global_analytics

    df = get_cohort_dataframe()
    if df is None:
        return {}

    pred = get_or_create_predictor()
    sample_df = df.sample(n=min(400, len(df)), random_state=42)

    global_importance = []
    if pred and pred.explainer is not None:
        X_df = sample_df[FEATURE_COLUMNS].copy()
        X_trans = np.asarray(pred.explainer.preprocessor.transform(X_df), dtype=float)
        shap_raw = pred.explainer.explainer.shap_values(X_trans)
        sv, _ = pred.explainer._extract_positive_class_arrays(
            shap_raw, pred.explainer.explainer.expected_value
        )
        mean_abs = np.mean(np.abs(sv), axis=0)
        global_importance = sorted(
            [
                {"feature": col, "mean_absolute_shap": round(float(val), 4)}
                for col, val in zip(FEATURE_COLUMNS, mean_abs)
            ],
            key=lambda x: x["mean_absolute_shap"],
            reverse=True,
        )

    corr_cols = FEATURE_COLUMNS + ["at_risk"]
    corr_series = df[corr_cols].corr(numeric_only=True)["at_risk"].drop("at_risk")
    correlations = [
        {"feature": col, "correlation_with_risk": round(float(val), 4)}
        for col, val in corr_series.items()
    ]
    correlations.sort(key=lambda x: abs(x["correlation_with_risk"]), reverse=True)

    constellation_df = df.sample(n=min(180, len(df)), random_state=7).dropna(
        subset=["attendance_pct", "assignment_completion_pct", "previous_score"]
    )
    X_const = constellation_df[FEATURE_COLUMNS]
    const_probs = pred.pipeline.predict_proba(X_const)[:, 1] if pred else np.zeros(len(constellation_df))
    constellation_points = []
    for (_, row), prob in zip(constellation_df.iterrows(), const_probs):
        constellation_points.append({
            "student_id": str(row["student_id"]),
            "period": int(row["assessment_period"]),
            "attendance_pct": round(float(row["attendance_pct"]), 1),
            "assignment_completion_pct": round(float(row["assignment_completion_pct"]), 1),
            "previous_score": round(float(row["previous_score"]), 1),
            "score_trend": round(float(row["score_trend"]), 1) if row["score_trend"] == row["score_trend"] else 0.0,
            "late_assignment_count": int(row["late_assignment_count"]),
            "risk_probability": round(float(prob), 4),
            "at_risk": int(row["at_risk"]),
        })

    _cached_global_analytics = _sanitize_nans({
        "classifier_architecture": pred.model_architecture if pred else None,
        "model_name": pred.model_name if pred else None,
        "is_calibrated": pred.is_calibrated if pred else None,
        "selected_threshold": pred.threshold if pred else DEFAULT_THRESHOLD,
        "global_shap_importance": global_importance,
        "feature_correlations": correlations,
        "constellation_sample": constellation_points,
        "sample_size": int(len(sample_df)),
    })
    return _cached_global_analytics


@app.get("/interventions", tags=["Interventions"])
def get_interventions():
    """Returns persisted teacher intervention tracking records and status breakdown."""
    store = _load_or_seed_interventions()
    records = sorted(
        store.values(),
        key=lambda item: (item.get("status") == "Resolved", -(item.get("risk_probability") or 0.0)),
    )
    counts = {"Pending Review": 0, "Outreach Scheduled": 0, "Mentoring Active": 0, "Resolved": 0}
    for rec in records:
        st = rec.get("status", "Pending Review")
        if st in counts:
            counts[st] += 1
    return {
        "total": len(records),
        "status_summary": counts,
        "interventions": records,
    }


@app.post("/interventions", tags=["Interventions"])
def upsert_intervention(payload: InterventionUpsertRequest):
    """Creates or updates teacher follow-up status and advisory notes for a student."""
    store = _load_or_seed_interventions()
    sid = payload.student_id.strip().upper()
    existing = store.get(sid, {})

    record = {
        "student_id": sid,
        "assessment_period": payload.assessment_period or existing.get("assessment_period", 4),
        "risk_probability": payload.risk_probability if payload.risk_probability is not None else existing.get("risk_probability"),
        "priority": payload.priority,
        "recommended_support": payload.recommended_support or existing.get("recommended_support", "Teacher check-in recommended."),
        "status": payload.status,
        "teacher_notes": payload.teacher_notes.strip(),
        "source": "teacher_workflow",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    store[sid] = record
    _save_interventions(store)
    return {"saved": True, "record": record}


@app.get("/reports/students", tags=["Reports"])
def get_student_reports(
    period: Optional[int] = None,
    risk_tier: Optional[str] = None,
    limit: int = 100,
    format: str = "json",
):
    """
    Generates student risk reports combining real model risk probabilities, risk tiers,
    rule-based actions, and teacher intervention notes. Supports JSON and downloadable CSV.
    """
    if period is not None and period not in SUPPORTED_ASSESSMENT_PERIODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"period must be in {SUPPORTED_ASSESSMENT_PERIODS}, got {period}",
        )
    if risk_tier is not None and risk_tier not in ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="risk_tier must be LOW_RISK, MODERATE_RISK, or HIGH_RISK",
        )
    if format not in ("json", "csv"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="format must be 'json' or 'csv'")

    df = get_cohort_dataframe()
    pred = get_or_create_predictor()
    if df is None or pred is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Dataset or model unavailable.")

    target_period = period if period is not None else 4
    subset_df = df[df["assessment_period"] == target_period].copy()
    probs = pred.pipeline.predict_proba(enforce_feature_order(subset_df))[:, 1]
    subset_df["_risk_prob"] = probs

    interventions = _load_or_seed_interventions()
    records = []
    tier_counts = {"HIGH_RISK": 0, "MODERATE_RISK": 0, "LOW_RISK": 0}

    for _, row in subset_df.sort_values("_risk_prob", ascending=False).iterrows():
        prob = round(float(row["_risk_prob"]), 4)
        band = pred.determine_risk_band(prob)
        tier = band["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if risk_tier and tier != risk_tier:
            continue
        sid = str(row["student_id"])
        feat_dict = {col: row[col] for col in FEATURE_COLUMNS}
        top_plan = generate_support_recommendations(feat_dict, prob, threshold=pred.threshold)[0]
        iv = interventions.get(sid, {})
        records.append({
            "student_id": sid,
            "assessment_period": int(row["assessment_period"]),
            "attendance_pct": None if row["attendance_pct"] != row["attendance_pct"] else round(float(row["attendance_pct"]), 1),
            "assignment_completion_pct": None if row["assignment_completion_pct"] != row["assignment_completion_pct"] else round(float(row["assignment_completion_pct"]), 1),
            "late_assignment_count": int(row["late_assignment_count"]),
            "previous_score": round(float(row["previous_score"]), 1),
            "previous_score_2": None if row["previous_score_2"] != row["previous_score_2"] else round(float(row["previous_score_2"]), 1),
            "previous_score_avg": round(float(row["previous_score_avg"]), 1),
            "score_trend": round(float(row["score_trend"]), 1),
            "risk_probability": prob,
            "predicted_at_risk": int(prob >= pred.threshold),
            "risk_tier": tier,
            "risk_label": band["label"],
            "recommended_support_category": top_plan["category"],
            "intervention_status": iv.get("status", "Not Tracked"),
            "teacher_notes": iv.get("teacher_notes", ""),
        })

    # Period-level summary across entire cohort for Assessment-Period Reports
    all_probs = pred.pipeline.predict_proba(enforce_feature_order(df))[:, 1]
    df_with_prob = df.copy()
    df_with_prob["_prob"] = all_probs
    period_reports = []
    for p in sorted(df_with_prob["assessment_period"].unique()):
        p_sub = df_with_prob[df_with_prob["assessment_period"] == p]
        period_reports.append({
            "period": int(p),
            "student_count": int(len(p_sub)),
            "flagged_count": int((p_sub["_prob"] >= pred.threshold).sum()),
            "actual_at_risk_count": int(p_sub["at_risk"].sum()),
            "avg_attendance": round(float(p_sub["attendance_pct"].mean()), 1),
            "avg_completion": round(float(p_sub["assignment_completion_pct"].mean()), 1),
            "avg_score": round(float(p_sub["previous_score"].mean()), 1),
            "avg_risk_probability_pct": round(float(p_sub["_prob"].mean()) * 100.0, 1),
        })

    if format == "csv":
        export_rows = records[: max(1, min(2500, limit))]
        output = io.StringIO()
        fieldnames = [
            "student_id", "assessment_period", "attendance_pct", "assignment_completion_pct",
            "late_assignment_count", "previous_score", "previous_score_2", "previous_score_avg",
            "score_trend", "risk_probability", "predicted_at_risk", "risk_tier",
            "recommended_support_category", "intervention_status", "teacher_notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in export_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="eduguard_period_{target_period}_risk_report.csv"'
            },
        )

    capped = records[: max(1, min(500, int(limit)))]
    return _sanitize_nans({
        "period": target_period,
        "total_matching_records": len(records),
        "returned_records": len(capped),
        "applied_threshold": pred.threshold,
        "tier_distribution": tier_counts,
        "period_reports": period_reports,
        "records": capped,
    })


@app.get("/audit-logs", tags=["Governance"])
def get_audit_logs():
    """Returns governance records, feature exclusion registry, and verified validation metrics."""
    pred = get_or_create_predictor()
    meta = pred.metadata if pred else {}
    val_metrics = meta.get("validation_metrics", {})

    return {
        "audit_version": MODEL_VERSION,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "responsible_ai_policy": {
            "advisory_only": True,
            "automated_sanctions_prohibited": True,
            "human_in_the_loop_mandatory": True,
            "model_name": pred.model_name if pred else None,
            "champion_architecture": pred.model_architecture if pred else None,
            "is_calibrated": pred.is_calibrated if pred else None,
            "model_version": pred.version if pred else MODEL_VERSION,
        },
        "feature_exclusion_registry": [
            {"attribute": "Student Name / Identity", "status": "EXCLUDED", "rationale": "Non-generalizable, privacy protection"},
            {"attribute": "Gender / Sex", "status": "EXCLUDED", "rationale": "Protected characteristic, non-pedagogical"},
            {"attribute": "Race / Ethnicity", "status": "EXCLUDED", "rationale": "Protected characteristic, fairness compliance"},
            {"attribute": "Religion / Caste", "status": "EXCLUDED", "rationale": "Protected characteristic, non-bias guarantee"},
            {"attribute": "Disability Status", "status": "EXCLUDED", "rationale": "Requires individualized accommodations, not algorithmic risk"},
            {"attribute": "Socio-Economic Status / Zipcode", "status": "EXCLUDED", "rationale": "Eliminates systemic disparity bias"},
            {"attribute": "Next Assessment Outcome Score", "status": "EXCLUDED", "rationale": "Target derivation only; strict zero-leakage invariant"},
        ],
        "validation_ledger": {
            "validation_methodology": "Dual (Temporal + Student-Disjoint) with Validation-Split Sigmoid Calibration",
            "selected_threshold": meta.get("selected_threshold", pred.threshold if pred else DEFAULT_THRESHOLD),
            "threshold_rationale": meta.get("threshold_optimization_rationale"),
            "safety_net_recall": val_metrics.get("recall"),
            "educator_precision": val_metrics.get("precision"),
            "average_precision": val_metrics.get("average_precision"),
            "specificity": val_metrics.get("specificity"),
            "brier_score": val_metrics.get("brier_score"),
        },
    }
