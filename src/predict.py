"""
EduGuard AI - Production Prediction & Advisory Support Engine
Loads persistent calibrated pipeline and model metadata to generate real-time inferences,
calibrated risk probabilities, unified risk bands, SHAP feature signals, and clearly
labeled rule-based teacher support recommendations.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    DISCLAIMER_TEXT,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_VERSION,
    RISK_CATEGORIES,
)
from src.explain import EduGuardExplainer
from src.feature_engineering import enforce_feature_order, prepare_inference_features
from src.preprocessing import unwrap_base_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.Predict")


def generate_support_recommendations(
    features: Dict[str, Any],
    risk_prob: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, str]]:
    """
    Generates actionable, non-punitive rule-based teacher support guidance from pre-assessment inputs.

    Note: These recommendations are deterministic rule-based heuristics conditioned on observed
    school indicators and the calibrated threshold—separate from SHAP statistical attributions.
    """
    plans: List[Dict[str, str]] = []

    att = features.get("attendance_pct")
    comp = features.get("assignment_completion_pct")
    late = features.get("late_assignment_count", 0)
    trend = features.get("score_trend", 0.0)
    prev_score = features.get("previous_score", 50.0)

    # 1. Attendance Signal
    if att is not None and not np.isnan(att) and att < 75.0:
        plans.append({
            "category": "Attendance & Engagement Check-in",
            "priority": "High" if att < 60.0 else "Medium",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Classroom attendance is at {att:.1f}%. Recommend an informal, supportive teacher check-in "
                f"to understand barriers (e.g., transit, scheduling, or health) and share missed lesson notes."
            ),
        })

    # 2. Assignment Completion Signal
    if comp is not None and not np.isnan(comp) and comp < 70.0:
        plans.append({
            "category": "Assignment Workflow Support",
            "priority": "High" if comp < 50.0 else "Medium",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Assignment completion rate is {comp:.1f}%. Suggest a brief teacher review of coursework pacing, "
                f"task clarity, and access to school study hall or guided practice hours."
            ),
        })

    # 3. Late Submissions Signal
    if late is not None and late >= 2:
        plans.append({
            "category": "Time Management Guidance",
            "priority": "Medium" if late >= 3 else "Low",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Recorded {int(late)} late assignment submissions. Offer milestone reminders and check whether "
                f"chunked deadlines or planner support would assist the student."
            ),
        })

    # 4. Declining Score Momentum
    if trend is not None and trend <= -8.0:
        plans.append({
            "category": "Concept Review & Confidence Rebuilding",
            "priority": "High",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Recent assessment trajectory dropped by {abs(trend):.1f} marks. Recommend targeted teacher "
                f"review of recent topics and supplementary practice problems before the next exam."
            ),
        })

    # 5. Low Absolute Score Signal
    if prev_score < 42.0:
        plans.append({
            "category": "Foundational Mastery Support",
            "priority": "High",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Prior assessment score was {prev_score:.1f}/100. Recommend pairing the student with school "
                f"peer tutoring or dedicated teacher extra-help sessions on prerequisite skills."
            ),
        })

    # 6. Holistic Multi-Signal Advisory calibrated against threshold
    if risk_prob >= threshold and len(plans) >= 2:
        plans.insert(0, {
            "category": "Holistic Academic Coordinator Check-in",
            "priority": "Urgent",
            "source": "rule_based_advisory",
            "advisory_action": (
                f"Calibrated risk ({risk_prob*100:.1f}%) exceeds the operational alert threshold ({threshold*100:.0f}%) "
                f"with multiple academic signals. Recommend a collaborative teacher-student mentoring plan."
            ),
        })

    if not plans:
        plans.append({
            "category": "Positive Reinforcement",
            "priority": "Routine",
            "source": "rule_based_advisory",
            "advisory_action": "Student classroom engagement trajectory is stable. Continue standard positive reinforcement.",
        })

    return plans


class EduGuardPredictor:
    """Production inference engine loading persistent pipeline and metadata."""

    def __init__(
        self,
        model_artifact_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        enable_shap: bool = True,
    ):
        self.model_artifact_path = model_artifact_path or MODEL_ARTIFACT_PATH
        self.metadata_path = metadata_path or METADATA_PATH

        if not self.model_artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_artifact_path}")

        logger.info(f"Loading persistent model pipeline from {self.model_artifact_path}")
        self.pipeline = joblib.load(self.model_artifact_path)
        self.base_pipeline, self.is_calibrated = unwrap_base_pipeline(self.pipeline)
        self.model_architecture = self.base_pipeline.named_steps["classifier"].__class__.__name__

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.threshold = float(self.metadata.get("selected_threshold", DEFAULT_THRESHOLD))
            self.version = self.metadata.get("model_version", MODEL_VERSION)
            self.model_name = self.metadata.get("model_name", self.model_architecture)
        else:
            logger.warning(f"Metadata file not found at {self.metadata_path}. Using default threshold.")
            self.threshold = DEFAULT_THRESHOLD
            self.version = MODEL_VERSION
            self.model_name = self.model_architecture
            self.metadata = {}

        self.enable_shap = enable_shap
        self._explainer = None

    @property
    def explainer(self) -> Optional[EduGuardExplainer]:
        """Lazy loads SHAP explainer on demand to optimize startup latency."""
        if self._explainer is None and self.enable_shap:
            try:
                logger.info("Initializing lazy SHAP explainer...")
                self._explainer = EduGuardExplainer(
                    model_pipeline_path=self.model_artifact_path,
                    threshold=self.threshold,
                )
            except Exception as exc:
                logger.warning(f"Could not initialize SHAP explainer: {exc}. Running without SHAP.")
        return self._explainer

    def determine_risk_band(self, probability: float) -> Dict[str, str]:
        """Maps calibrated probability to unified risk tiers consistent with RISK_CATEGORIES."""
        if probability < (self.threshold * 0.70):
            cfg = RISK_CATEGORIES["LOW_RISK"]
            return {
                "tier": "LOW_RISK",
                "label": cfg["label"],
                "color": cfg["color"],
                "action": "Continue Standard Classroom Support",
            }
        elif probability < self.threshold:
            cfg = RISK_CATEGORIES["MODERATE_RISK"]
            return {
                "tier": "MODERATE_RISK",
                "label": cfg["label"],
                "color": cfg["color"],
                "action": "Light Teacher Check-in Advised",
            }
        else:
            cfg = RISK_CATEGORIES["HIGH_RISK"]
            return {
                "tier": "HIGH_RISK",
                "label": cfg["label"],
                "color": cfg["color"],
                "action": "Proactive Academic Support Recommended",
            }

    def predict_single(self, payload: Dict[str, Any], include_shap: bool = True) -> Dict[str, Any]:
        """
        Executes real-time prediction using authoritative derived-feature calculation.
        """
        validated = prepare_inference_features(payload, recompute_derived=True)
        df_single = enforce_feature_order(validated)

        probs = self.pipeline.predict_proba(df_single)[0]
        risk_probability = float(np.clip(probs[1], 0.0, 1.0))
        prediction_binary = int(risk_probability >= self.threshold)
        risk_band = self.determine_risk_band(risk_probability)

        explanation_data = None
        if include_shap and self.explainer is not None:
            try:
                explanation_data = self.explainer.explain_instance(validated)
            except Exception as exc:
                logger.warning(f"SHAP explanation failed safely during predict_single: {exc}")
                explanation_data = {
                    "shap_available": False,
                    "explanation_space": "unavailable",
                    "risk_probability": round(risk_probability, 4),
                    "expected_base_value": round(self.threshold, 4),
                    "feature_contributions": [],
                    "top_risk_drivers": [],
                    "top_protective_factors": [],
                    "plain_language_explanation": (
                        "Calibrated probability prediction completed, while detailed SHAP decomposition "
                        "is temporarily unavailable for this snapshot."
                    ),
                    "explanation_error": str(exc),
                    "disclaimer": DISCLAIMER_TEXT,
                }

        support_plan = generate_support_recommendations(
            validated,
            risk_probability,
            threshold=self.threshold,
        )

        return {
            "prediction": prediction_binary,
            "prediction_label": "At-Risk (< 40)" if prediction_binary == 1 else "Not At-Risk (>= 40)",
            "risk_probability": round(risk_probability, 4),
            "applied_threshold": round(self.threshold, 3),
            "risk_band": risk_band,
            "input_features": validated,
            "explanation": explanation_data,
            "support_plan": support_plan,
            "model_name": self.model_name,
            "model_version": self.version,
            "is_calibrated": self.is_calibrated,
            "human_review_disclaimer": DISCLAIMER_TEXT,
        }

    def predict_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes genuine vectorized batch probability inference + individual advisory outputs."""
        if not payloads:
            return []

        validated_list = [prepare_inference_features(p, recompute_derived=True) for p in payloads]
        df_batch = enforce_feature_order(pd.DataFrame(validated_list))
        probs = self.pipeline.predict_proba(df_batch)[:, 1]

        results = []
        for validated, raw_prob in zip(validated_list, probs):
            risk_probability = float(np.clip(raw_prob, 0.0, 1.0))
            prediction_binary = int(risk_probability >= self.threshold)
            results.append({
                "prediction": prediction_binary,
                "prediction_label": "At-Risk (< 40)" if prediction_binary == 1 else "Not At-Risk (>= 40)",
                "risk_probability": round(risk_probability, 4),
                "applied_threshold": round(self.threshold, 3),
                "risk_band": self.determine_risk_band(risk_probability),
                "input_features": validated,
                "explanation": None,
                "support_plan": generate_support_recommendations(
                    validated,
                    risk_probability,
                    threshold=self.threshold,
                ),
                "model_name": self.model_name,
                "model_version": self.version,
                "is_calibrated": self.is_calibrated,
                "human_review_disclaimer": DISCLAIMER_TEXT,
            })
        return results


if __name__ == "__main__":
    predictor = EduGuardPredictor()
    test_input = {
        "attendance_pct": 68.0,
        "assignment_completion_pct": 52.0,
        "late_assignment_count": 3,
        "previous_score": 36.0,
        "previous_score_2": 42.0,
        "assessment_period": 2,
    }
    result = predictor.predict_single(test_input)
    print("\n--- EduGuard Prediction Engine Output ---")
    print(f"Model: {result['model_name']} (Calibrated={result['is_calibrated']}, v{result['model_version']})")
    print(f"Prediction: {result['prediction_label']} (Risk Prob: {result['risk_probability']*100:.1f}%)")
    print(f"Risk Band: {result['risk_band']['label']} | Action: {result['risk_band']['action']}")
    print(f"Threshold Applied: {result['applied_threshold']}")
    for sp in result["support_plan"]:
        print(f"  [{sp['priority'].upper()} | {sp['source']}] {sp['category']}: {sp['advisory_action']}")
