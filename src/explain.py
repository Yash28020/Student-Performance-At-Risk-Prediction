"""
EduGuard AI - Explainable AI (SHAP) Engine
Provides global and local model interpretability across tree ensembles and calibrated
pipelines, translating statistical SHAP attributions into transparent, non-causal
teacher guidance.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib cache protection
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    DEFAULT_THRESHOLD,
    DISCLAIMER_TEXT,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
)
from src.feature_engineering import enforce_feature_order, prepare_inference_features
from src.preprocessing import unwrap_base_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.Explain")


class EduGuardExplainer:
    """Production SHAP explanation engine supporting calibrated & uncalibrated classifiers."""

    def __init__(
        self,
        model_pipeline_path: Optional[Path] = None,
        background_df: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ):
        self.model_pipeline_path = model_pipeline_path or MODEL_ARTIFACT_PATH
        if not self.model_pipeline_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_pipeline_path}")

        logger.info(f"Loading pipeline artifact for SHAP explainer: {self.model_pipeline_path}")
        self.pipeline = joblib.load(self.model_pipeline_path)
        self.base_pipeline, self.is_calibrated = unwrap_base_pipeline(self.pipeline)
        self.preprocessor = self.base_pipeline.named_steps["preprocessor"]
        self.classifier = self.base_pipeline.named_steps["classifier"]

        # Load calibrated threshold dynamically
        if threshold is not None:
            self.threshold = float(threshold)
        elif METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.threshold = float(meta.get("selected_threshold", DEFAULT_THRESHOLD))
        else:
            self.threshold = DEFAULT_THRESHOLD

        self._background_trans = self._prepare_background_sample(background_df)
        self.explainer, self.explanation_space = self._build_shap_explainer()

    def _prepare_background_sample(self, background_df: Optional[pd.DataFrame]) -> np.ndarray:
        """Prepares clean numeric background array in transformed feature space."""
        if background_df is not None and len(background_df) > 0:
            src_df = background_df[FEATURE_COLUMNS].iloc[:150].copy()
            return np.asarray(self.preprocessor.transform(src_df), dtype=float)

        if RAW_DATA_PATH.exists():
            raw = pd.read_csv(RAW_DATA_PATH)
            sample_df = raw[FEATURE_COLUMNS].sample(n=min(150, len(raw)), random_state=RANDOM_SEED)
            return np.asarray(self.preprocessor.transform(sample_df), dtype=float)

        # Synthetic neutral fallback if raw CSV is absent in isolated test contexts
        return np.zeros((10, len(FEATURE_COLUMNS)), dtype=float)

    def _build_shap_explainer(self) -> Tuple[Any, str]:
        """Initializes appropriate SHAP explainer for the unwrapped base classifier."""
        clf_name = self.classifier.__class__.__name__
        suffix = "pre_calibration_probability" if self.is_calibrated else "predicted_probability"

        if "XGB" in clf_name:
            explainer = shap.TreeExplainer(
                self.classifier,
                data=self._background_trans,
                model_output="probability",
            )
            return explainer, suffix

        if "Forest" in clf_name or "Tree" in clf_name or "ExtraTrees" in clf_name:
            explainer = shap.TreeExplainer(self.classifier)
            return explainer, suffix

        if "LogisticRegression" in clf_name or "Linear" in clf_name:
            explainer = shap.LinearExplainer(self.classifier, self._background_trans)
            return explainer, ("pre_calibration_log_odds" if self.is_calibrated else "log_odds")

        explainer = shap.Explainer(self.classifier.predict_proba, self._background_trans)
        return explainer, suffix

    @staticmethod
    def _extract_positive_class_arrays(
        shap_raw: Any,
        expected_value: Any,
    ) -> Tuple[np.ndarray, float]:
        """Normalizes SHAP outputs across single/batch calls and binary/multiclass formats."""
        if hasattr(shap_raw, "values"):
            arr = np.asarray(shap_raw.values, dtype=float)
            base_arr = np.asarray(shap_raw.base_values, dtype=float)
            if arr.ndim == 3:
                sv = arr[:, :, 1]
                base = float(np.ravel(base_arr)[1] if base_arr.size >= 2 else np.ravel(base_arr)[0])
            else:
                sv = arr
                base = float(np.ravel(base_arr)[0])
            return sv, base

        if isinstance(shap_raw, list):
            sv = np.asarray(shap_raw[1], dtype=float)
            base = float(expected_value[1])
            return sv, base

        arr = np.asarray(shap_raw, dtype=float)
        if arr.ndim == 3:
            sv = arr[:, :, 1]
            base = float(expected_value[1])
            return sv, base

        if np.ndim(expected_value) > 0:
            exp_flat = np.asarray(expected_value, dtype=float).ravel()
            base = float(exp_flat[1] if len(exp_flat) >= 2 else exp_flat[0])
        else:
            base = float(expected_value)
        return arr, base

    def explain_instance(self, feature_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates local SHAP explanation for a single student snapshot.

        Uses canonical derived-feature resolution (`prepare_inference_features`)
        so direct calls and predictor calls behave identically.
        """
        validated = prepare_inference_features(feature_payload, recompute_derived=True)
        df_single = enforce_feature_order(validated)

        X_trans = np.asarray(self.preprocessor.transform(df_single), dtype=float)
        calibrated_risk_prob = float(self.pipeline.predict_proba(df_single)[0, 1])
        base_model_prob = float(self.base_pipeline.predict_proba(df_single)[0, 1])

        try:
            shap_raw = self.explainer.shap_values(X_trans)
            sv_matrix, base_val = self._extract_positive_class_arrays(shap_raw, self.explainer.expected_value)
            sv = np.asarray(sv_matrix[0], dtype=float)
            shap_available = True
            error_msg = None
        except Exception as exc:
            logger.error(f"SHAP local explanation failed cleanly: {exc}")
            sv = np.zeros(len(FEATURE_COLUMNS), dtype=float)
            base_val = base_model_prob
            shap_available = False
            error_msg = str(exc)

        contributions = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = df_single[col].iloc[0]
            val_out = None if pd.isna(val) else round(float(val), 2)
            contrib = float(sv[i])
            direction = "INCREASES_RISK" if contrib > 0 else "REDUCES_RISK"
            contributions.append({
                "feature": col,
                "value": val_out,
                "shap_value": round(contrib, 4),
                "absolute_impact": round(abs(contrib), 4),
                "direction": direction,
            })

        contributions_sorted = sorted(contributions, key=lambda c: c["absolute_impact"], reverse=True)
        risk_drivers = [c for c in contributions_sorted if c["direction"] == "INCREASES_RISK"]
        protective_factors = [c for c in contributions_sorted if c["direction"] == "REDUCES_RISK"]

        narrative = self._generate_plain_language_narrative(
            risk_prob=calibrated_risk_prob,
            risk_drivers=risk_drivers,
            protective_factors=protective_factors,
            shap_available=shap_available,
        )

        calibration_note = (
            "SHAP attributions explain the underlying ensemble model's pre-calibration risk probability "
            "prior to validation sigmoid calibration; the reported `risk_probability` reflects the calibrated forecast."
            if self.is_calibrated
            else "SHAP attributions directly decompose the uncalibrated ensemble probability."
        )

        result: Dict[str, Any] = {
            "shap_available": shap_available,
            "explanation_space": self.explanation_space,
            "risk_probability": round(calibrated_risk_prob, 4),
            "uncalibrated_model_probability": round(base_model_prob, 4),
            "expected_base_value": round(base_val, 4),
            "feature_contributions": contributions_sorted,
            "top_risk_drivers": risk_drivers[:3],
            "top_protective_factors": protective_factors[:3],
            "plain_language_explanation": narrative,
            "calibration_attribution_note": calibration_note,
            "disclaimer": (
                "SHAP attributions measure statistical predictive associations within this synthetic cohort model, "
                "NOT deterministic causes of student performance. Always combine insights with direct teacher review."
            ),
        }
        if error_msg is not None:
            result["explanation_error"] = error_msg
        return result

    def _generate_plain_language_narrative(
        self,
        risk_prob: float,
        risk_drivers: List[Dict[str, Any]],
        protective_factors: List[Dict[str, Any]],
        shap_available: bool = True,
    ) -> str:
        """Constructs an accessible non-causal summary aligned with calibrated thresholds."""
        pct = round(risk_prob * 100.0, 1)
        if risk_prob >= self.threshold:
            urgency = "high advisory priority"
        elif risk_prob >= self.threshold * 0.70:
            urgency = "moderate advisory priority"
        else:
            urgency = "low advisory priority"

        paragraphs = [
            f"Based on pre-assessment school records, the calibrated model estimates a {pct}% probability "
            f"({urgency}, alert threshold {self.threshold * 100:.0f}%) of scoring below 40/100 on the next assessment."
        ]

        if not shap_available:
            paragraphs.append(
                "Local SHAP feature breakdown was temporarily unavailable for this record; "
                "teachers should review attendance, assignment completion, and recent assessment scores directly."
            )
            return " ".join(paragraphs)

        if risk_drivers:
            top_drivers = [
                f"{d['feature'].replace('_', ' ').title()} ({d['value'] if d['value'] is not None else 'Missing'})"
                for d in risk_drivers[:2]
            ]
            paragraphs.append(
                f"Statistical patterns most strongly associated with elevated risk include: {' and '.join(top_drivers)}."
            )

        if protective_factors:
            top_safe = [
                f"{s['feature'].replace('_', ' ').title()} ({s['value'] if s['value'] is not None else 'Missing'})"
                for s in protective_factors[:2]
            ]
            paragraphs.append(
                f"Protective academic signals associated with lower risk include: {' and '.join(top_safe)}."
            )

        paragraphs.append(
            "Next Step: Treat this summary as a teacher review aid—check recent coursework topics and "
            "consider a supportive one-on-one conversation with the student."
        )
        return " ".join(paragraphs)

    def generate_global_explanations(
        self,
        sample_df: Optional[pd.DataFrame] = None,
        sample_size: int = 500,
    ) -> Dict[str, Any]:
        """Computes global SHAP summary statistics and exports high-resolution plots."""
        if sample_df is None:
            raw = pd.read_csv(RAW_DATA_PATH)
            sample_df = raw.sample(n=min(sample_size, len(raw)), random_state=RANDOM_SEED)

        X_df = sample_df[FEATURE_COLUMNS].copy()
        X_trans = np.asarray(self.preprocessor.transform(X_df), dtype=float)

        shap_raw = self.explainer.shap_values(X_trans)
        sv, _ = self._extract_positive_class_arrays(shap_raw, self.explainer.expected_value)

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        global_importance = sorted(
            [
                {"feature": col, "mean_absolute_shap": round(float(val), 4)}
                for col, val in zip(FEATURE_COLUMNS, mean_abs_shap)
            ],
            key=lambda x: x["mean_absolute_shap"],
            reverse=True,
        )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        pretty_names = [c.replace("_", " ").title() for c in FEATURE_COLUMNS]

        # Plot 1: SHAP Beeswarm Summary Plot
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            sv,
            X_trans,
            feature_names=pretty_names,
            show=False,
        )
        plt.title(
            f"Global SHAP Feature Impact ({self.classifier.__class__.__name__}: Scoring < 40 Risk)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        summary_plot_path = FIGURES_DIR / "09_shap_summary.png"
        plt.savefig(summary_plot_path, dpi=300)
        plt.close("all")

        # Plot 2: SHAP Bar Plot of Feature Importance
        fig, ax = plt.subplots(figsize=(8, 5))
        features_rev = [item["feature"].replace("_", " ").title() for item in reversed(global_importance)]
        vals_rev = [item["mean_absolute_shap"] for item in reversed(global_importance)]
        ax.barh(features_rev, vals_rev, color="#6366F1", height=0.55)
        ax.set_title(f"Global Mean |SHAP| Importance ({self.classifier.__class__.__name__})", fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Risk Probability)")
        plt.tight_layout()
        bar_plot_path = FIGURES_DIR / "10_shap_bar.png"
        plt.savefig(bar_plot_path, dpi=300)
        plt.close(fig)

        logger.info(f"Global SHAP plots saved: {summary_plot_path.name}, {bar_plot_path.name}")
        return {
            "classifier_architecture": self.classifier.__class__.__name__,
            "explanation_space": self.explanation_space,
            "sample_size": int(len(sample_df)),
            "global_feature_importance": global_importance,
            "summary_plot_path": str(summary_plot_path),
            "bar_plot_path": str(bar_plot_path),
        }


if __name__ == "__main__":
    explainer = EduGuardExplainer()
    logger.info("Computing global SHAP visualizations...")
    global_res = explainer.generate_global_explanations()
    print("\n--- Global Feature Importance Ranking (Mean |SHAP|) ---")
    for item in global_res["global_feature_importance"]:
        print(f"  {item['feature']:<28}: {item['mean_absolute_shap']}")

    sample_student = {
        "attendance_pct": 65.0,
        "assignment_completion_pct": 58.0,
        "late_assignment_count": 3,
        "previous_score": 38.0,
        "previous_score_2": 45.0,
        "assessment_period": 3,
    }
    local_exp = explainer.explain_instance(sample_student)
    print("\n--- Sample Local Student SHAP Explanation ---")
    print(f"Predicted Calibrated Risk Probability: {local_exp['risk_probability'] * 100:.1f}%")
    print(f"Narrative: {local_exp['plain_language_explanation']}")
    for d in local_exp["top_risk_drivers"]:
        print(f"  [+] {d['feature']}: {d['value']} (SHAP: +{d['shap_value']})")
    for s in local_exp["top_protective_factors"]:
        print(f"  [-] {s['feature']}: {s['value']} (SHAP: {s['shap_value']})")
