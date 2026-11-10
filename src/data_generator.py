"""
EduGuard AI - Longitudinal Synthetic Data Generator
Generates realistic, reproducible synthetic educational snapshots for 2,500 students
across five assessment periods (10,000 historical prediction records).

Implements latent student traits, period difficulty effects, realistic outliers,
controlled missingness, and strict leakage prevention.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure EduGuard-AI project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    AT_RISK_THRESHOLD_SCORE,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUM_PERIODS,
    NUM_STUDENTS,
    OUTCOME_SCORE_COLUMN,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.feature_engineering import apply_feature_engineering

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.DataGenerator")


def generate_synthetic_dataset(
    num_students: int = NUM_STUDENTS,
    num_periods: int = NUM_PERIODS,
    random_seed: int = RANDOM_SEED,
    missing_rate: float = 0.015,
    outlier_rate: float = 0.03,
) -> pd.DataFrame:
    """
    Generate longitudinal synthetic student performance data with multi-period snapshots.

    Args:
        num_students: Total unique students to simulate (default: 2,500).
        num_periods: Total assessment cycles (default: 5, yielding 4 prediction snapshots per student).
        random_seed: Seed for reproducibility (default: 42).
        missing_rate: Fraction of random missing values to inject into engagement features.
        outlier_rate: Fraction of students subjected to realistic behavioral shocks.

    Returns:
        pd.DataFrame containing 10,000 snapshot rows with feature columns, target, and outcome score.
    """
    logger.info(
        f"Initiating synthetic data simulation: {num_students} students, "
        f"{num_periods} periods, seed={random_seed}"
    )
    rng = np.random.RandomState(random_seed)

    # 1. Synthesize Latent Student Traits (Independent student-level factors)
    # Aptitude / Academic Preparedness ~ Normal(53.5, 15.5) clipped to [12, 98]
    base_aptitude = np.clip(rng.normal(loc=53.5, scale=15.5, size=num_students), 12.0, 98.0)

    # Study Discipline & Conscientiousness ~ Beta-shaped distribution in [0.10, 0.98]
    base_discipline = np.clip(rng.beta(a=3.2, b=2.2, size=num_students), 0.10, 0.98)

    # Individual score volatility (susceptibility to test anxiety, fatigue) ~ Gamma(2.2, 2.2)
    individual_volatility = rng.gamma(shape=2.2, scale=2.2, size=num_students)

    # Diagnostic baseline entrance test score (Period 0 baseline score)
    diagnostic_score = np.clip(
        base_aptitude + rng.normal(0, 5.5, size=num_students) + (base_discipline - 0.5) * 10.0,
        5.0,
        100.0,
    )

    # Assessment period difficulty adjustments (midterm difficulty spike in periods 3 & 4)
    period_difficulty = {
        1: 0.0,   # Baseline foundational concepts
        2: 2.5,   # Progressive intermediate topics
        3: 6.0,   # Midterm comprehensive exam (difficulty surge)
        4: 4.0,   # Advanced specialized modules
        5: 3.0,   # Final integrative assessment
    }

    # Simulate scores for all students across periods 0 through 5
    # Period 0 is baseline entrance; periods 1..5 are curriculum assessments
    student_scores = np.zeros((num_students, num_periods + 1))
    student_scores[:, 0] = diagnostic_score

    for p in range(1, num_periods + 1):
        diff = period_difficulty.get(p, 2.0)
        # Score depends on student base aptitude, discipline, cumulative learning, minus test difficulty + noise
        noise = rng.normal(loc=0.0, scale=individual_volatility, size=num_students)
        period_score = (
            base_aptitude
            + (base_discipline - 0.5) * 16.0
            - diff
            + (p * 0.8)  # Gradual learning curve effect
            + noise
        )
        student_scores[:, p] = np.clip(period_score, 0.0, 100.0)

    # 2. Inject Realistic Academic Behavioral Archetypes & Outliers
    num_outliers = int(num_students * outlier_rate)
    outlier_indices = rng.choice(num_students, size=num_outliers, replace=False)

    # Archetype A: Burnout / Illness shock around period 3
    shock_sub = outlier_indices[: num_outliers // 3]
    student_scores[shock_sub, 3] = np.clip(student_scores[shock_sub, 3] - rng.uniform(20, 35, size=len(shock_sub)), 5.0, 100.0)

    # Archetype B: Crammer (High aptitude, poor attendance, decent scores)
    crammer_sub = outlier_indices[num_outliers // 3 : 2 * (num_outliers // 3)]
    base_discipline[crammer_sub] = np.clip(base_discipline[crammer_sub] - 0.40, 0.10, 0.45)
    base_aptitude[crammer_sub] = np.clip(base_aptitude[crammer_sub] + 12.0, 50.0, 95.0)

    # Archetype C: Diligent but struggling (High attendance/completion, low exam scores)
    struggler_sub = outlier_indices[2 * (num_outliers // 3) :]
    base_discipline[struggler_sub] = np.clip(base_discipline[struggler_sub] + 0.35, 0.85, 0.99)
    base_aptitude[struggler_sub] = np.clip(base_aptitude[struggler_sub] - 22.0, 10.0, 38.0)

    # 3. Construct Longitudinal Snapshot Dataset
    # For prediction at period t in {1, 2, 3, 4}:
    # Observable features prior to assessment t+1:
    # - attendance_pct (measured during period t)
    # - assignment_completion_pct (measured during period t)
    # - late_assignment_count (measured during period t)
    # - previous_score: assessment score from period t
    # - previous_score_2: assessment score from period t-1
    # - previous_score_avg: rolling 2-assessment mean (previous_score, previous_score_2),
    #   recomputed authoritatively post-missingness via apply_feature_engineering()
    # - score_trend: previous_score - previous_score_2 (0.0 when previous_score_2 is NaN)
    # - assessment_period: t
    # Target: at_risk = 1 if score at period t+1 < 40 else 0
    records = []

    for s_idx in range(num_students):
        s_id = f"STU_{s_idx + 1:05d}"
        disc = base_discipline[s_idx]

        for p in range(1, num_periods):  # Periods 1, 2, 3, 4 predict periods 2, 3, 4, 5
            # Attendance in period p: centered around student discipline + periodic fluctuation
            att_noise = rng.normal(loc=0.0, scale=6.5)
            attendance_pct = np.clip(disc * 100.0 + att_noise, 5.0, 100.0)

            # Assignment completion: strongly correlated with discipline and attendance
            assign_noise = rng.normal(loc=0.0, scale=7.0)
            assignment_completion_pct = np.clip(disc * 92.0 + (attendance_pct * 0.08) + assign_noise, 0.0, 100.0)

            # Late assignment count: Poisson distributed, inverse to discipline
            late_lambda = max(0.2, (1.0 - disc) * 5.0 + rng.uniform(-0.5, 0.5))
            late_assignment_count = int(rng.poisson(lam=late_lambda))

            # Scores
            curr_score = student_scores[s_idx, p]
            prev_score_2 = student_scores[s_idx, p - 1]
            prev_score_avg = (curr_score + prev_score_2) / 2.0
            score_trend = curr_score - prev_score_2

            # Future Target (Period p+1)
            next_score = student_scores[s_idx, p + 1]
            at_risk = 1 if next_score < AT_RISK_THRESHOLD_SCORE else 0

            records.append({
                ID_COLUMN: s_id,
                "assessment_period": int(p),
                "attendance_pct": round(float(attendance_pct), 2),
                "assignment_completion_pct": round(float(assignment_completion_pct), 2),
                "late_assignment_count": late_assignment_count,
                "previous_score": round(float(curr_score), 2),
                "previous_score_2": round(float(prev_score_2), 2),
                "previous_score_avg": round(float(prev_score_avg), 2),
                "score_trend": round(float(score_trend), 2),
                OUTCOME_SCORE_COLUMN: round(float(next_score), 2),
                TARGET_COLUMN: at_risk,
            })

    df = pd.DataFrame(records)

    # 4. Inject Realistic Missing Values (MCAR / MAR) into engagement features
    # Missing attendance or completion can occur when a student transfers sections or system lags
    if missing_rate > 0:
        mask_att = rng.rand(len(df)) < missing_rate
        df.loc[mask_att, "attendance_pct"] = np.nan

        mask_comp = rng.rand(len(df)) < (missing_rate * 1.2)
        df.loc[mask_comp, "assignment_completion_pct"] = np.nan

        # Small missingness in previous_score_2 for period 1 (e.g. transfer students without record)
        mask_prev2 = (df["assessment_period"] == 1) & (rng.rand(len(df)) < missing_rate)
        df.loc[mask_prev2, "previous_score_2"] = np.nan

    # 5. Apply canonical derived feature engineering post-missingness (single source of truth)
    df = apply_feature_engineering(df)

    logger.info(
        f"Synthetic generation complete: {len(df)} records created across {df[ID_COLUMN].nunique()} students."
    )
    logger.info(
        f"Target distribution ('{TARGET_COLUMN}'): "
        f"{df[TARGET_COLUMN].value_counts(normalize=True).to_dict()}"
    )

    return df


def save_synthetic_dataset(
    output_path: Optional[Path] = None,
    random_seed: int = RANDOM_SEED,
) -> Path:
    """
    Generate and save the synthetic dataset to disk.

    Args:
        output_path: Destination CSV path. Defaults to RAW_DATA_PATH.
        random_seed: Reproducible seed.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        output_path = RAW_DATA_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(random_seed=random_seed)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset saved successfully to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic educational performance data.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for generation.")
    parser.add_argument("--output", type=str, default=str(RAW_DATA_PATH), help="Path to save output CSV.")
    args = parser.parse_args()

    save_synthetic_dataset(output_path=Path(args.output), random_seed=args.seed)
