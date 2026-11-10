"""
EduGuard AI - Exploratory Data Analysis & Visualization Suite
Generates high-resolution publication-quality figures for the technical report
and dashboard, adhering to professional dark navy / enterprise styling.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe matplotlib cache directory
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EduGuard.EDA")

from src.config import (
    FIGURES_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    OUTCOME_SCORE_COLUMN,
)

# Apply consistent enterprise styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PRIMARY_COLOR = "#6366F1"   # Indigo / Violet
SECONDARY_COLOR = "#06B6D4" # Cyan
ACCENT_WARN = "#EF4444"     # Coral Red
ACCENT_SAFE = "#10B981"     # Emerald Green
DARK_BG = "#0F172A"         # Slate Navy


def load_dataset() -> pd.DataFrame:
    """Load raw dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def plot_target_distribution(df: pd.DataFrame) -> Path:
    """1. Target Class Distribution (at_risk vs not at_risk)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Not At-Risk (>= 40)", "At-Risk (< 40)"]
    colors = [ACCENT_SAFE, ACCENT_WARN]

    bars = ax.bar(labels, counts.values, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 120, f"{h:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, len(df) * 0.95)
    ax.set_ylabel("Snapshot Record Count")
    ax.set_title("Target Class Distribution (Next Assessment Risk Label)")
    fig.tight_layout()

    out_path = FIGURES_DIR / "01_target_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_attendance_distribution(df: pd.DataFrame) -> Path:
    """2. Attendance Distribution split by Risk Label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="attendance_pct",
        hue=TARGET_COLUMN,
        palette={0: ACCENT_SAFE, 1: ACCENT_WARN},
        kde=True,
        element="step",
        bins=30,
        ax=ax,
        alpha=0.45,
    )
    ax.set_title("Student Classroom Attendance Distribution by Risk Category")
    ax.set_xlabel("Attendance Percentage (%)")
    ax.set_ylabel("Count")
    ax.legend(title="Risk Status", labels=["At-Risk (< 40)", "Not At-Risk (>= 40)"])
    fig.tight_layout()

    out_path = FIGURES_DIR / "02_attendance_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_assignment_completion_distribution(df: pd.DataFrame) -> Path:
    """3. Assignment Completion Rate Distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=df[df[TARGET_COLUMN] == 0],
        x="assignment_completion_pct",
        label="Not At-Risk",
        color=ACCENT_SAFE,
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    sns.kdeplot(
        data=df[df[TARGET_COLUMN] == 1],
        x="assignment_completion_pct",
        label="At-Risk",
        color=ACCENT_WARN,
        fill=True,
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("Assignment Completion Density Comparison")
    ax.set_xlabel("Assignment Completion Percentage (%)")
    ax.set_ylabel("Kernel Density Estimate")
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / "03_assignment_completion_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_previous_score_distribution(df: pd.DataFrame) -> Path:
    """4. Previous Assessment Score Distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="previous_score",
        hue=TARGET_COLUMN,
        palette={0: ACCENT_SAFE, 1: ACCENT_WARN},
        legend=False,
        ax=ax,
        width=0.4,
    )
    ax.axhline(40, color="#EF4444", linestyle="--", linewidth=1.5, label="Risk Threshold (40 Marks)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not At-Risk (>= 40)", "At-Risk (< 40)"])
    ax.set_title("Distribution of Current Period Scores Across Future Risk Groups")
    ax.set_xlabel("Next Assessment Risk Status")
    ax.set_ylabel("Previous Assessment Score (0-100)")
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / "04_previous_score_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_score_trends_over_periods(df: pd.DataFrame) -> Path:
    """5. Score Trends Over Assessment Periods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    period_stats = df.groupby(["assessment_period", TARGET_COLUMN])["previous_score"].mean().unstack()

    x = np.arange(len(period_stats.index))
    width = 0.35

    ax.bar(x - width/2, period_stats[0], width, label="Not At-Risk Cohort Avg", color=PRIMARY_COLOR, alpha=0.85)
    ax.bar(x + width/2, period_stats[1], width, label="At-Risk Cohort Avg", color=ACCENT_WARN, alpha=0.85)

    ax.axhline(40, color="gray", linestyle=":", label="Risk Boundary")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Period {p}" for p in period_stats.index])
    ax.set_title("Longitudinal Average Performance Trajectory by Assessment Period")
    ax.set_xlabel("Assessment Snapshot Cycle")
    ax.set_ylabel("Average Assessment Score")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / "05_score_trends_over_periods.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """6. Correlation Heatmap across Pre-Assessment Features and Risk Target."""
    fig, ax = plt.subplots(figsize=(9, 7))
    cols_to_corr = FEATURE_COLUMNS + [TARGET_COLUMN]
    corr_matrix = df[cols_to_corr].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature & Target Pearson Correlation Matrix (Strict Pre-Assessment Features)")
    fig.tight_layout()

    out_path = FIGURES_DIR / "06_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_missing_values_analysis(df: pd.DataFrame) -> Path:
    """7. Missing Value Visualization."""
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_pct = (df[FEATURE_COLUMNS].isna().mean() * 100).sort_values(ascending=False)

    colors = [ACCENT_WARN if v > 1.0 else SECONDARY_COLOR for v in missing_pct.values]
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, height=0.55)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%",
                va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(missing_pct.values) * 1.35 if max(missing_pct.values) > 0 else 5)
    ax.set_title("Missing Value Percentage Across Modeled Features (MCAR/MAR)")
    ax.set_xlabel("Missing Percentage (%)")
    fig.tight_layout()

    out_path = FIGURES_DIR / "07_missing_values_analysis.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_outlier_analysis(df: pd.DataFrame) -> Path:
    """8. Outlier Analysis (Late Assignments vs Score by Risk)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["late_assignment_count"],
        df["previous_score"],
        c=df[TARGET_COLUMN],
        cmap="coolwarm",
        alpha=0.35,
        edgecolors="none",
        s=30,
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Not At-Risk", "At-Risk"])

    ax.axhline(40, color="red", linestyle="--", alpha=0.7, label="Passing Boundary (40)")
    ax.set_title("Bivariate Outlier & Cluster Analysis: Late Submissions vs Score")
    ax.set_xlabel("Late Assignment Count")
    ax.set_ylabel("Previous Assessment Score")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = FIGURES_DIR / "08_outlier_analysis.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_figures():
    """Generate and save all 8 publication-grade figures."""
    logger.info("Generating EDA figures...")
    df = load_dataset()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_target_distribution(df),
        plot_attendance_distribution(df),
        plot_assignment_completion_distribution(df),
        plot_previous_score_distribution(df),
        plot_score_trends_over_periods(df),
        plot_correlation_heatmap(df),
        plot_missing_values_analysis(df),
        plot_outlier_analysis(df),
    ]

    for p in paths:
        logger.info(f"Generated figure: {p.name}")
    logger.info("All 8 figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
