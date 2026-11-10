# EduGuard AI — Data Documentation

This directory houses raw and processed synthetic school student performance records used strictly for offline modeling, validation, explainability, and dashboard workflows.

## Data Governance & Ethics
- **Synthetic Origin:** All records are 100% synthetic (2,500 simulated students across 5 academic periods -> 10,000 pre-assessment prediction snapshots, `seed=42`). No real student records or PII are present, eliminating FERPA/GDPR privacy exposure.
- **Unverified Real-World Generalization:** Because distributions are synthetic, metrics reported in this repository do not claim real-world school validation. Live institutional deployment would require local data agreements, domain re-calibration, and human governance approval.
- **Zero Sensitive Attributes:** No names, gender, race, ethnicity, caste, religion, socio-economic markers, or disability statuses are generated, stored, or modeled.
- **Zero-Leakage Target Isolation:** `next_assessment_score` is excluded from model feature matrices (`X`) and retained strictly for target derivation (`at_risk = 1 if next_assessment_score < 40 else 0`).

## Authoritative Derived-Feature Definition
- `previous_score_avg`: **Rolling two-assessment mean** computed post-missingness as `(previous_score + previous_score_2) / 2`, falling back to `previous_score` when `previous_score_2` is unavailable (`NaN`).
- `score_trend`: `previous_score - previous_score_2` (`0.0` when `previous_score_2` is missing).

## Directory Layout
- `raw/student_performance.csv`: Full 10,000-row longitudinal benchmark dataset generated via `python -m src.data_generator --seed 42`.
- `processed/`: Validated temporal and student-disjoint splits exported by `src/train.py` (`train_temporal.csv`, `val_temporal_cal.csv`, `val_temporal_sel.csv`, `test_temporal.csv`, `train_disjoint.csv`, `val_disjoint_cal.csv`, `val_disjoint_sel.csv`, `test_disjoint.csv`), plus runtime `intervention_notes.json` created by teacher workflow updates.
