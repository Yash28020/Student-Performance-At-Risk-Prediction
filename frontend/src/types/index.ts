// Exact Type Definitions aligned with FastAPI Backend (`api/main.py`)

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  champion_architecture?: string;
  is_calibrated?: boolean;
  selected_threshold?: number;
  model_artifact?: string;
  model_version: string;
  timestamp_utc: string;
}

export interface StudentFeatureInput {
  attendance_pct: number | null;
  assignment_completion_pct: number | null;
  late_assignment_count: number;
  previous_score: number;
  previous_score_2?: number | null;
  previous_score_avg?: number | null;
  score_trend?: number | null;
  assessment_period: number;
}

export interface RiskBandDetail {
  tier: 'LOW_RISK' | 'MODERATE_RISK' | 'HIGH_RISK' | string;
  label: string;
  color: string;
  action: string;
}

export interface SupportPlanItem {
  category: string;
  priority: string;
  source?: string;
  advisory_action: string;
}

export interface SHAPContribution {
  feature: string;
  value: number | null;
  shap_value: number;
  absolute_impact: number;
  direction: 'INCREASES_RISK' | 'REDUCES_RISK' | string;
}

export interface InfluentialSignals {
  shap_available?: boolean;
  risk_probability: number;
  expected_base_value: number;
  explanation_space?: string;
  base_model_architecture?: string;
  calibration_note?: string;
  feature_contributions: SHAPContribution[];
  top_risk_drivers: SHAPContribution[];
  top_protective_factors: SHAPContribution[];
  plain_language_explanation: string;
  disclaimer: string;
}

export interface PredictionResponse {
  prediction: 0 | 1;
  prediction_label: string;
  risk_probability: number;
  applied_threshold: number;
  risk_band: RiskBandDetail;
  input_features: Record<string, number | null>;
  influential_signals?: InfluentialSignals | null;
  explanation?: InfluentialSignals | null; // present in /students/{id} latest_prediction
  support_plan: SupportPlanItem[];
  model_name?: string | null;
  model_version: string;
  is_calibrated?: boolean | null;
  human_review_disclaimer: string;
}

export interface BatchPredictionRow {
  row_index: number;
  prediction: 0 | 1;
  prediction_label: string;
  risk_probability: number;
  applied_threshold: number;
  risk_band: RiskBandDetail;
  input_features?: Record<string, number | null>;
  support_plan: SupportPlanItem[];
  model_name?: string | null;
  model_version?: string;
  is_calibrated?: boolean | null;
}

export interface BatchPredictResponse {
  total_records: number;
  flagged_at_risk_count: number;
  applied_threshold: number;
  model_name?: string | null;
  model_version: string;
  is_calibrated?: boolean | null;
  predictions: BatchPredictionRow[];
}

export interface ConfusionMatrix {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface SplitEvaluationMetrics {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity?: number;
  f1_score: number;
  f2_score: number;
  average_precision?: number;
  pr_auc: number;
  trapezoidal_pr_auc?: number;
  roc_auc: number;
  brier_score: number;
  confusion_matrix: ConfusionMatrix;
  sample_size: number;
  positive_rate: number;
}

export interface CalibrationSummary {
  calibration_method?: string;
  uncalibrated_brier_val_sel?: number;
  calibrated_brier_val_sel?: number;
  uncalibrated_brier_temporal_test?: number;
  calibrated_brier_temporal_test?: number;
  uncalibrated_brier_disjoint_test?: number;
  calibrated_brier_disjoint_test?: number;
}

export interface CalibrationBin {
  bin_range: string;
  count: number;
  mean_predicted_prob: number;
  empirical_positive_rate: number;
}

export interface CalibrationFoldReport {
  brier_score: number;
  expected_calibration_error: number;
  bins: CalibrationBin[];
}

export interface CalibrationAnalysisReport {
  method?: string;
  temporal_val_brier_before?: number;
  temporal_val_brier_after?: number;
  disjoint_val_brier_before?: number;
  disjoint_val_brier_after?: number;
  temporal_test_before?: CalibrationFoldReport;
  temporal_test_after?: CalibrationFoldReport;
  calibration_plot_path?: string;
}

export interface ErrorCohortSummary {
  count: number;
  share_pct: number;
  mean_predicted_prob: number;
  mean_attendance_pct: number;
  mean_assignment_completion_pct: number;
  mean_late_assignment_count: number;
  mean_previous_score: number;
  mean_previous_score_2: number;
  mean_score_trend: number;
  mean_next_assessment_score: number;
}

export interface ErrorAnalysisReport {
  true_positives?: ErrorCohortSummary;
  false_positives?: ErrorCohortSummary;
  false_negatives?: ErrorCohortSummary;
  true_negatives?: ErrorCohortSummary;
}

export interface CandidateSelectionScore {
  mean_average_precision: number;
  mean_roc_auc: number;
  temporal_val_optimized_f2: number;
  mean_brier_score: number;
  composite_score: number;
}

export interface ModelBenchmarkEntry {
  training_time_sec: number;
  inference_latency_ms: number;
  is_calibrated?: boolean;
  calibration_method?: string;
  selection_composite_score?: number;
  calibration_summary?: CalibrationSummary;
  temporal_validation: SplitEvaluationMetrics;
  student_disjoint_validation: SplitEvaluationMetrics;
}

export interface ModelInfo {
  model_version: string;
  model_name?: string;
  champion_architecture: string;
  is_calibrated?: boolean;
  calibration_method?: string;
  calibration_summary?: CalibrationSummary;
  selection_rationale?: string;
  selection_formula?: string;
  candidate_selection_scores?: Record<string, CandidateSelectionScore>;
  target_definition: string;
  derived_feature_rule?: string;
  selected_threshold: number;
  threshold_rationale: string;
  validation_metrics: SplitEvaluationMetrics;
  held_out_test_metrics_temporal: SplitEvaluationMetrics;
  held_out_test_metrics_disjoint: SplitEvaluationMetrics;
  feature_schema: string[];
  risk_bands: Record<string, { label: string; description: string; color: string }>;
  responsible_ai_disclaimer: string;
  training_timestamp_utc?: string;
  split_sizes?: Record<string, number>;
  data_generator_parameters?: {
    num_students: number;
    num_periods: number;
    total_snapshots: number;
    random_seed: number;
  };
  environment_versions?: Record<string, string>;
  model_benchmarks?: Record<string, ModelBenchmarkEntry>;
  disjoint_test_by_period?: Record<string, SplitEvaluationMetrics>;
  calibration_analysis?: CalibrationAnalysisReport;
  temporal_test_error_analysis?: ErrorAnalysisReport;
  threshold_optimization_sweep?: SplitEvaluationMetrics[];
  data_quality_status?: string;
}

export interface StudentRecord {
  student_id: string;
  assessment_period: number;
  attendance_pct: number | null;
  assignment_completion_pct: number | null;
  late_assignment_count: number;
  previous_score: number;
  previous_score_2: number | null;
  previous_score_avg: number | null;
  score_trend: number | null;
  at_risk: number;
}

export interface StudentsListResponse {
  total: number;
  limit: number;
  offset: number;
  students: StudentRecord[];
}

export interface StudentDetailResponse {
  student_id: string;
  total_snapshots: number;
  history: StudentRecord[];
  latest_snapshot: StudentRecord | null;
  latest_prediction: PredictionResponse | null;
}

export interface PeriodTrend {
  period: number;
  avg_attendance: number;
  avg_completion: number;
  avg_score: number;
  passing_avg_score: number | null;
  at_risk_avg_score: number | null;
  at_risk_count: number;
  at_risk_rate: number;
}

export interface EngagementBand {
  band: string;
  count: number;
  risk_rate: number;
}

export interface CohortSummary {
  total_unique_students: number;
  total_snapshots: number;
  at_risk_count: number;
  passing_count: number;
  at_risk_rate_pct: number;
  latest_period_at_risk_students: number;
  avg_attendance: number;
  avg_completion: number;
  avg_late_assignments: number;
  avg_score: number;
  period_trends: PeriodTrend[];
  attendance_bands: EngagementBand[];
  completion_bands: EngagementBand[];
}

export interface GlobalSHAPItem {
  feature: string;
  mean_absolute_shap: number;
}

export interface FeatureCorrelationItem {
  feature: string;
  correlation_with_risk: number;
}

export interface ConstellationPoint {
  student_id: string;
  period: number;
  attendance_pct: number;
  assignment_completion_pct: number;
  previous_score: number;
  score_trend: number;
  late_assignment_count: number;
  risk_probability: number;
  at_risk: number;
}

export interface GlobalAnalyticsResponse {
  classifier_architecture?: string;
  model_name?: string;
  is_calibrated?: boolean;
  selected_threshold?: number;
  global_shap_importance: GlobalSHAPItem[];
  feature_correlations: FeatureCorrelationItem[];
  constellation_sample: ConstellationPoint[];
  sample_size: number;
}

export interface FeatureExclusionEntry {
  attribute: string;
  status: string;
  rationale: string;
}

export interface AuditLogs {
  audit_version: string;
  generated_timestamp: string;
  responsible_ai_policy: {
    advisory_only: boolean;
    automated_sanctions_prohibited: boolean;
    human_in_the_loop_mandatory: boolean;
    model_name?: string;
    champion_architecture?: string;
    is_calibrated?: boolean;
    model_version: string;
  };
  feature_exclusion_registry: FeatureExclusionEntry[];
  validation_ledger: {
    validation_methodology: string;
    selected_threshold: number;
    threshold_rationale: string;
    safety_net_recall: number;
    educator_precision: number;
    average_precision?: number;
    specificity?: number;
    brier_score?: number;
  };
}

export type InterventionStatus =
  | 'Pending Review'
  | 'Outreach Scheduled'
  | 'Mentoring Active'
  | 'Resolved';

export type InterventionPriority = 'Urgent' | 'High' | 'Medium' | 'Low' | 'Routine';

export interface InterventionRecord {
  student_id: string;
  assessment_period: number;
  risk_probability: number | null;
  priority: InterventionPriority;
  recommended_support: string;
  status: InterventionStatus;
  teacher_notes: string;
  source?: string;
  updated_at_utc: string;
}

export interface InterventionsResponse {
  total: number;
  status_summary: Record<InterventionStatus, number>;
  interventions: InterventionRecord[];
}

export interface InterventionUpsertPayload {
  student_id: string;
  status: InterventionStatus;
  priority: InterventionPriority;
  teacher_notes: string;
  recommended_support?: string;
  assessment_period?: number;
  risk_probability?: number | null;
}

export interface StudentReportRow {
  student_id: string;
  assessment_period: number;
  attendance_pct: number | null;
  assignment_completion_pct: number | null;
  late_assignment_count: number;
  previous_score: number;
  previous_score_2: number | null;
  previous_score_avg: number;
  score_trend: number;
  risk_probability: number;
  predicted_at_risk: number;
  risk_tier: string;
  risk_label: string;
  recommended_support_category: string;
  intervention_status: string;
  teacher_notes: string;
}

export interface PeriodReportSummary {
  period: number;
  student_count: number;
  flagged_count: number;
  actual_at_risk_count: number;
  avg_attendance: number;
  avg_completion: number;
  avg_score: number;
  avg_risk_probability_pct: number;
}

export interface StudentReportsResponse {
  period: number;
  total_matching_records: number;
  returned_records: number;
  applied_threshold: number;
  tier_distribution: Record<string, number>;
  period_reports: PeriodReportSummary[];
  records: StudentReportRow[];
}

export type WorldId = 0 | 1 | 2 | 3 | 4 | 5 | 6;

export interface WorldConfig {
  id: WorldId;
  shortName: string;
  name: string;
  moduleName: string;
  subtitle: string;
  route: string;
  accentColor: string;
  glowColor: string;
  themeGradient: string;
}
