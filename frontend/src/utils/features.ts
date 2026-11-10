import type { StudentFeatureInput } from '../types';

export interface DerivedAcademicMetrics {
  previous_score_avg: number;
  score_trend: number;
}

/**
 * Authoritative client-side mirror of Python `src.feature_engineering.compute_derived_features`.
 * Uses the rolling two-assessment mean so training, offline evaluation, API, and UI remain identical.
 */
export function computeDerivedFeatures(
  previousScore: number,
  previousScore2?: number | null
): DerivedAcademicMetrics {
  const hasPrev2 =
    previousScore2 !== undefined &&
    previousScore2 !== null &&
    Number.isFinite(previousScore2);

  const prevAvg = hasPrev2
    ? (previousScore + (previousScore2 as number)) / 2.0
    : previousScore;
  const trend = hasPrev2
    ? previousScore - (previousScore2 as number)
    : 0.0;

  return {
    previous_score_avg: Number(prevAvg.toFixed(2)),
    score_trend: Number(trend.toFixed(2)),
  };
}

export function withDerivedFeatures(input: StudentFeatureInput): StudentFeatureInput {
  const derived = computeDerivedFeatures(input.previous_score, input.previous_score_2);
  return {
    ...input,
    previous_score_avg: derived.previous_score_avg,
    score_trend: derived.score_trend,
  };
}

