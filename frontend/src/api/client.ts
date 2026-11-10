import axios from 'axios';
import type {
  PredictionResponse,
  BatchPredictResponse,
  StudentFeatureInput,
  HealthResponse,
  ModelInfo,
  CohortSummary,
  AuditLogs,
  StudentsListResponse,
  StudentDetailResponse,
  GlobalAnalyticsResponse,
  InterventionsResponse,
  InterventionUpsertPayload,
  InterventionRecord,
  StudentReportsResponse,
} from '../types';

export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export function extractApiError(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d?.msg || JSON.stringify(d)).join('; ');
    }
    if (err.code === 'ERR_NETWORK') {
      return `Unable to reach EduGuard API at ${API_BASE}. Ensure the FastAPI service is running.`;
    }
    return err.message || fallback;
  }
  if (err instanceof Error) return err.message;
  return fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

export async function fetchPlatformStatus() {
  const { data } = await api.get('/');
  return data;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const { data } = await api.get<ModelInfo>('/model-info');
  return data;
}

export async function predictStudent(
  features: StudentFeatureInput
): Promise<PredictionResponse> {
  const { data } = await api.post<PredictionResponse>('/predict', features);
  return data;
}

export async function predictBatch(
  students: StudentFeatureInput[]
): Promise<BatchPredictResponse> {
  const { data } = await api.post<BatchPredictResponse>('/predict/batch', { students });
  return data;
}

export async function fetchStudents(params?: {
  search?: string;
  period?: number;
  risk_status?: number;
  limit?: number;
  offset?: number;
}): Promise<StudentsListResponse> {
  const { data } = await api.get<StudentsListResponse>('/students', { params });
  return data;
}

export async function fetchStudentDetail(studentId: string): Promise<StudentDetailResponse> {
  const { data } = await api.get<StudentDetailResponse>(`/students/${studentId}`);
  return data;
}

export async function fetchCohortSummary(): Promise<CohortSummary> {
  const { data } = await api.get<CohortSummary>('/cohort-summary');
  return data;
}

export async function fetchGlobalAnalytics(): Promise<GlobalAnalyticsResponse> {
  const { data } = await api.get<GlobalAnalyticsResponse>('/analytics/global');
  return data;
}

export async function fetchAuditLogs(): Promise<AuditLogs> {
  const { data } = await api.get<AuditLogs>('/audit-logs');
  return data;
}

export async function fetchInterventions(): Promise<InterventionsResponse> {
  const { data } = await api.get<InterventionsResponse>('/interventions');
  return data;
}

export async function saveIntervention(
  payload: InterventionUpsertPayload
): Promise<{ saved: boolean; record: InterventionRecord }> {
  const { data } = await api.post<{ saved: boolean; record: InterventionRecord }>(
    '/interventions',
    payload
  );
  return data;
}

export async function fetchStudentReports(params?: {
  period?: number;
  risk_tier?: string;
  limit?: number;
}): Promise<StudentReportsResponse> {
  const { data } = await api.get<StudentReportsResponse>('/reports/students', {
    params: { ...params, format: 'json' },
  });
  return data;
}

export function getStudentReportsCsvUrl(params?: {
  period?: number;
  risk_tier?: string;
  limit?: number;
}): string {
  const search = new URLSearchParams();
  if (params?.period !== undefined) search.set('period', String(params.period));
  if (params?.risk_tier) search.set('risk_tier', params.risk_tier);
  search.set('limit', String(params?.limit ?? 500));
  search.set('format', 'csv');
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
  return `${base}/reports/students?${search.toString()}`;
}
