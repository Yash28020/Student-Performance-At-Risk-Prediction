import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  BookOpen,
  Search,
  Sparkles,
  ScrollText,
  UserCheck,
  Sliders,
  Table2,
  Plus,
  Trash2,
  Upload,
  CheckCircle2,
  ArrowUpRight,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WorldTransition } from '../components/WorldTransition';
import { RiskGauge } from '../components/RiskGauge';
import { SHAPChart } from '../components/SHAPChart';
import { LoadingSpell } from '../components/LoadingSpell';
import { ErrorPortal } from '../components/ErrorPortal';
import { AdvisoryBanner } from '../components/AdvisoryBanner';
import {
  extractApiError,
  fetchStudents,
  fetchStudentDetail,
  predictBatch,
  predictStudent,
  saveIntervention,
} from '../api/client';
import { computeDerivedFeatures, withDerivedFeatures } from '../utils/features';
import type {
  BatchPredictResponse,
  PredictionResponse,
  StudentDetailResponse,
  StudentFeatureInput,
  StudentRecord,
} from '../types';

const DEFAULT_BATCH_ROWS: StudentFeatureInput[] = [
  {
    attendance_pct: 58,
    assignment_completion_pct: 52,
    late_assignment_count: 5,
    previous_score: 36,
    previous_score_2: 43,
    assessment_period: 3,
  },
  {
    attendance_pct: 74,
    assignment_completion_pct: 66,
    late_assignment_count: 2,
    previous_score: 45,
    previous_score_2: 49,
    assessment_period: 3,
  },
  {
    attendance_pct: 91,
    assignment_completion_pct: 94,
    late_assignment_count: 0,
    previous_score: 79,
    previous_score_2: 75,
    assessment_period: 4,
  },
];

export const World2Academy: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initialStudent = searchParams.get('student');

  const [mode, setMode] = useState<'archives' | 'oracle' | 'batch'>('archives');

  // Archives state
  const [search, setSearch] = useState(initialStudent || '');
  const [periodFilter, setPeriodFilter] = useState<string>('4');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [students, setStudents] = useState<StudentRecord[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(initialStudent);
  const [studentDetail, setStudentDetail] = useState<StudentDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [queuingIntervention, setQueuingIntervention] = useState(false);
  const [queuedConfirmation, setQueuedConfirmation] = useState<string | null>(null);
  const [queueError, setQueueError] = useState<string | null>(null);

  // Live Single Student Oracle state
  const [hasPriorScore2, setHasPriorScore2] = useState(true);
  const [oracleInput, setOracleInput] = useState<StudentFeatureInput>({
    attendance_pct: 68.0,
    assignment_completion_pct: 58.0,
    late_assignment_count: 3,
    previous_score: 38.0,
    previous_score_2: 46.0,
    assessment_period: 3,
  });
  const [oraclePrediction, setOraclePrediction] = useState<PredictionResponse | null>(null);
  const [loadingOracle, setLoadingOracle] = useState(false);
  const [oracleError, setOracleError] = useState<string | null>(null);

  // Batch Risk Scorer state
  const [batchRows, setBatchRows] = useState<StudentFeatureInput[]>(DEFAULT_BATCH_ROWS);
  const [batchResult, setBatchResult] = useState<BatchPredictResponse | null>(null);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);

  const handleSelectStudent = async (studentId: string) => {
    setSelectedStudentId(studentId);
    setLoadingDetail(true);
    setDetailError(null);
    setQueuedConfirmation(null);
    setQueueError(null);
    try {
      const detail = await fetchStudentDetail(studentId);
      setStudentDetail(detail);
    } catch (err) {
      setStudentDetail(null);
      setDetailError(extractApiError(err, `Failed to load scroll dossier for ${studentId}.`));
    } finally {
      setLoadingDetail(false);
    }
  };

  const loadRoster = async () => {
    setLoadingList(true);
    setListError(null);
    try {
      const res = await fetchStudents({
        search: search || undefined,
        period: periodFilter === 'all' ? undefined : Number(periodFilter),
        risk_status: riskFilter === 'all' ? undefined : Number(riskFilter),
        limit: 24,
        offset: 0,
      });
      setStudents(res.students);
      setTotalMatches(res.total);
      if (initialStudent && !studentDetail) {
        handleSelectStudent(initialStudent);
      } else if (res.students.length > 0 && !selectedStudentId) {
        handleSelectStudent(res.students[0].student_id);
      }
    } catch (err) {
      setListError(extractApiError(err, 'Could not fetch Academy archives.'));
    } finally {
      setLoadingList(false);
    }
  };

  const handleQueueIntervention = async () => {
    if (!studentDetail || !studentDetail.latest_prediction || !studentDetail.latest_snapshot) return;
    setQueuingIntervention(true);
    setQueueError(null);
    setQueuedConfirmation(null);
    const pred = studentDetail.latest_prediction;
    const topPlan = pred.support_plan[0];
    const priorityMap: Record<string, 'Urgent' | 'High' | 'Medium' | 'Low' | 'Routine'> = {
      Urgent: 'Urgent',
      High: 'High',
      Medium: 'Medium',
      Low: 'Low',
      Routine: 'Routine',
    };
    try {
      await saveIntervention({
        student_id: studentDetail.student_id,
        status: 'Outreach Scheduled',
        priority: priorityMap[topPlan?.priority || 'High'] || 'High',
        recommended_support: topPlan
          ? `${topPlan.category}: ${topPlan.advisory_action}`
          : pred.risk_band.action,
        teacher_notes: '',
        assessment_period: studentDetail.latest_snapshot.assessment_period,
        risk_probability: pred.risk_probability,
      });
      setQueuedConfirmation(`${studentDetail.student_id} queued in Intervention Center.`);
    } catch (err) {
      setQueueError(extractApiError(err, 'Unable to save intervention entry.'));
    } finally {
      setQueuingIntervention(false);
    }
  };

  const runOraclePrediction = async (input: StudentFeatureInput, usePrev2 = hasPriorScore2) => {
    setLoadingOracle(true);
    setOracleError(null);
    try {
      const normalized: StudentFeatureInput = {
        ...input,
        previous_score_2: usePrev2 ? input.previous_score_2 : null,
      };
      const payload = withDerivedFeatures(normalized);
      const res = await predictStudent(payload);
      setOraclePrediction(res);
    } catch (err) {
      setOracleError(extractApiError(err, 'Prediction request failed. Check input ranges and API connectivity.'));
    } finally {
      setLoadingOracle(false);
    }
  };

  const runBatchPrediction = async (rows: StudentFeatureInput[]) => {
    setLoadingBatch(true);
    setBatchError(null);
    try {
      const normalizedRows = rows.map(withDerivedFeatures);
      const res = await predictBatch(normalizedRows);
      setBatchResult(res);
    } catch (err) {
      setBatchResult(null);
      setBatchError(extractApiError(err, 'Batch scoring failed. Validate all numeric bounds.'));
    } finally {
      setLoadingBatch(false);
    }
  };

  const handleBatchCsvUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBatchError(null);
    try {
      const text = await file.text();
      const lines = text
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
      if (lines.length === 0) {
        setBatchError('Uploaded CSV file is empty.');
        return;
      }
      const parsed: StudentFeatureInput[] = [];
      const firstCells = lines[0].split(',').map((c) => c.trim().toLowerCase());
      const hasHeader = firstCells.includes('attendance_pct') || firstCells.includes('previous_score');
      const dataLines = hasHeader ? lines.slice(1) : lines;

      for (let i = 0; i < dataLines.length && i < 100; i++) {
        const cols = dataLines[i].split(',').map((c) => c.trim());
        if (hasHeader) {
          const getVal = (name: string, fallback: number | null) => {
            const idx = firstCells.indexOf(name);
            if (idx < 0 || idx >= cols.length || cols[idx] === '') return fallback;
            const n = Number(cols[idx]);
            return Number.isFinite(n) ? n : fallback;
          };
          parsed.push({
            attendance_pct: getVal('attendance_pct', 75),
            assignment_completion_pct: getVal('assignment_completion_pct', 75),
            late_assignment_count: Math.round(getVal('late_assignment_count', 1) ?? 1),
            previous_score: getVal('previous_score', 55) ?? 55,
            previous_score_2: getVal('previous_score_2', null),
            assessment_period: Math.round(getVal('assessment_period', 3) ?? 3),
          });
        } else {
          if (cols.length < 4) continue;
          parsed.push({
            attendance_pct: cols[0] === '' ? null : Number(cols[0]),
            assignment_completion_pct: cols[1] === '' ? null : Number(cols[1]),
            late_assignment_count: Number(cols[2]),
            previous_score: Number(cols[3]),
            previous_score_2: cols[4] !== undefined && cols[4] !== '' ? Number(cols[4]) : null,
            assessment_period: cols[5] !== undefined && cols[5] !== '' ? Number(cols[5]) : 3,
          });
        }
      }
      if (parsed.length === 0) {
        setBatchError('No valid student rows parsed from CSV.');
        return;
      }
      setBatchRows(parsed);
      await runBatchPrediction(parsed);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Unable to read CSV file.');
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      loadRoster();
    }, 250);
    return () => clearTimeout(timer);
  }, [search, periodFilter, riskFilter]);

  useEffect(() => {
    if (mode === 'oracle' && !oraclePrediction && !oracleError) {
      runOraclePrediction(oracleInput, hasPriorScore2);
    }
    if (mode === 'batch' && !batchResult && !batchError) {
      runBatchPrediction(batchRows);
    }
  }, [mode]);

  const activePred = studentDetail?.latest_prediction;
  const activeSignals = activePred?.explanation || activePred?.influential_signals;
  const oracleDerived = computeDerivedFeatures(
    oracleInput.previous_score,
    hasPriorScore2 ? oracleInput.previous_score_2 : null
  );

  return (
    <WorldTransition worldKey="world-2">
      <div className="relative min-h-screen bg-[#120D1A] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="academy" />
        <ParticleField theme="academy" density={45} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Academy Header */}
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/35 bg-amber-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-amber-200">
                <BookOpen className="h-3.5 w-3.5" />
                World 2 · Student Intelligence, Batch Scoring & Explainability
              </div>
              <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-[#F5E6C8] via-[#F4C76B] to-[#C9A84C] bg-clip-text sm:text-4xl">
                The Wizarding Academy
              </h1>
              <p className="mt-1 text-sm text-amber-100/75">
                Inspect student dossiers, evaluate live single or batch predictions, review SHAP drivers, and queue teacher interventions.
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex flex-wrap rounded-xl border border-amber-300/25 bg-slate-900/70 p-1 backdrop-blur-md">
              <button
                onClick={() => setMode('archives')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  mode === 'archives'
                    ? 'border border-amber-400/40 bg-amber-400/20 text-amber-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <ScrollText className="h-3.5 w-3.5" />
                Student Profiles
              </button>
              <button
                onClick={() => setMode('oracle')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  mode === 'oracle'
                    ? 'border border-purple-400/40 bg-purple-500/25 text-purple-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                Single SHAP Oracle
              </button>
              <button
                onClick={() => setMode('batch')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  mode === 'batch'
                    ? 'border border-emerald-400/40 bg-emerald-500/20 text-emerald-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Table2 className="h-3.5 w-3.5" />
                Batch Risk Scorer
              </button>
            </div>
          </div>

          <AdvisoryBanner accentColor="#C9A84C" />

          {mode === 'archives' ? (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Left Scroll Selector Panel */}
              <div className="rounded-2xl border border-amber-400/25 bg-[#1B1326]/85 p-5 backdrop-blur-xl lg:col-span-4">
                <h2 className="font-fantasy text-base font-bold text-amber-200">
                  Student Scroll Registry ({totalMatches.toLocaleString()})
                </h2>

                {/* Search & Filters */}
                <div className="mt-3 space-y-2.5">
                  <div className="relative">
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-amber-300/70" />
                    <input
                      type="text"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search ID (e.g. STU_00012)"
                      className="w-full rounded-xl border border-amber-400/25 bg-slate-950/70 py-2 pl-9 pr-3 text-xs text-white placeholder-slate-400 focus:border-amber-300 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <select
                      value={periodFilter}
                      onChange={(e) => setPeriodFilter(e.target.value)}
                      className="rounded-xl border border-amber-400/20 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 focus:border-amber-300 focus:outline-none"
                    >
                      <option value="all">All Periods</option>
                      <option value="1">Period 1</option>
                      <option value="2">Period 2</option>
                      <option value="3">Period 3</option>
                      <option value="4">Period 4</option>
                    </select>

                    <select
                      value={riskFilter}
                      onChange={(e) => setRiskFilter(e.target.value)}
                      className="rounded-xl border border-amber-400/20 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 focus:border-amber-300 focus:outline-none"
                    >
                      <option value="all">All Outcomes</option>
                      <option value="1">At-Risk (&lt; 40)</option>
                      <option value="0">Passing (≥ 40)</option>
                    </select>
                  </div>
                </div>

                {/* Enchanted Roster Scrolls */}
                <div className="mt-4 max-h-[540px] space-y-2 overflow-y-auto pr-1">
                  {loadingList ? (
                    <p className="py-8 text-center text-xs text-slate-400">Searching student registry...</p>
                  ) : listError ? (
                    <ErrorPortal message={listError} onRetry={loadRoster} />
                  ) : students.length === 0 ? (
                    <p className="py-8 text-center text-xs text-slate-400">No students matched your query.</p>
                  ) : (
                    students.map((st, idx) => {
                      const isSelected = st.student_id === selectedStudentId;
                      return (
                        <button
                          key={`${st.student_id}-${st.assessment_period}-${idx}`}
                          onClick={() => handleSelectStudent(st.student_id)}
                          className={`w-full rounded-xl border p-3 text-left transition ${
                            isSelected
                              ? 'border-amber-300/70 bg-amber-500/15 shadow-[0_0_18px_rgba(201,168,76,0.2)]'
                              : 'border-white/10 bg-slate-950/50 hover:border-amber-300/35 hover:bg-slate-900/70'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs font-bold text-amber-200">
                              {st.student_id}
                            </span>
                            <span className={st.at_risk === 1 ? 'risk-badge-high' : 'risk-badge-low'}>
                              {st.at_risk === 1 ? 'At-Risk Outcome' : 'Passing'}
                            </span>
                          </div>
                          <div className="mt-2 grid grid-cols-3 gap-1 text-[11px] text-slate-300/85">
                            <span>Score: <b>{st.previous_score.toFixed(0)}</b></span>
                            <span>Att: <b>{st.attendance_pct !== null ? `${st.attendance_pct.toFixed(0)}%` : 'N/A'}</b></span>
                            <span>Per: <b>{st.assessment_period}</b></span>
                          </div>
                        </button>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Right Student Dossier Detail */}
              <div className="lg:col-span-8">
                {loadingDetail ? (
                  <LoadingSpell message="Unrolling Student Dossier..." accentColor="#C9A84C" />
                ) : detailError ? (
                  <ErrorPortal
                    message={detailError}
                    onRetry={() => selectedStudentId && handleSelectStudent(selectedStudentId)}
                  />
                ) : studentDetail && activePred ? (
                  <div className="space-y-5">
                    {/* Parchment / Enchanted Hero Scroll Card */}
                    <div className="relative overflow-hidden rounded-2xl border border-amber-400/35 bg-gradient-to-br from-[#2B1F2E]/90 via-[#1E1729]/95 to-[#15101F]/95 p-6 shadow-2xl backdrop-blur-xl">
                      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-amber-300/15 pb-4">
                        <div>
                          <p className="text-[11px] font-bold uppercase tracking-widest text-amber-300/80">
                            Student Dossier · Latest Snapshot (Period {studentDetail.latest_snapshot?.assessment_period})
                          </p>
                          <h2 className="mt-1 font-fantasy text-2xl font-bold text-[#F5E6C8]">
                            {studentDetail.student_id}
                          </h2>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className="rounded-full border px-3 py-1 text-xs font-bold"
                            style={{
                              color: activePred.risk_band.color,
                              borderColor: `${activePred.risk_band.color}66`,
                              backgroundColor: `${activePred.risk_band.color}1F`,
                            }}
                          >
                            {activePred.risk_band.label}
                          </span>
                          <button
                            onClick={handleQueueIntervention}
                            disabled={queuingIntervention}
                            className="inline-flex items-center gap-1.5 rounded-full border border-cyan-400/40 bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/30 disabled:opacity-60"
                          >
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            {queuingIntervention ? 'Saving...' : 'Queue Teacher Follow-Up'}
                          </button>
                        </div>
                      </div>

                      {queuedConfirmation && (
                        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-400/40 bg-emerald-950/40 px-3.5 py-2 text-xs text-emerald-200">
                          <span>{queuedConfirmation}</span>
                          <button
                            onClick={() => navigate('/cyberpunk?tab=interventions')}
                            className="inline-flex items-center gap-1 font-semibold underline hover:text-white"
                          >
                            Open Intervention Center <ArrowUpRight className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                      {queueError && (
                        <div className="mt-3 rounded-xl border border-rose-400/40 bg-rose-950/40 px-3.5 py-2 text-xs text-rose-200">
                          {queueError}
                        </div>
                      )}

                      <div className="mt-5 grid grid-cols-1 items-center gap-6 md:grid-cols-12">
                        {/* Animated Risk Gauge */}
                        <div className="flex flex-col items-center md:col-span-5">
                          <RiskGauge
                            probability={activePred.risk_probability}
                            threshold={activePred.applied_threshold}
                            riskLabel={activePred.prediction_label}
                            tier={activePred.risk_band.tier}
                          />
                          <p className="mt-1 text-center text-xs text-slate-300">
                            Advisory Focus: <b className="text-amber-200">{activePred.risk_band.action}</b>
                          </p>
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 gap-3 md:col-span-7">
                          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-400">Classroom Attendance</p>
                            <p className="mt-1 font-mono text-lg font-bold text-white">
                              {studentDetail.latest_snapshot?.attendance_pct !== null &&
                              studentDetail.latest_snapshot?.attendance_pct !== undefined
                                ? `${studentDetail.latest_snapshot.attendance_pct.toFixed(1)}%`
                                : 'Imputed'}
                            </p>
                          </div>
                          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-400">Assignments</p>
                            <p className="mt-1 font-mono text-lg font-bold text-white">
                              {studentDetail.latest_snapshot?.assignment_completion_pct !== null &&
                              studentDetail.latest_snapshot?.assignment_completion_pct !== undefined
                                ? `${studentDetail.latest_snapshot.assignment_completion_pct.toFixed(1)}%`
                                : 'Imputed'}
                            </p>
                          </div>
                          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-400">Current Score</p>
                            <p className="mt-1 font-mono text-lg font-bold text-amber-200">
                              {studentDetail.latest_snapshot?.previous_score.toFixed(1)} / 100
                            </p>
                          </div>
                          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-400">Score Momentum</p>
                            <p
                              className={`mt-1 font-mono text-lg font-bold ${
                                (studentDetail.latest_snapshot?.score_trend ?? 0) >= 0
                                  ? 'text-emerald-400'
                                  : 'text-rose-400'
                              }`}
                            >
                              {(studentDetail.latest_snapshot?.score_trend ?? 0) >= 0 ? '+' : ''}
                              {(studentDetail.latest_snapshot?.score_trend ?? 0).toFixed(1)} pts
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Longitudinal Academic History Chart */}
                    <div className="rounded-2xl border border-amber-400/20 bg-slate-900/70 p-5 backdrop-blur-xl">
                      <h3 className="font-fantasy text-base font-bold text-amber-200">
                        Multi-Period Academic Trajectory
                      </h3>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={studentDetail.history} margin={{ top: 12, right: 20, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                          <XAxis dataKey="assessment_period" tickFormatter={(v: number) => `Period ${v}`} stroke="#94A3B8" />
                          <YAxis domain={[0, 100]} stroke="#94A3B8" />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'rgba(15,23,42,0.95)',
                              border: '1px solid #C9A84C',
                              borderRadius: '10px',
                            }}
                          />
                          <ReferenceLine y={40} stroke="#F43F5E" strokeDasharray="4 4" />
                          <Line type="monotone" dataKey="previous_score" name="Assessment Score" stroke="#F4C76B" strokeWidth={3} dot={{ r: 5 }} isAnimationActive={false} />
                          <Line type="monotone" dataKey="attendance_pct" name="Attendance (%)" stroke="#5DE6E0" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Local SHAP Explainability Section */}
                    {activeSignals && (
                      <div className="rounded-2xl border border-purple-400/25 bg-slate-900/75 p-5 backdrop-blur-xl">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <Sparkles className="h-4 w-4 text-purple-300" />
                            <h3 className="font-fantasy text-base font-bold text-purple-200">
                              SHAP Model Explainability (Statistical Attribution)
                            </h3>
                          </div>
                          <span className="rounded-full border border-purple-400/35 bg-purple-500/15 px-2.5 py-0.5 font-mono text-[10px] text-purple-200">
                            {activeSignals.explanation_space || 'Probability Space'}
                          </span>
                        </div>

                        <div className="mb-4 rounded-xl border border-purple-300/20 bg-purple-950/30 p-3.5 text-xs leading-relaxed text-purple-100">
                          {activeSignals.plain_language_explanation}
                        </div>

                        {activeSignals.feature_contributions.length > 0 ? (
                          <SHAPChart
                            contributions={activeSignals.feature_contributions}
                            baseValue={activeSignals.expected_base_value}
                          />
                        ) : (
                          <p className="text-xs text-slate-400">
                            Local SHAP values unavailable for this prediction snapshot; prediction remains operational.
                          </p>
                        )}
                      </div>
                    )}

                    {/* Support Recommendations */}
                    <div className="rounded-2xl border border-amber-400/25 bg-slate-900/75 p-5 backdrop-blur-xl">
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <UserCheck className="h-4 w-4 text-amber-300" />
                          <h3 className="font-fantasy text-base font-bold text-amber-200">
                            Rule-Based Teacher Support Suggestions
                          </h3>
                        </div>
                        <span className="rounded-full border border-amber-400/30 bg-amber-500/10 px-2.5 py-0.5 font-mono text-[10px] text-amber-200">
                          Advisory Heuristic · Separate from SHAP
                        </span>
                      </div>
                      <div className="space-y-2.5">
                        {activePred.support_plan.map((sp, i) => (
                          <div
                            key={i}
                            className="rounded-xl border border-amber-300/20 bg-[#23192D]/70 p-3.5 text-xs"
                          >
                            <div className="flex items-center gap-2">
                              <span className="rounded bg-amber-400/20 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-amber-200">
                                {sp.priority}
                              </span>
                              <span className="font-bold text-white">{sp.category}</span>
                            </div>
                            <p className="mt-1.5 leading-relaxed text-slate-300">{sp.advisory_action}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-slate-900/50 p-12 text-center text-sm text-slate-400">
                    Select any student scroll from the Academy library to inspect predictions and SHAP explanations.
                  </div>
                )}
              </div>
            </div>
          ) : mode === 'oracle' ? (
            /* MODE 2: Interactive Live Single Student Prediction + SHAP Oracle */
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="rounded-2xl border border-purple-400/30 bg-slate-900/75 p-6 backdrop-blur-xl lg:col-span-5">
                <h2 className="font-fantasy text-lg font-bold text-amber-200">
                  Pre-Assessment Indicator Inputs
                </h2>
                <p className="mt-1 text-xs text-slate-400">
                  Adjust school student indicators to invoke the live calibrated champion pipeline and SHAP explainer.
                </p>

                <div className="mt-5 space-y-4">
                  <div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">Classroom Attendance (%)</span>
                      <span className="font-mono font-bold text-amber-300">{oracleInput.attendance_pct}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={0.5}
                      value={oracleInput.attendance_pct ?? 0}
                      onChange={(e) =>
                        setOracleInput({ ...oracleInput, attendance_pct: Number(e.target.value) })
                      }
                      className="mt-1.5 w-full accent-amber-400"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">Assignment Completion (%)</span>
                      <span className="font-mono font-bold text-amber-300">
                        {oracleInput.assignment_completion_pct}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={0.5}
                      value={oracleInput.assignment_completion_pct ?? 0}
                      onChange={(e) =>
                        setOracleInput({
                          ...oracleInput,
                          assignment_completion_pct: Number(e.target.value),
                        })
                      }
                      className="mt-1.5 w-full accent-amber-400"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">Current Assessment Score (t)</span>
                      <span className="font-mono font-bold text-amber-300">{oracleInput.previous_score}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={oracleInput.previous_score}
                      onChange={(e) =>
                        setOracleInput({ ...oracleInput, previous_score: Number(e.target.value) })
                      }
                      className="mt-1.5 w-full accent-amber-400"
                    />
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs">
                      <label className="flex items-center gap-2 text-slate-300">
                        <input
                          type="checkbox"
                          checked={hasPriorScore2}
                          onChange={(e) => setHasPriorScore2(e.target.checked)}
                          className="accent-amber-400"
                        />
                        <span>Prior Score Available (t-1)</span>
                      </label>
                      <span className="font-mono font-bold text-amber-300">
                        {hasPriorScore2 ? (oracleInput.previous_score_2 ?? 46) : 'Fallback to t'}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      disabled={!hasPriorScore2}
                      value={oracleInput.previous_score_2 ?? 46}
                      onChange={(e) =>
                        setOracleInput({ ...oracleInput, previous_score_2: Number(e.target.value) })
                      }
                      className="mt-1.5 w-full accent-amber-400 disabled:opacity-40"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-slate-300">Late Assignments</label>
                      <input
                        type="number"
                        min={0}
                        max={25}
                        value={oracleInput.late_assignment_count}
                        onChange={(e) =>
                          setOracleInput({
                            ...oracleInput,
                            late_assignment_count: Math.max(0, Math.min(25, Number(e.target.value))),
                          })
                        }
                        className="mt-1 w-full rounded-lg border border-white/15 bg-slate-950/70 px-3 py-2 text-xs text-white"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-300">Period</label>
                      <select
                        value={oracleInput.assessment_period}
                        onChange={(e) =>
                          setOracleInput({
                            ...oracleInput,
                            assessment_period: Number(e.target.value),
                          })
                        }
                        className="mt-1 w-full rounded-lg border border-white/15 bg-slate-950/70 px-3 py-2 text-xs text-white"
                      >
                        {[1, 2, 3, 4].map((p) => (
                          <option key={p} value={p}>
                            Period {p}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Read-only Authoritative Derived Feature Preview */}
                  <div className="rounded-xl border border-amber-400/20 bg-slate-950/70 p-3 text-[11px]">
                    <p className="font-semibold uppercase tracking-wider text-amber-200/90">
                      Authoritative Derived Features (Auto-Computed)
                    </p>
                    <div className="mt-1.5 flex justify-between font-mono text-slate-300">
                      <span>Rolling 2-Score Mean: <b className="text-amber-300">{oracleDerived.previous_score_avg}</b></span>
                      <span>
                        Momentum:{' '}
                        <b className={oracleDerived.score_trend >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {oracleDerived.score_trend >= 0 ? '+' : ''}{oracleDerived.score_trend}
                        </b>
                      </span>
                    </div>
                  </div>

                  <button
                    onClick={() => runOraclePrediction(oracleInput, hasPriorScore2)}
                    disabled={loadingOracle}
                    className="w-full rounded-xl bg-gradient-to-r from-amber-400 to-amber-500 py-3 font-fantasy text-xs font-bold uppercase tracking-widest text-slate-950 shadow-lg transition hover:brightness-110 disabled:opacity-60"
                  >
                    {loadingOracle ? 'Computing ML + SHAP...' : 'Invoke Real-Time SHAP Appraisal'}
                  </button>
                </div>
              </div>

              <div className="space-y-5 lg:col-span-7">
                {oracleError ? (
                  <ErrorPortal
                    title="Oracle Inference Error"
                    message={oracleError}
                    onRetry={() => runOraclePrediction(oracleInput, hasPriorScore2)}
                  />
                ) : oraclePrediction ? (
                  <>
                    <div className="rounded-2xl border border-amber-400/30 bg-slate-900/75 p-6 backdrop-blur-xl">
                      <div className="grid grid-cols-1 items-center gap-4 md:grid-cols-2">
                        <RiskGauge
                          probability={oraclePrediction.risk_probability}
                          threshold={oraclePrediction.applied_threshold}
                          riskLabel={oraclePrediction.prediction_label}
                          tier={oraclePrediction.risk_band.tier}
                        />
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className="inline-block rounded-full border px-3 py-1 text-xs font-bold"
                              style={{
                                color: oraclePrediction.risk_band.color,
                                borderColor: `${oraclePrediction.risk_band.color}66`,
                                backgroundColor: `${oraclePrediction.risk_band.color}22`,
                              }}
                            >
                              {oraclePrediction.risk_band.label}
                            </span>
                            {oraclePrediction.model_name && (
                              <span className="rounded-full border border-white/15 bg-white/5 px-2.5 py-0.5 font-mono text-[10px] text-slate-300">
                                {oraclePrediction.model_name}
                              </span>
                            )}
                          </div>
                          <p className="mt-3 text-xs leading-relaxed text-slate-300">
                            {oraclePrediction.influential_signals?.plain_language_explanation}
                          </p>
                        </div>
                      </div>
                    </div>

                    {oraclePrediction.influential_signals && (
                      <div className="rounded-2xl border border-purple-400/25 bg-slate-900/75 p-6 backdrop-blur-xl">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <h3 className="font-fantasy text-base font-bold text-purple-200">
                            Local SHAP Contributions (Model Explainability)
                          </h3>
                          <span className="font-mono text-[10px] text-purple-300/80">
                            {oraclePrediction.influential_signals.explanation_space || 'Probability Space'}
                          </span>
                        </div>
                        {oraclePrediction.influential_signals.feature_contributions.length > 0 ? (
                          <SHAPChart
                            contributions={oraclePrediction.influential_signals.feature_contributions}
                            baseValue={oraclePrediction.influential_signals.expected_base_value}
                          />
                        ) : (
                          <p className="text-xs text-slate-400">
                            Local SHAP explanation unavailable; calibrated probability prediction active.
                          </p>
                        )}
                      </div>
                    )}

                    {/* Clear separation: Rule-Based Recommendations */}
                    <div className="rounded-2xl border border-amber-400/20 bg-slate-900/75 p-5 backdrop-blur-xl">
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <h3 className="font-fantasy text-base font-bold text-amber-200">
                          Rule-Based Support Recommendations
                        </h3>
                        <span className="rounded-full border border-amber-400/30 bg-amber-500/10 px-2.5 py-0.5 font-mono text-[10px] text-amber-200">
                          Rule-Based Advisory · Non-Causal
                        </span>
                      </div>
                      <div className="space-y-2">
                        {oraclePrediction.support_plan.map((sp, idx) => (
                          <div key={idx} className="rounded-xl border border-white/10 bg-slate-950/60 p-3 text-xs">
                            <span className="mr-2 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-amber-200">
                              {sp.priority}
                            </span>
                            <b className="text-white">{sp.category}:</b>{' '}
                            <span className="text-slate-300">{sp.advisory_action}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <LoadingSpell message="Awakening SHAP Oracle..." accentColor="#C9A84C" />
                )}
              </div>
            </div>
          ) : (
            /* MODE 3: Batch Risk Scorer (/predict/batch) */
            <div className="space-y-6">
              <div className="rounded-2xl border border-emerald-400/25 bg-slate-900/75 p-6 backdrop-blur-xl">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="font-fantasy text-lg font-bold text-emerald-200">
                      Classroom Batch Risk Scoring Console
                    </h2>
                    <p className="text-xs text-slate-400">
                      Edit rows inline or upload a CSV (columns: attendance_pct, assignment_completion_pct, late_assignment_count, previous_score, previous_score_2, assessment_period).
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5">
                    <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-amber-400/35 bg-amber-500/10 px-3.5 py-2 text-xs font-semibold text-amber-200 transition hover:bg-amber-500/20">
                      <Upload className="h-3.5 w-3.5" />
                      Upload CSV
                      <input type="file" accept=".csv" onChange={handleBatchCsvUpload} className="hidden" />
                    </label>
                    <button
                      onClick={() =>
                        setBatchRows([
                          ...batchRows,
                          {
                            attendance_pct: 80,
                            assignment_completion_pct: 78,
                            late_assignment_count: 1,
                            previous_score: 52,
                            previous_score_2: 50,
                            assessment_period: 3,
                          },
                        ])
                      }
                      className="inline-flex items-center gap-1.5 rounded-xl border border-white/15 bg-white/5 px-3.5 py-2 text-xs font-semibold text-slate-200 transition hover:bg-white/15"
                    >
                      <Plus className="h-3.5 w-3.5" />
                      Add Student Row
                    </button>
                    <button
                      onClick={() => runBatchPrediction(batchRows)}
                      disabled={loadingBatch || batchRows.length === 0}
                      className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-400 to-teal-500 px-5 py-2 font-fantasy text-xs font-bold uppercase tracking-wider text-slate-950 shadow-lg transition hover:brightness-110 disabled:opacity-60"
                    >
                      {loadingBatch ? 'Scoring Batch...' : `Score ${batchRows.length} Students`}
                    </button>
                  </div>
                </div>

                {/* Editable Input Rows */}
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-white/10 text-slate-400">
                        <th className="py-2 pr-2">#</th>
                        <th className="pr-2">Attendance %</th>
                        <th className="pr-2">Completion %</th>
                        <th className="pr-2">Late Count</th>
                        <th className="pr-2">Score (t)</th>
                        <th className="pr-2">Score (t-1)</th>
                        <th className="pr-2">Period</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {batchRows.map((row, idx) => (
                        <tr key={idx}>
                          <td className="py-2 pr-2 text-slate-400">{idx + 1}</td>
                          <td className="py-1.5 pr-2">
                            <input
                              type="number"
                              min={0}
                              max={100}
                              value={row.attendance_pct ?? ''}
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = { ...row, attendance_pct: e.target.value === '' ? null : Number(e.target.value) };
                                setBatchRows(next);
                              }}
                              className="w-20 rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            />
                          </td>
                          <td className="py-1.5 pr-2">
                            <input
                              type="number"
                              min={0}
                              max={100}
                              value={row.assignment_completion_pct ?? ''}
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = { ...row, assignment_completion_pct: e.target.value === '' ? null : Number(e.target.value) };
                                setBatchRows(next);
                              }}
                              className="w-20 rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            />
                          </td>
                          <td className="py-1.5 pr-2">
                            <input
                              type="number"
                              min={0}
                              max={30}
                              value={row.late_assignment_count}
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = { ...row, late_assignment_count: Number(e.target.value) };
                                setBatchRows(next);
                              }}
                              className="w-16 rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            />
                          </td>
                          <td className="py-1.5 pr-2">
                            <input
                              type="number"
                              min={0}
                              max={100}
                              value={row.previous_score}
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = { ...row, previous_score: Number(e.target.value) };
                                setBatchRows(next);
                              }}
                              className="w-20 rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            />
                          </td>
                          <td className="py-1.5 pr-2">
                            <input
                              type="number"
                              min={0}
                              max={100}
                              value={row.previous_score_2 ?? ''}
                              placeholder="N/A"
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = {
                                  ...row,
                                  previous_score_2: e.target.value === '' ? null : Number(e.target.value),
                                };
                                setBatchRows(next);
                              }}
                              className="w-20 rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            />
                          </td>
                          <td className="py-1.5 pr-2">
                            <select
                              value={row.assessment_period}
                              onChange={(e) => {
                                const next = [...batchRows];
                                next[idx] = { ...row, assessment_period: Number(e.target.value) };
                                setBatchRows(next);
                              }}
                              className="rounded-lg border border-white/15 bg-slate-950/70 px-2 py-1 text-white"
                            >
                              {[1, 2, 3, 4].map((p) => (
                                <option key={p} value={p}>P{p}</option>
                              ))}
                            </select>
                          </td>
                          <td>
                            {batchRows.length > 1 && (
                              <button
                                onClick={() => setBatchRows(batchRows.filter((_, i) => i !== idx))}
                                className="rounded p-1 text-slate-400 hover:bg-rose-500/20 hover:text-rose-300"
                                title="Remove Row"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {batchError && (
                <ErrorPortal
                  title="Batch Validation / Scoring Error"
                  message={batchError}
                  onRetry={() => runBatchPrediction(batchRows)}
                />
              )}

              {batchResult && (
                <div className="rounded-2xl border border-amber-400/25 bg-slate-900/80 p-6 backdrop-blur-xl">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="font-fantasy text-base font-bold text-amber-200">
                        Batch Prediction Output ({batchResult.total_records} Records)
                      </h3>
                      <p className="text-xs text-slate-400">
                        {batchResult.model_name || 'Champion Model'} · Applied Threshold τ* = {batchResult.applied_threshold}
                      </p>
                    </div>
                    <span className="rounded-full border border-rose-400/40 bg-rose-500/15 px-3 py-1 font-mono text-xs font-bold text-rose-200">
                      {batchResult.flagged_at_risk_count} / {batchResult.total_records} Flagged At-Risk
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-white/10 font-mono text-slate-400">
                          <th className="py-2 pr-3">Row</th>
                          <th className="pr-3">Risk Prob</th>
                          <th className="pr-3">Classification</th>
                          <th className="pr-3">Risk Band</th>
                          <th>Top Rule-Based Support Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5">
                        {batchResult.predictions.map((pred, idx) => (
                          <tr key={pred.row_index ?? idx}>
                            <td className="py-2.5 pr-3 font-mono text-slate-300">
                              #{(pred.row_index ?? idx) + 1}
                            </td>
                            <td className="py-2.5 pr-3 font-mono font-bold text-white">
                              {(pred.risk_probability * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 pr-3">
                              <span className={`whitespace-nowrap ${pred.prediction === 1 ? 'risk-badge-high' : 'risk-badge-low'}`}>
                                {pred.prediction_label}
                              </span>
                            </td>
                            <td className="py-2.5 pr-3">
                              <span
                                className="inline-block whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold"
                                style={{
                                  color: pred.risk_band.color,
                                  borderColor: `${pred.risk_band.color}55`,
                                  backgroundColor: `${pred.risk_band.color}1A`,
                                }}
                              >
                                {pred.risk_band.label}
                              </span>
                            </td>
                            <td className="py-2.5 text-slate-300">
                              {pred.support_plan?.[0]
                                ? `${pred.support_plan[0].category}: ${pred.support_plan[0].advisory_action}`
                                : pred.risk_band.action}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </WorldTransition>
  );
};
