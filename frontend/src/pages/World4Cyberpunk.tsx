import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Cpu,
  Sliders,
  Gauge,
  Activity,
  Zap,
  Layers,
  ClipboardCheck,
  Save,
  ArrowUpRight,
  Plus,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WorldTransition } from '../components/WorldTransition';
import { MetricCard } from '../components/MetricCard';
import { RiskGauge } from '../components/RiskGauge';
import { LoadingSpell } from '../components/LoadingSpell';
import { ErrorPortal } from '../components/ErrorPortal';
import { AdvisoryBanner } from '../components/AdvisoryBanner';
import {
  extractApiError,
  fetchInterventions,
  fetchModelInfo,
  predictStudent,
  saveIntervention,
} from '../api/client';
import { computeDerivedFeatures, withDerivedFeatures } from '../utils/features';
import type {
  InterventionPriority,
  InterventionRecord,
  InterventionStatus,
  InterventionsResponse,
  ModelInfo,
  PredictionResponse,
  StudentFeatureInput,
} from '../types';

const STATUS_OPTIONS: InterventionStatus[] = [
  'Pending Review',
  'Outreach Scheduled',
  'Mentoring Active',
  'Resolved',
];

export const World4Cyberpunk: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initialTab = searchParams.get('tab');

  const [activeTab, setActiveTab] = useState<'observatory' | 'whatif' | 'interventions'>(
    initialTab === 'interventions'
      ? 'interventions'
      : initialTab === 'whatif'
        ? 'whatif'
        : 'observatory'
  );
  const [splitView, setSplitView] = useState<'temporal' | 'disjoint'>('temporal');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // What-If Simulator State
  const [baseInput, setBaseInput] = useState<StudentFeatureInput>({
    attendance_pct: 62,
    assignment_completion_pct: 50,
    late_assignment_count: 3,
    previous_score: 36,
    previous_score_2: 44,
    assessment_period: 2,
  });

  const [hypoInput, setHypoInput] = useState<StudentFeatureInput>({
    attendance_pct: 88,
    assignment_completion_pct: 88,
    late_assignment_count: 0,
    previous_score: 58,
    previous_score_2: 36,
    assessment_period: 2,
  });

  const [basePred, setBasePred] = useState<PredictionResponse | null>(null);
  const [hypoPred, setHypoPred] = useState<PredictionResponse | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  // Intervention Center State
  const [interventionsData, setInterventionsData] = useState<InterventionsResponse | null>(null);
  const [loadingInterventions, setLoadingInterventions] = useState(false);
  const [interventionsError, setInterventionsError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [draftStatuses, setDraftStatuses] = useState<Record<string, InterventionStatus>>({});
  const [draftNotes, setDraftNotes] = useState<Record<string, string>>({});
  const [savingId, setSavingId] = useState<string | null>(null);
  const [savedToast, setSavedToast] = useState<string | null>(null);
  const [newStudentId, setNewStudentId] = useState('');
  const [newStudentNote, setNewStudentNote] = useState('');

  const loadModel = async () => {
    setLoading(true);
    setError(null);
    try {
      const m = await fetchModelInfo();
      setModelInfo(m);
    } catch (err) {
      setError(extractApiError(err, 'Unable to reach Cyberpunk AI Observatory.'));
    } finally {
      setLoading(false);
    }
  };

  const loadInterventions = async () => {
    setLoadingInterventions(true);
    setInterventionsError(null);
    try {
      const res = await fetchInterventions();
      setInterventionsData(res);
      const nextStatuses: Record<string, InterventionStatus> = {};
      const nextNotes: Record<string, string> = {};
      for (const rec of res.interventions) {
        nextStatuses[rec.student_id] = rec.status;
        nextNotes[rec.student_id] = rec.teacher_notes || '';
      }
      setDraftStatuses(nextStatuses);
      setDraftNotes(nextNotes);
    } catch (err) {
      setInterventionsError(extractApiError(err, 'Unable to fetch teacher intervention queue.'));
    } finally {
      setLoadingInterventions(false);
    }
  };

  const runWhatIfSimulation = async (b: StudentFeatureInput, h: StudentFeatureInput) => {
    setSimulating(true);
    setSimError(null);
    try {
      const [resB, resH] = await Promise.all([
        predictStudent(withDerivedFeatures(b)),
        predictStudent(withDerivedFeatures(h)),
      ]);
      setBasePred(resB);
      setHypoPred(resH);
    } catch (err) {
      setSimError(extractApiError(err, 'What-If simulation request failed. Check input values or API health.'));
    } finally {
      setSimulating(false);
    }
  };

  const handleSaveFollowUp = async (rec: InterventionRecord) => {
    setSavingId(rec.student_id);
    setSavedToast(null);
    setInterventionsError(null);
    try {
      await saveIntervention({
        student_id: rec.student_id,
        status: draftStatuses[rec.student_id] || rec.status,
        priority: rec.priority as InterventionPriority,
        teacher_notes: draftNotes[rec.student_id] ?? rec.teacher_notes ?? '',
        recommended_support: rec.recommended_support,
        assessment_period: rec.assessment_period,
        risk_probability: rec.risk_probability,
      });
      await loadInterventions();
      setSavedToast(`Saved follow-up for ${rec.student_id}.`);
    } catch (err) {
      setInterventionsError(extractApiError(err, `Failed saving notes for ${rec.student_id}.`));
    } finally {
      setSavingId(null);
    }
  };

  const handleAddManualIntervention = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = newStudentId.trim().toUpperCase();
    if (!cleaned || cleaned.length < 3) return;
    setSavingId(cleaned);
    setInterventionsError(null);
    try {
      await saveIntervention({
        student_id: cleaned,
        status: 'Outreach Scheduled',
        priority: 'High',
        teacher_notes: newStudentNote.trim(),
        recommended_support: 'Teacher-Initiated Academic Check-in (Manual Workflow)',
        assessment_period: 4,
      });
      setNewStudentId('');
      setNewStudentNote('');
      await loadInterventions();
      setSavedToast(`Added ${cleaned} to Intervention Center.`);
    } catch (err) {
      setInterventionsError(extractApiError(err, `Unable to track student ${cleaned}.`));
    } finally {
      setSavingId(null);
    }
  };

  useEffect(() => {
    loadModel();
    runWhatIfSimulation(baseInput, hypoInput);
    loadInterventions();
  }, []);

  if (loading) {
    return (
      <div className="relative min-h-screen bg-[#050814]">
        <WorldBackdrop variant="cyberpunk" />
        <LoadingSpell message="Booting Cyberpunk Observatory..." accentColor="#00F5FF" />
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="relative min-h-screen bg-[#050814] p-8">
        <WorldBackdrop variant="cyberpunk" />
        <ErrorPortal message={error || 'Model telemetry unreachable.'} onRetry={loadModel} />
      </div>
    );
  }

  const activeMetrics =
    splitView === 'temporal'
      ? modelInfo.held_out_test_metrics_temporal
      : modelInfo.held_out_test_metrics_disjoint;

  const cm = activeMetrics.confusion_matrix;
  const championName = modelInfo.model_name || modelInfo.champion_architecture;
  const championBenchmark = modelInfo.model_benchmarks?.[championName];
  const latencyText = championBenchmark
    ? `${championBenchmark.inference_latency_ms.toFixed(3)} ms / record latency`
    : 'Pipeline Inference Ready';

  const calAnalysis = modelInfo.calibration_analysis;
  const calBefore = calAnalysis?.temporal_test_before;
  const calAfter = calAnalysis?.temporal_test_after;

  const benchmarkRows = modelInfo.model_benchmarks
    ? Object.entries(modelInfo.model_benchmarks).map(([name, entry]) => ({
        rawName: name,
        name: name.replace('Classifier', ''),
        ap: Number((entry.temporal_validation.average_precision ?? entry.temporal_validation.pr_auc).toFixed(3)),
        pr_auc: Number(entry.temporal_validation.pr_auc.toFixed(3)),
        f1: Number(entry.temporal_validation.f1_score.toFixed(3)),
        recall: Number((entry.temporal_validation.recall * 100).toFixed(1)),
        specificity: Number(((entry.temporal_validation.specificity ?? 0) * 100).toFixed(1)),
        precision: Number((entry.temporal_validation.precision * 100).toFixed(1)),
        brier: Number(entry.temporal_validation.brier_score.toFixed(4)),
        latency_ms: entry.inference_latency_ms,
      }))
    : [];

  const sweepChartData = (modelInfo.threshold_optimization_sweep || []).map((row) => ({
    threshold: row.threshold,
    precision: Number((row.precision * 100).toFixed(1)),
    recall: Number((row.recall * 100).toFixed(1)),
    f2: Number((row.f2_score * 100).toFixed(1)),
  }));

  const calibrationChartData = (calAfter?.bins || []).map((b) => ({
    predicted: Number((b.mean_predicted_prob * 100).toFixed(1)),
    observed: Number((b.empirical_positive_rate * 100).toFixed(1)),
    perfect: Number((b.mean_predicted_prob * 100).toFixed(1)),
    count: b.count,
  }));

  const periodBreakdownRows = modelInfo.disjoint_test_by_period
    ? Object.entries(modelInfo.disjoint_test_by_period).map(([period, m]) => ({
        period: period.replace(/^period_?/i, 'Period '),
        n: m.sample_size,
        ap: (m.average_precision ?? m.pr_auc).toFixed(3),
        recall: `${(m.recall * 100).toFixed(1)}%`,
        precision: `${(m.precision * 100).toFixed(1)}%`,
        specificity: `${((m.specificity ?? 0) * 100).toFixed(1)}%`,
      }))
    : [];

  const errAudit = modelInfo.temporal_test_error_analysis;

  const deltaPct =
    basePred && hypoPred ? (hypoPred.risk_probability - basePred.risk_probability) * 100 : 0;

  const baseDerived = computeDerivedFeatures(baseInput.previous_score, baseInput.previous_score_2);
  const hypoDerived = computeDerivedFeatures(hypoInput.previous_score, hypoInput.previous_score_2);

  const visibleInterventions = (interventionsData?.interventions || []).filter(
    (rec) => statusFilter === 'ALL' || rec.status === statusFilter
  );

  return (
    <WorldTransition worldKey="world-4">
      <div className="relative min-h-screen bg-[#050814] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="cyberpunk" />
        <ParticleField theme="cyberpunk" density={40} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/40 bg-cyan-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-cyan-300">
                <Cpu className="h-3.5 w-3.5" />
                World 4 · Reliability Observatory, What-If & Interventions
              </div>
              <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-[#00F5FF] via-[#56A8F5] to-[#A855F7] bg-clip-text sm:text-4xl">
                The Cyberpunk Future
              </h1>
              <p className="mt-1 text-sm text-cyan-100/75">
                Held-out model benchmarks, probability calibration, counterfactual sensitivity probes, and teacher intervention tracking.
              </p>
            </div>

            <div className="flex flex-wrap rounded-xl border border-cyan-400/30 bg-slate-950/80 p-1 backdrop-blur-md">
              <button
                onClick={() => setActiveTab('observatory')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  activeTab === 'observatory'
                    ? 'border border-cyan-400/50 bg-cyan-500/20 text-cyan-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Activity className="h-3.5 w-3.5" />
                Reliability Observatory
              </button>
              <button
                onClick={() => setActiveTab('whatif')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  activeTab === 'whatif'
                    ? 'border border-violet-400/50 bg-violet-500/25 text-violet-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                What-If Simulator
              </button>
              <button
                onClick={() => setActiveTab('interventions')}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                  activeTab === 'interventions'
                    ? 'border border-emerald-400/50 bg-emerald-500/20 text-emerald-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <ClipboardCheck className="h-3.5 w-3.5" />
                Intervention Center
              </button>
            </div>
          </div>

          <AdvisoryBanner
            text={
              activeTab === 'observatory'
                ? `Observatory Integrity: Zero fabricated metrics. Showing verified held-out metrics for champion ${championName}${modelInfo.is_calibrated ? ' (Sigmoid Calibrated)' : ''}.`
                : activeTab === 'whatif'
                  ? `Non-Causal Disclaimer: The What-If Simulator executes the real ${championName} pipeline to measure predictive sensitivity, NOT causal effect guarantees.`
                  : 'Teacher Workflow Policy: Recommendations shown here are heuristic rules paired with ML prioritization. All outreach decisions remain strictly human-led.'
            }
            accentColor="#00F5FF"
          />

          {activeTab === 'observatory' ? (
            <>
              {/* Split Switcher */}
              <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <Layers className="h-4 w-4 text-cyan-400" />
                  <span>Evaluation Split Regime:</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSplitView('temporal')}
                    className={`rounded-lg border px-3.5 py-1.5 font-mono text-xs transition ${
                      splitView === 'temporal'
                        ? 'border-cyan-400 bg-cyan-500/20 text-cyan-200'
                        : 'border-white/15 bg-white/5 text-slate-400 hover:text-white'
                    }`}
                  >
                    Temporal Held-Out Test (Period 4)
                  </button>
                  <button
                    onClick={() => setSplitView('disjoint')}
                    className={`rounded-lg border px-3.5 py-1.5 font-mono text-xs transition ${
                      splitView === 'disjoint'
                        ? 'border-violet-400 bg-violet-500/20 text-violet-200'
                        : 'border-white/15 bg-white/5 text-slate-400 hover:text-white'
                    }`}
                  >
                    Student-Disjoint Test (Unseen Students)
                  </button>
                </div>
              </div>

              {/* KPI Strip */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  title="Average Precision (AP)"
                  value={(activeMetrics.average_precision ?? activeMetrics.pr_auc).toFixed(3)}
                  subtitle={`Trap PR-AUC: ${activeMetrics.pr_auc.toFixed(3)} | ROC-AUC: ${activeMetrics.roc_auc.toFixed(3)}`}
                  badgeText="Class-Imbalance Benchmark"
                  badgeColor="bg-cyan-500/15 text-cyan-300 border-cyan-400/30"
                  accentColor="#00F5FF"
                  icon={<Gauge className="h-5 w-5" />}
                />
                <MetricCard
                  title="Safety-Net Recall"
                  value={`${(activeMetrics.recall * 100).toFixed(1)}%`}
                  subtitle={`Specificity: ${((activeMetrics.specificity ?? 0) * 100).toFixed(1)}% · ${cm.true_positives}/${cm.true_positives + cm.false_negatives} TP`}
                  badgeText="F2-Score Priority"
                  badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-400/30"
                  accentColor="#10B981"
                  icon={<Activity className="h-5 w-5" />}
                />
                <MetricCard
                  title="Educator Precision"
                  value={`${(activeMetrics.precision * 100).toFixed(1)}%`}
                  subtitle={`F1: ${activeMetrics.f1_score.toFixed(3)} | F2: ${activeMetrics.f2_score.toFixed(3)} | τ*=${activeMetrics.threshold}`}
                  badgeText={`Accuracy: ${(activeMetrics.accuracy * 100).toFixed(1)}%`}
                  badgeColor="bg-violet-500/15 text-violet-300 border-violet-400/30"
                  accentColor="#8A2BE2"
                  icon={<Zap className="h-5 w-5" />}
                />
                <MetricCard
                  title="Calibration (Brier)"
                  value={activeMetrics.brier_score.toFixed(4)}
                  subtitle={
                    splitView === 'temporal' && calBefore && calAfter
                      ? `Uncalibrated ${calBefore.brier_score.toFixed(4)} → ${calAfter.brier_score.toFixed(4)} (ECE ${calAfter.expected_calibration_error.toFixed(3)})`
                      : calAnalysis?.disjoint_val_brier_after !== undefined
                        ? `Disjoint Val Brier: ${calAnalysis.disjoint_val_brier_before?.toFixed(4)} → ${calAnalysis.disjoint_val_brier_after.toFixed(4)}`
                        : 'Validation-split Sigmoid calibrated'
                  }
                  badgeText={latencyText}
                  badgeColor="bg-sky-500/15 text-sky-300 border-sky-400/30"
                  accentColor="#56A8F5"
                  icon={<Cpu className="h-5 w-5" />}
                />
              </div>

              {/* Confusion Matrix & Multi-Model Benchmark Chart */}
              <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
                <div className="rounded-2xl border border-cyan-400/25 bg-slate-950/75 p-6 backdrop-blur-xl lg:col-span-5">
                  <h2 className="font-fantasy text-lg font-bold text-cyan-200">
                    Confusion Matrix ({splitView === 'temporal' ? 'Temporal Test' : 'Disjoint Test'})
                  </h2>
                  <p className="text-xs text-slate-400">
                    N = {activeMetrics.sample_size.toLocaleString()} held-out test records at τ* = {activeMetrics.threshold}
                  </p>

                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-emerald-400/30 bg-emerald-950/35 p-4 text-center">
                      <p className="font-mono text-[10px] uppercase text-emerald-300">True Negatives</p>
                      <p className="mt-1 font-mono text-2xl font-bold text-white">
                        {cm.true_negatives.toLocaleString()}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-400">Correct Passing</p>
                    </div>
                    <div className="rounded-xl border border-amber-400/30 bg-amber-950/30 p-4 text-center">
                      <p className="font-mono text-[10px] uppercase text-amber-300">False Positives</p>
                      <p className="mt-1 font-mono text-2xl font-bold text-white">
                        {cm.false_positives.toLocaleString()}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-400">Preventive Check-ins</p>
                    </div>
                    <div className="rounded-xl border border-rose-500/40 bg-rose-950/35 p-4 text-center">
                      <p className="font-mono text-[10px] uppercase text-rose-300">False Negatives</p>
                      <p className="mt-1 font-mono text-2xl font-bold text-white">
                        {cm.false_negatives.toLocaleString()}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-400">Missed Signals</p>
                    </div>
                    <div className="rounded-xl border border-cyan-400/40 bg-cyan-950/40 p-4 text-center">
                      <p className="font-mono text-[10px] uppercase text-cyan-300">True Positives</p>
                      <p className="mt-1 font-mono text-2xl font-bold text-white">
                        {cm.true_positives.toLocaleString()}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-400">Early Detections</p>
                    </div>
                  </div>
                </div>

                {/* Model Comparison Bar + Table */}
                <div className="rounded-2xl border border-cyan-400/25 bg-slate-950/75 p-6 backdrop-blur-xl lg:col-span-7">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="font-fantasy text-lg font-bold text-cyan-200">
                        Multi-Model Validation Benchmarks
                      </h2>
                      <p className="text-xs text-slate-400">
                        Programmatic selection on validation folds (Champion: <b className="text-cyan-300">{championName}</b>)
                      </p>
                    </div>
                  </div>

                  <ResponsiveContainer width="100%" height={190}>
                    <BarChart data={benchmarkRows} margin={{ top: 14, right: 16, left: 0, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="name" stroke="#94A3B8" tick={{ fill: '#CBD5E1', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} stroke="#94A3B8" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(5, 10, 20, 0.96)',
                          border: '1px solid #00F5FF',
                          borderRadius: '10px',
                        }}
                      />
                      <Legend />
                      <Bar dataKey="ap" name="Avg Precision" fill="#00F5FF" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                      <Bar dataKey="f1" name="F1-Score" fill="#8A2BE2" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left font-mono text-xs">
                      <thead>
                        <tr className="border-b border-white/10 text-slate-400">
                          <th className="py-2">Model</th>
                          <th>AP</th>
                          <th>Recall</th>
                          <th>Spec</th>
                          <th>Precision</th>
                          <th>Brier</th>
                          <th>Latency</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5 text-slate-200">
                        {benchmarkRows.map((r) => {
                          const isChampion = r.rawName === championName || r.name === championName;
                          return (
                            <tr
                              key={r.rawName}
                              className={isChampion ? 'bg-cyan-500/15 text-cyan-200' : ''}
                            >
                              <td className="py-2 font-semibold">
                                {r.rawName} {isChampion ? '★' : ''}
                              </td>
                              <td>{r.ap}</td>
                              <td>{r.recall}%</td>
                              <td>{r.specificity}%</td>
                              <td>{r.precision}%</td>
                              <td>{r.brier}</td>
                              <td>{r.latency_ms} ms</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Calibration Curve & Disjoint Period Breakdown */}
              <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
                {calibrationChartData.length > 0 && (
                  <div className="rounded-2xl border border-sky-400/25 bg-slate-950/75 p-6 backdrop-blur-xl lg:col-span-6">
                    <h2 className="font-fantasy text-lg font-bold text-sky-200">
                      Calibrated Reliability Diagram (Temporal Test)
                    </h2>
                    <p className="text-xs text-slate-400">
                      Observed positive fraction vs. calibrated predicted probability bins
                      {calBefore && calAfter
                        ? ` · Brier ${calBefore.brier_score.toFixed(4)} → ${calAfter.brier_score.toFixed(4)} (ECE ${calAfter.expected_calibration_error.toFixed(3)})`
                        : ''}
                    </p>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={calibrationChartData} margin={{ top: 14, right: 20, left: 0, bottom: 6 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="predicted" unit="%" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 100]} unit="%" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(5, 10, 20, 0.96)',
                            border: '1px solid #56A8F5',
                            borderRadius: '10px',
                          }}
                        />
                        <Legend />
                        <Line type="linear" dataKey="perfect" name="Perfect Calibration" stroke="#64748B" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="observed" name="Observed Fraction (%)" stroke="#00F5FF" strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {periodBreakdownRows.length > 0 && (
                  <div className="rounded-2xl border border-violet-400/25 bg-slate-950/75 p-6 backdrop-blur-xl lg:col-span-6">
                    <h2 className="font-fantasy text-lg font-bold text-violet-200">
                      Disjoint Test Stability by Assessment Period
                    </h2>
                    <p className="text-xs text-slate-400">
                      Unseen student cohort breakdown across Periods 1–4 at τ* = {modelInfo.selected_threshold}
                    </p>
                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-left font-mono text-xs">
                        <thead>
                          <tr className="border-b border-white/10 text-slate-400">
                            <th className="py-2">Cycle</th>
                            <th>N</th>
                            <th>AP</th>
                            <th>Recall</th>
                            <th>Precision</th>
                            <th>Spec</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-slate-200">
                          {periodBreakdownRows.map((row) => (
                            <tr key={row.period}>
                              <td className="py-2 font-semibold text-violet-200">{row.period}</td>
                              <td>{row.n}</td>
                              <td>{row.ap}</td>
                              <td>{row.recall}</td>
                              <td>{row.precision}</td>
                              <td>{row.specificity}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {errAudit?.false_negatives && errAudit?.false_positives && (
                      <p className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-[11px] text-slate-300">
                        <b className="text-amber-300">Temporal Test Error Audit:</b>{' '}
                        {errAudit.false_negatives.count} False Negatives ({errAudit.false_negatives.share_pct}% · mean prior{' '}
                        {errAudit.false_negatives.mean_previous_score.toFixed(1)} → next{' '}
                        {errAudit.false_negatives.mean_next_assessment_score.toFixed(1)});{' '}
                        {errAudit.false_positives.count} False Positives (borderline prior{' '}
                        {errAudit.false_positives.mean_previous_score.toFixed(1)} → next{' '}
                        {errAudit.false_positives.mean_next_assessment_score.toFixed(1)}).
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* Threshold Optimization Sweep Curve */}
              {sweepChartData.length > 0 && (
                <div className="mt-6 rounded-2xl border border-violet-400/25 bg-slate-950/75 p-6 backdrop-blur-xl">
                  <h2 className="font-fantasy text-lg font-bold text-violet-200">
                    Validation Threshold Optimization Curve (Precision vs. Recall vs. F2)
                  </h2>
                  <p className="text-xs text-slate-400">{modelInfo.threshold_rationale}</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={sweepChartData} margin={{ top: 14, right: 20, left: 0, bottom: 6 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="threshold" stroke="#94A3B8" />
                      <YAxis domain={[25, 100]} stroke="#94A3B8" tickFormatter={(v: number) => `${v}%`} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(5, 10, 20, 0.96)',
                          border: '1px solid #8A2BE2',
                          borderRadius: '10px',
                        }}
                      />
                      <Legend />
                      <ReferenceLine
                        x={modelInfo.selected_threshold}
                        stroke="#F4C76B"
                        strokeDasharray="4 4"
                        label={{ value: `τ*=${modelInfo.selected_threshold}`, fill: '#F4C76B', fontSize: 11 }}
                      />
                      <Line type="monotone" dataKey="recall" name="Recall (%)" stroke="#10B981" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="precision" name="Precision (%)" stroke="#00F5FF" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="f2" name="F2 Utility Score (%)" stroke="#A855F7" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          ) : activeTab === 'whatif' ? (
            /* WHAT-IF SIMULATOR */
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {/* Baseline Controls */}
                <div className="rounded-2xl border border-rose-400/30 bg-slate-950/80 p-6 backdrop-blur-xl">
                  <h2 className="font-fantasy text-lg font-bold text-rose-300">
                    Baseline Student Telemetry
                  </h2>
                  <div className="mt-4 space-y-3.5 text-xs">
                    {[
                      { label: 'Classroom Attendance (%)', key: 'attendance_pct', min: 0, max: 100, step: 1 },
                      { label: 'Assignment Completion (%)', key: 'assignment_completion_pct', min: 0, max: 100, step: 1 },
                      { label: 'Current Assessment Score', key: 'previous_score', min: 0, max: 100, step: 1 },
                      { label: 'Prior Assessment Score', key: 'previous_score_2', min: 0, max: 100, step: 1 },
                      { label: 'Late Assignment Count', key: 'late_assignment_count', min: 0, max: 20, step: 1 },
                    ].map((ctl) => (
                      <div key={ctl.key}>
                        <div className="flex justify-between">
                          <span className="text-slate-300">{ctl.label}</span>
                          <span className="font-mono font-bold text-rose-300">
                            {baseInput[ctl.key as keyof StudentFeatureInput]}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={ctl.min}
                          max={ctl.max}
                          step={ctl.step}
                          value={Number(baseInput[ctl.key as keyof StudentFeatureInput] ?? 0)}
                          onChange={(e) =>
                            setBaseInput({ ...baseInput, [ctl.key]: Number(e.target.value) })
                          }
                          className="mt-1 w-full accent-rose-400"
                        />
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 flex justify-between rounded-lg border border-rose-400/20 bg-rose-950/25 px-3 py-2 font-mono text-[11px] text-rose-200">
                    <span>Rolling Mean: <b>{baseDerived.previous_score_avg}</b></span>
                    <span>Trend: <b>{baseDerived.score_trend >= 0 ? '+' : ''}{baseDerived.score_trend}</b></span>
                  </div>
                </div>

                {/* Hypothetical Controls */}
                <div className="rounded-2xl border border-cyan-400/35 bg-slate-950/80 p-6 backdrop-blur-xl">
                  <h2 className="font-fantasy text-lg font-bold text-cyan-300">
                    Hypothetical Post-Support Telemetry
                  </h2>
                  <div className="mt-4 space-y-3.5 text-xs">
                    {[
                      { label: 'Post-Support Attendance (%)', key: 'attendance_pct', min: 0, max: 100, step: 1 },
                      { label: 'Post-Support Completion (%)', key: 'assignment_completion_pct', min: 0, max: 100, step: 1 },
                      { label: 'Post-Support Assessment Score', key: 'previous_score', min: 0, max: 100, step: 1 },
                      { label: 'Prior Reference Score', key: 'previous_score_2', min: 0, max: 100, step: 1 },
                      { label: 'Post-Support Late Assignments', key: 'late_assignment_count', min: 0, max: 20, step: 1 },
                    ].map((ctl) => (
                      <div key={ctl.key}>
                        <div className="flex justify-between">
                          <span className="text-slate-300">{ctl.label}</span>
                          <span className="font-mono font-bold text-cyan-300">
                            {hypoInput[ctl.key as keyof StudentFeatureInput]}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={ctl.min}
                          max={ctl.max}
                          step={ctl.step}
                          value={Number(hypoInput[ctl.key as keyof StudentFeatureInput] ?? 0)}
                          onChange={(e) =>
                            setHypoInput({ ...hypoInput, [ctl.key]: Number(e.target.value) })
                          }
                          className="mt-1 w-full accent-cyan-400"
                        />
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 flex justify-between rounded-lg border border-cyan-400/20 bg-cyan-950/25 px-3 py-2 font-mono text-[11px] text-cyan-200">
                    <span>Rolling Mean: <b>{hypoDerived.previous_score_avg}</b></span>
                    <span>Trend: <b>{hypoDerived.score_trend >= 0 ? '+' : ''}{hypoDerived.score_trend}</b></span>
                  </div>
                </div>
              </div>

              <div className="flex justify-center">
                <button
                  onClick={() => runWhatIfSimulation(baseInput, hypoInput)}
                  disabled={simulating}
                  className="rounded-xl bg-gradient-to-r from-[#00F5FF] to-[#8A2BE2] px-8 py-3.5 font-fantasy text-xs font-black uppercase tracking-[0.2em] text-slate-950 shadow-[0_0_35px_rgba(0,245,255,0.45)] transition hover:scale-105 disabled:opacity-60"
                >
                  {simulating ? 'Executing Simulation...' : 'Execute Real-Time What-If Simulation'}
                </button>
              </div>

              {simError && (
                <ErrorPortal
                  title="Simulation Error"
                  message={simError}
                  onRetry={() => runWhatIfSimulation(baseInput, hypoInput)}
                />
              )}

              {basePred && hypoPred && (
                <div className="grid grid-cols-1 gap-6 rounded-2xl border border-cyan-400/30 bg-slate-950/85 p-6 backdrop-blur-xl md:grid-cols-3">
                  <div className="flex flex-col items-center">
                    <p className="mb-2 text-xs font-bold uppercase tracking-widest text-rose-300">
                      Baseline Risk
                    </p>
                    <RiskGauge
                      probability={basePred.risk_probability}
                      threshold={basePred.applied_threshold}
                      riskLabel={basePred.risk_band.label}
                      tier={basePred.risk_band.tier}
                      size={180}
                    />
                  </div>

                  <div className="flex flex-col items-center justify-center text-center">
                    <p className="text-xs font-bold uppercase tracking-widest text-slate-400">
                      Predicted Probability Shift
                    </p>
                    <p
                      className={`mt-2 font-mono text-4xl font-black ${
                        deltaPct <= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {deltaPct > 0 ? '+' : ''}
                      {deltaPct.toFixed(1)}%
                    </p>
                    <p className="mt-3 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs text-slate-200">
                      {basePred.risk_band.label} <span className="text-cyan-400">➔</span>{' '}
                      {hypoPred.risk_band.label}
                    </p>
                    <p className="mt-2 text-[11px] text-slate-400">
                      Model sensitivity estimate only — not a guarantee of assessment score change.
                    </p>
                  </div>

                  <div className="flex flex-col items-center">
                    <p className="mb-2 text-xs font-bold uppercase tracking-widest text-cyan-300">
                      Post-Support Risk
                    </p>
                    <RiskGauge
                      probability={hypoPred.risk_probability}
                      threshold={hypoPred.applied_threshold}
                      riskLabel={hypoPred.risk_band.label}
                      tier={hypoPred.risk_band.tier}
                      size={180}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* INTERVENTION CENTER TAB */
            <div className="space-y-6">
              {interventionsData && (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {STATUS_OPTIONS.map((st) => (
                    <button
                      key={st}
                      onClick={() => setStatusFilter(statusFilter === st ? 'ALL' : st)}
                      className={`rounded-2xl border p-4 text-left backdrop-blur-xl transition ${
                        statusFilter === st
                          ? 'border-cyan-400 bg-cyan-500/15'
                          : 'border-white/15 bg-slate-950/70 hover:border-white/30'
                      }`}
                    >
                      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                        {st}
                      </p>
                      <p className="mt-1 font-mono text-2xl font-bold text-white">
                        {interventionsData.status_summary[st] ?? 0}
                      </p>
                    </button>
                  ))}
                </div>
              )}

              {/* Add Student Note Form & Status Filter */}
              <form
                onSubmit={handleAddManualIntervention}
                className="flex flex-wrap items-end gap-3 rounded-2xl border border-cyan-400/25 bg-slate-950/75 p-4 backdrop-blur-xl"
              >
                <div className="min-w-[160px] flex-1">
                  <label className="text-[11px] font-semibold text-slate-300">
                    Track Student ID
                  </label>
                  <input
                    type="text"
                    value={newStudentId}
                    onChange={(e) => setNewStudentId(e.target.value)}
                    placeholder="e.g. STU_00125"
                    className="mt-1 w-full rounded-xl border border-white/15 bg-slate-900 px-3 py-2 font-mono text-xs text-white focus:border-cyan-400 focus:outline-none"
                  />
                </div>
                <div className="min-w-[240px] flex-[2]">
                  <label className="text-[11px] font-semibold text-slate-300">
                    Initial Teacher Note / Plan
                  </label>
                  <input
                    type="text"
                    value={newStudentNote}
                    onChange={(e) => setNewStudentNote(e.target.value)}
                    placeholder="e.g. Parent check-in & after-school math clinic"
                    className="mt-1 w-full rounded-xl border border-white/15 bg-slate-900 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
                  />
                </div>
                <button
                  type="submit"
                  className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-cyan-400 to-teal-400 px-4 py-2 font-fantasy text-xs font-bold uppercase tracking-wider text-slate-950 transition hover:brightness-110"
                >
                  <Plus className="h-3.5 w-3.5" /> Add Student
                </button>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="rounded-xl border border-white/15 bg-slate-900 px-3 py-2 text-xs text-slate-200"
                >
                  <option value="ALL">All Statuses</option>
                  {STATUS_OPTIONS.map((st) => (
                    <option key={st} value={st}>
                      {st}
                    </option>
                  ))}
                </select>
              </form>

              {savedToast && (
                <div className="rounded-xl border border-emerald-400/40 bg-emerald-950/40 px-4 py-2.5 text-xs text-emerald-200">
                  {savedToast}
                </div>
              )}

              {interventionsError && (
                <ErrorPortal
                  title="Intervention Store Error"
                  message={interventionsError}
                  onRetry={loadInterventions}
                />
              )}

              {loadingInterventions && !interventionsData ? (
                <LoadingSpell message="Loading Teacher Intervention Queue..." accentColor="#00F5FF" />
              ) : (
                <div className="space-y-3">
                  {visibleInterventions.map((rec) => (
                    <div
                      key={rec.student_id}
                      className="rounded-2xl border border-white/15 bg-slate-950/75 p-5 backdrop-blur-xl"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex flex-wrap items-center gap-2.5">
                          <button
                            onClick={() => navigate(`/academy?student=${rec.student_id}`)}
                            className="inline-flex items-center gap-1 font-mono text-sm font-bold text-amber-300 hover:underline"
                          >
                            {rec.student_id}
                            <ArrowUpRight className="h-3.5 w-3.5" />
                          </button>
                          {rec.risk_probability !== null && rec.risk_probability !== undefined && (
                            <span className="rounded-full border border-rose-400/40 bg-rose-500/15 px-2.5 py-0.5 font-mono text-[11px] font-bold text-rose-200">
                              Risk {(rec.risk_probability * 100).toFixed(1)}%
                            </span>
                          )}
                          <span className="rounded-full border border-amber-400/35 bg-amber-500/10 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-amber-200">
                            {rec.priority} · Rule-Based Suggestion
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <select
                            aria-label={`Status for ${rec.student_id}`}
                            value={draftStatuses[rec.student_id] || rec.status}
                            onChange={(e) =>
                              setDraftStatuses({
                                ...draftStatuses,
                                [rec.student_id]: e.target.value as InterventionStatus,
                              })
                            }
                            className="rounded-xl border border-cyan-400/30 bg-slate-900 px-3 py-1.5 text-xs text-cyan-100 focus:border-cyan-400 focus:outline-none"
                          >
                            {STATUS_OPTIONS.map((st) => (
                              <option key={st} value={st}>
                                {st}
                              </option>
                            ))}
                          </select>

                          <button
                            onClick={() => handleSaveFollowUp(rec)}
                            disabled={savingId === rec.student_id}
                            className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-500/20 px-3.5 py-1.5 text-xs font-semibold text-cyan-200 border border-cyan-400/40 transition hover:bg-cyan-500/35 disabled:opacity-50"
                          >
                            <Save className="h-3.5 w-3.5" />
                            {savingId === rec.student_id ? 'Saving...' : 'Save Follow-Up'}
                          </button>
                        </div>
                      </div>

                      <p className="mt-2.5 text-xs text-slate-300">
                        <b className="text-slate-200">Suggested Support:</b> {rec.recommended_support}
                      </p>

                      <div className="mt-3">
                        <textarea
                          rows={2}
                          value={draftNotes[rec.student_id] ?? ''}
                          onChange={(e) =>
                            setDraftNotes({
                              ...draftNotes,
                              [rec.student_id]: e.target.value,
                            })
                          }
                          placeholder="Add teacher mentoring observation, student outreach date, or guardian conversation notes..."
                          className="w-full rounded-xl border border-white/10 bg-slate-900/90 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-400 focus:outline-none"
                        />
                      </div>
                    </div>
                  ))}
                  {visibleInterventions.length === 0 && (
                    <p className="py-8 text-center text-xs text-slate-400">
                      No intervention entries match the selected workflow status.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </WorldTransition>
  );
};
