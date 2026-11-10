import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Gem,
  ShieldAlert,
  Lock,
  Eye,
  Scale,
  CheckCircle,
  FileSpreadsheet,
  Download,
  ArrowUpRight,
} from 'lucide-react';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WorldTransition } from '../components/WorldTransition';
import { LoadingSpell } from '../components/LoadingSpell';
import { ErrorPortal } from '../components/ErrorPortal';
import { AdvisoryBanner } from '../components/AdvisoryBanner';
import {
  extractApiError,
  fetchAuditLogs,
  fetchInterventions,
  fetchModelInfo,
  fetchStudentReports,
  getStudentReportsCsvUrl,
} from '../api/client';
import type {
  AuditLogs,
  InterventionsResponse,
  ModelInfo,
  StudentReportsResponse,
} from '../types';

export const World6Ocean: React.FC = () => {
  const navigate = useNavigate();
  const [tab, setTab] = useState<'governance' | 'reports'>('governance');

  const [audit, setAudit] = useState<AuditLogs | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [interventions, setInterventions] = useState<InterventionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Reports Center State
  const [reportPeriod, setReportPeriod] = useState<number>(4);
  const [reportTier, setReportTier] = useState<string>('ALL');
  const [reportsData, setReportsData] = useState<StudentReportsResponse | null>(null);
  const [loadingReports, setLoadingReports] = useState(false);
  const [reportsError, setReportsError] = useState<string | null>(null);

  const loadGovernance = async () => {
    setLoading(true);
    setError(null);
    try {
      const [a, m, iv] = await Promise.all([
        fetchAuditLogs(),
        fetchModelInfo(),
        fetchInterventions(),
      ]);
      setAudit(a);
      setModelInfo(m);
      setInterventions(iv);
    } catch (err) {
      setError(extractApiError(err, 'Unable to unlock Oceanic Governance Archive.'));
    } finally {
      setLoading(false);
    }
  };

  const loadReports = async (period = reportPeriod, tier = reportTier) => {
    setLoadingReports(true);
    setReportsError(null);
    try {
      const res = await fetchStudentReports({
        period,
        risk_tier: tier === 'ALL' ? undefined : tier,
        limit: 100,
      });
      setReportsData(res);
    } catch (err) {
      setReportsError(extractApiError(err, 'Unable to compile student risk reports.'));
    } finally {
      setLoadingReports(false);
    }
  };

  useEffect(() => {
    loadGovernance();
    loadReports(4, 'ALL');
  }, []);

  useEffect(() => {
    if (tab === 'reports') {
      loadReports(reportPeriod, reportTier);
    }
  }, [tab, reportPeriod, reportTier]);

  if (loading) {
    return (
      <div className="relative min-h-screen bg-[#030D17]">
        <WorldBackdrop variant="ocean" />
        <LoadingSpell message="Unsealing Crystal Governance Vault..." accentColor="#5DE6E0" />
      </div>
    );
  }

  if (error || !audit || !modelInfo) {
    return (
      <div className="relative min-h-screen bg-[#030D17] p-8">
        <WorldBackdrop variant="ocean" />
        <ErrorPortal message={error || 'Governance ledger unavailable.'} onRetry={loadGovernance} />
      </div>
    );
  }

  const genParams = modelInfo.data_generator_parameters;
  const csvDownloadUrl = getStudentReportsCsvUrl({
    period: reportPeriod,
    risk_tier: reportTier === 'ALL' ? undefined : reportTier,
    limit: 500,
  });

  const temporalTest = modelInfo.held_out_test_metrics_temporal;
  const disjointTest = modelInfo.held_out_test_metrics_disjoint;

  return (
    <WorldTransition worldKey="world-6">
      <div className="relative min-h-screen bg-[#030D17] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="ocean" />
        <ParticleField theme="ocean" density={50} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/35 bg-cyan-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-cyan-200">
                <Gem className="h-3.5 w-3.5" />
                World 6 · Responsible AI, Governance & Export Center
              </div>
              <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-[#A8E6E3] via-[#5DE6E0] to-[#00FFD4] bg-clip-text sm:text-4xl">
                The Oceanic Crystal Realm
              </h1>
              <p className="mt-1 text-sm text-cyan-100/75">
                Responsible AI covenants, feature exclusion vault, validation ledger, and downloadable CSV risk reports.
              </p>
            </div>

            <div className="flex rounded-xl border border-cyan-300/25 bg-[#071B2E]/85 p-1 backdrop-blur-md">
              <button
                onClick={() => setTab('governance')}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
                  tab === 'governance'
                    ? 'border border-cyan-300/45 bg-cyan-500/20 text-cyan-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Lock className="h-3.5 w-3.5" />
                Governance & Exclusions
              </button>
              <button
                onClick={() => setTab('reports')}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition ${
                  tab === 'reports'
                    ? 'border border-emerald-300/45 bg-emerald-500/20 text-emerald-200'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <FileSpreadsheet className="h-3.5 w-3.5" />
                Reports & CSV Exports
              </button>
            </div>
          </div>

          <AdvisoryBanner text={modelInfo.responsible_ai_disclaimer} accentColor="#5DE6E0" />

          {tab === 'governance' ? (
            <>
              {/* Four Covenant Pillars */}
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/80 p-6 backdrop-blur-xl">
                  <div className="flex items-center gap-2.5 text-cyan-300">
                    <Eye className="h-5 w-5" />
                    <h2 className="font-fantasy text-lg font-bold text-white">
                      1. Data Provenance & Synthetic Boundaries
                    </h2>
                  </div>
                  <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-300">
                    <li>
                      • <b className="text-cyan-200">100% Synthetic Cohort:</b>{' '}
                      {genParams
                        ? `${genParams.num_students.toLocaleString()} simulated school students, ${genParams.total_snapshots.toLocaleString()} snapshots (seed=${genParams.random_seed}).`
                        : 'Deterministic synthetic school cohort simulation.'}
                    </li>
                    <li>
                      • <b className="text-cyan-200">Zero Personal Data:</b> Eliminates student privacy exposure under FERPA and GDPR mandates.
                    </li>
                    <li>
                      • <b className="text-cyan-200">Unverified Real-World Generalization:</b> Performance metrics validate consistency on synthetic trajectories only and must be re-calibrated before any live school pilot.
                    </li>
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/80 p-6 backdrop-blur-xl">
                  <div className="flex items-center gap-2.5 text-emerald-300">
                    <CheckCircle className="h-5 w-5" />
                    <h2 className="font-fantasy text-lg font-bold text-white">
                      2. Human-in-the-Loop Mandate
                    </h2>
                  </div>
                  <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-300">
                    <li>
                      • <b className="text-emerald-200">Teacher Contextual Review:</b> Predictions serve strictly as early warning cues for teachers and academic coordinators.
                    </li>
                    <li>
                      • <b className="text-emerald-200">Zero Automated Sanctions:</b> System MUST NEVER assign grades, suspend students, lock classes, or automate disciplinary action.
                    </li>
                    <li>
                      • <b className="text-emerald-200">Support-First Guidance:</b> Recommendations focus exclusively on tutoring, homework check-ins, and caring parent/teacher dialogue.
                    </li>
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/80 p-6 backdrop-blur-xl">
                  <div className="flex items-center gap-2.5 text-amber-300">
                    <Scale className="h-5 w-5" />
                    <h2 className="font-fantasy text-lg font-bold text-white">
                      3. Error Cost, Calibration & Threshold Governance
                    </h2>
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-slate-300">
                    {audit.validation_ledger.threshold_rationale}
                  </p>
                  <div className="mt-4 grid grid-cols-3 gap-2 font-mono text-xs sm:grid-cols-6">
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Threshold</p>
                      <p className="mt-0.5 font-bold text-cyan-300">{audit.validation_ledger.selected_threshold}</p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Val Recall</p>
                      <p className="mt-0.5 font-bold text-emerald-300">
                        {(audit.validation_ledger.safety_net_recall * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Val Prec</p>
                      <p className="mt-0.5 font-bold text-amber-300">
                        {(audit.validation_ledger.educator_precision * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Val AP</p>
                      <p className="mt-0.5 font-bold text-cyan-200">
                        {audit.validation_ledger.average_precision !== undefined
                          ? audit.validation_ledger.average_precision.toFixed(3)
                          : 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Val Spec</p>
                      <p className="mt-0.5 font-bold text-sky-300">
                        {audit.validation_ledger.specificity !== undefined
                          ? `${(audit.validation_ledger.specificity * 100).toFixed(1)}%`
                          : 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-2 text-center">
                      <p className="text-[10px] text-slate-400">Val Brier</p>
                      <p className="mt-0.5 font-bold text-teal-200">
                        {audit.validation_ledger.brier_score !== undefined
                          ? audit.validation_ledger.brier_score.toFixed(4)
                          : 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/80 p-6 backdrop-blur-xl">
                  <div className="flex items-center gap-2.5 text-rose-300">
                    <ShieldAlert className="h-5 w-5" />
                    <h2 className="font-fantasy text-lg font-bold text-white">
                      4. Compliance Ledger Status
                    </h2>
                  </div>
                  <div className="mt-3 space-y-2 font-mono text-xs">
                    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <span className="text-slate-300">Active Champion Model</span>
                      <span className="text-cyan-200">
                        {audit.responsible_ai_policy.model_name || modelInfo.champion_architecture}{' '}
                        {audit.responsible_ai_policy.is_calibrated ? '(Calibrated)' : ''}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <span className="text-slate-300">Validation Protocol</span>
                      <span className="text-right text-cyan-200">{audit.validation_ledger.validation_methodology}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <span className="text-slate-300">Data Quality Audit</span>
                      <span className="text-emerald-300">{modelInfo.data_quality_status || 'PASS'}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <span className="text-slate-300">Version & Audit Time</span>
                      <span className="text-slate-400">
                        v{audit.responsible_ai_policy.model_version} · {audit.generated_timestamp.slice(0, 19)}Z
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Crystal Feature Exclusion Registry */}
              <div className="mt-6 rounded-2xl border border-cyan-300/25 bg-[#071B2E]/85 p-6 backdrop-blur-xl">
                <div className="mb-4 flex items-center gap-2.5">
                  <Lock className="h-5 w-5 text-cyan-300" />
                  <div>
                    <h2 className="font-fantasy text-lg font-bold text-cyan-200">
                      Strict Feature Exclusion Crystal Registry
                    </h2>
                    <p className="text-xs text-slate-400">
                      Protected attributes and post-assessment outcomes hermetically sealed out of the model pipeline
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
                  {audit.feature_exclusion_registry.map((entry, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-cyan-300/20 bg-slate-950/60 p-4 transition hover:border-cyan-300/45"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-bold text-white">{entry.attribute}</span>
                        <span className="rounded-full border border-rose-400/40 bg-rose-500/15 px-2.5 py-0.5 font-mono text-[10px] font-bold text-rose-300">
                          {entry.status}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-slate-400">{entry.rationale}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            /* REPORTS & CSV EXPORTS CENTER */
            <div className="space-y-6">
              {/* Filter Bar + CSV Download Trigger */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-cyan-300/25 bg-[#071B2E]/85 p-5 backdrop-blur-xl">
                <div className="flex flex-wrap items-center gap-3">
                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      Assessment Period
                    </label>
                    <select
                      aria-label="Report Assessment Period"
                      value={reportPeriod}
                      onChange={(e) => setReportPeriod(Number(e.target.value))}
                      className="mt-1 rounded-xl border border-cyan-300/30 bg-slate-950 px-3 py-2 text-xs text-white"
                    >
                      {[1, 2, 3, 4].map((p) => (
                        <option key={p} value={p}>
                          Period {p}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      Risk Tier Filter
                    </label>
                    <select
                      aria-label="Report Risk Tier"
                      value={reportTier}
                      onChange={(e) => setReportTier(e.target.value)}
                      className="mt-1 rounded-xl border border-cyan-300/30 bg-slate-950 px-3 py-2 text-xs text-white"
                    >
                      <option value="ALL">All Tiers</option>
                      <option value="HIGH_RISK">High Risk</option>
                      <option value="MODERATE_RISK">Moderate Risk</option>
                      <option value="LOW_RISK">Low Risk</option>
                    </select>
                  </div>
                </div>

                <a
                  href={csvDownloadUrl}
                  download={`eduguard_period_${reportPeriod}_risk_report.csv`}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#5DE6E0] to-emerald-400 px-5 py-2.5 font-fantasy text-xs font-bold uppercase tracking-wider text-slate-950 shadow-lg transition hover:brightness-110"
                >
                  <Download className="h-4 w-4" />
                  Download CSV Report
                </a>
              </div>

              {/* Assessment-Period Report & Intervention/Evaluation Summaries */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                {reportsData && (
                  <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/85 p-5 backdrop-blur-xl lg:col-span-7">
                    <h2 className="font-fantasy text-base font-bold text-cyan-200">
                      Assessment-Period Cohort Summary
                    </h2>
                    <div className="mt-3 overflow-x-auto">
                      <table className="w-full text-left font-mono text-xs">
                        <thead>
                          <tr className="border-b border-white/10 text-slate-400">
                            <th className="py-2">Period</th>
                            <th>Students</th>
                            <th>Flagged</th>
                            <th>At-Risk</th>
                            <th>Att %</th>
                            <th>Score</th>
                            <th>Mean Risk</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-slate-200">
                          {reportsData.period_reports.map((pr) => (
                            <tr
                              key={pr.period}
                              className={pr.period === reportPeriod ? 'bg-cyan-500/10 text-cyan-200' : ''}
                            >
                              <td className="py-2 font-bold">Period {pr.period}</td>
                              <td>{pr.student_count.toLocaleString()}</td>
                              <td>{pr.flagged_count.toLocaleString()}</td>
                              <td>{pr.actual_at_risk_count.toLocaleString()}</td>
                              <td>{pr.avg_attendance}%</td>
                              <td>{pr.avg_score}</td>
                              <td>{pr.avg_risk_probability_pct}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                <div className="space-y-4 lg:col-span-5">
                  <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/85 p-5 backdrop-blur-xl">
                    <h2 className="font-fantasy text-base font-bold text-emerald-200">
                      Intervention & Model Evaluation Summary
                    </h2>
                    {interventions && (
                      <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-[11px]">
                        {Object.entries(interventions.status_summary).map(([st, cnt]) => (
                          <div key={st} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                            <p className="text-slate-400">{st}</p>
                            <p className="font-bold text-white">
                              {cnt} {cnt === 1 ? 'student' : 'students'}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="mt-3 space-y-1.5 rounded-xl border border-white/10 bg-slate-950/60 p-3 font-mono text-[11px] text-slate-300">
                      <p>
                        <b className="text-cyan-200">Temporal Test (P4):</b> AP{' '}
                        {(temporalTest.average_precision ?? temporalTest.pr_auc).toFixed(3)} · Recall{' '}
                        {(temporalTest.recall * 100).toFixed(1)}% · Prec{' '}
                        {(temporalTest.precision * 100).toFixed(1)}% · Brier{' '}
                        {temporalTest.brier_score.toFixed(4)}
                      </p>
                      <p>
                        <b className="text-violet-200">Disjoint Test:</b> AP{' '}
                        {(disjointTest.average_precision ?? disjointTest.pr_auc).toFixed(3)} · Recall{' '}
                        {(disjointTest.recall * 100).toFixed(1)}% · Prec{' '}
                        {(disjointTest.precision * 100).toFixed(1)}% · Brier{' '}
                        {disjointTest.brier_score.toFixed(4)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {reportsError && (
                <ErrorPortal
                  title="Student Report Load Error"
                  message={reportsError}
                  onRetry={() => loadReports(reportPeriod, reportTier)}
                />
              )}

              {loadingReports ? (
                <LoadingSpell message="Compiling Period Risk Report..." accentColor="#5DE6E0" />
              ) : (
                reportsData && (
                  <div className="rounded-2xl border border-cyan-300/25 bg-[#071B2E]/85 p-6 backdrop-blur-xl">
                    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h2 className="font-fantasy text-base font-bold text-cyan-200">
                          Student Risk Report — Period {reportsData.period} (Showing {reportsData.returned_records} of{' '}
                          {reportsData.total_matching_records.toLocaleString()})
                        </h2>
                        <p className="text-xs text-slate-400">
                          Decision threshold τ* = {reportsData.applied_threshold} · High Risk:{' '}
                          {reportsData.tier_distribution.HIGH_RISK ?? 0} · Moderate:{' '}
                          {reportsData.tier_distribution.MODERATE_RISK ?? 0} · Low:{' '}
                          {reportsData.tier_distribution.LOW_RISK ?? 0}
                        </p>
                      </div>
                    </div>

                    <div className="max-h-[460px] overflow-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="sticky top-0 bg-[#071B2E]">
                          <tr className="border-b border-white/15 font-mono text-slate-400">
                            <th className="py-2 pr-3">Student</th>
                            <th className="pr-3">Score (t)</th>
                            <th className="pr-3">Rolling Mean</th>
                            <th className="pr-3">Att %</th>
                            <th className="pr-3">Risk %</th>
                            <th className="pr-3">Tier</th>
                            <th className="pr-3">Rule-Based Support</th>
                            <th>Intervention</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-slate-200">
                          {reportsData.records.map((row) => (
                            <tr key={`${row.student_id}-${row.assessment_period}`}>
                              <td className="py-2 pr-3 font-mono font-bold">
                                <button
                                  onClick={() => navigate(`/academy?student=${row.student_id}`)}
                                  className="inline-flex items-center gap-1 text-amber-300 hover:underline"
                                >
                                  {row.student_id}
                                  <ArrowUpRight className="h-3 w-3" />
                                </button>
                              </td>
                              <td className="py-2 pr-3 font-mono">{row.previous_score.toFixed(1)}</td>
                              <td className="py-2 pr-3 font-mono">{row.previous_score_avg.toFixed(1)}</td>
                              <td className="py-2 pr-3 font-mono">
                                {row.attendance_pct !== null ? `${row.attendance_pct.toFixed(1)}%` : 'N/A'}
                              </td>
                              <td className="py-2 pr-3 font-mono font-bold">
                                {(row.risk_probability * 100).toFixed(1)}%
                              </td>
                              <td className="py-2 pr-3">
                                <span
                                  className={
                                    row.risk_tier === 'HIGH_RISK'
                                      ? 'risk-badge-high'
                                      : row.risk_tier === 'MODERATE_RISK'
                                        ? 'rounded-full border border-amber-400/40 bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold text-amber-200'
                                        : 'risk-badge-low'
                                  }
                                >
                                  {row.risk_tier.replace('_', ' ')}
                                </span>
                              </td>
                              <td className="py-2 pr-3 text-slate-300">{row.recommended_support_category}</td>
                              <td className="py-2 font-mono text-[11px] text-cyan-200">{row.intervention_status}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )
              )}
            </div>
          )}
        </div>
      </div>
    </WorldTransition>
  );
};
