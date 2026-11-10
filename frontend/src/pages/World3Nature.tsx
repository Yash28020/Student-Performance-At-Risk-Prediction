import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Trees, Droplets, CheckCircle2, Clock, Award, Sparkles, Compass, ArrowRight } from 'lucide-react';
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
  Cell,
  Legend,
} from 'recharts';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WorldTransition } from '../components/WorldTransition';
import { MetricCard } from '../components/MetricCard';
import { LoadingSpell } from '../components/LoadingSpell';
import { ErrorPortal } from '../components/ErrorPortal';
import { AdvisoryBanner } from '../components/AdvisoryBanner';
import { extractApiError, fetchCohortSummary, fetchGlobalAnalytics } from '../api/client';
import { formatFeatureName } from '../constants/worlds';
import type { CohortSummary, GlobalAnalyticsResponse } from '../types';

export const World3Nature: React.FC = () => {
  const navigate = useNavigate();
  const [cohort, setCohort] = useState<CohortSummary | null>(null);
  const [analytics, setAnalytics] = useState<GlobalAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [c, a] = await Promise.all([fetchCohortSummary(), fetchGlobalAnalytics()]);
      setCohort(c);
      setAnalytics(a);
    } catch (err) {
      setError(extractApiError(err, 'Unable to summon Nature Realm cohort data.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="relative min-h-screen bg-[#071510]">
        <WorldBackdrop variant="nature" />
        <LoadingSpell message="Cultivating Cohort Intelligence..." accentColor="#2D9978" />
      </div>
    );
  }

  if (error || !cohort) {
    return (
      <div className="relative min-h-screen bg-[#071510] p-8">
        <WorldBackdrop variant="nature" />
        <ErrorPortal message={error || 'Cohort telemetry unavailable.'} onRetry={loadData} />
      </div>
    );
  }

  const bandColors = ['#F43F5E', '#F59E0B', '#38BDF8', '#2D9978'];
  const shapBars = (analytics?.global_shap_importance || []).map((item) => ({
    feature: formatFeatureName(item.feature),
    importance: item.mean_absolute_shap,
  }));

  return (
    <WorldTransition worldKey="world-3">
      <div className="relative min-h-screen bg-[#071510] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="nature" />
        <ParticleField theme="nature" density={50} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/35 bg-emerald-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-emerald-300">
                <Trees className="h-3.5 w-3.5" />
                World 3 · Cohort Intelligence & Explainability
              </div>
              <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-emerald-200 via-[#5DBB8A] to-teal-300 bg-clip-text sm:text-4xl">
                The Mythical Nature Realm
              </h1>
              <p className="mt-1 text-sm text-emerald-100/75">
                Living ecosystem of anonymized attendance bands, coursework vitality, cohort progression, and statistical feature drivers.
              </p>
            </div>
            <button
              onClick={() => navigate('/celestial')}
              className="inline-flex items-center gap-2 rounded-xl border border-emerald-400/35 bg-emerald-500/15 px-4 py-2 text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/25"
            >
              Explore Celestial SHAP Map
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          <AdvisoryBanner
            text={`Cohort Intelligence Notice: Strictly anonymized macro patterns across ${cohort.total_snapshots.toLocaleString()} school assessment snapshots. Individual student identities remain shielded.`}
            accentColor="#2D9978"
          />

          {/* KPI Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="Mean Classroom Attendance"
              value={`${cohort.avg_attendance}%`}
              subtitle="Cohort participation baseline across Periods 1–4"
              badgeText="Engagement Vitality"
              badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-400/30"
              accentColor="#2D9978"
              icon={<Droplets className="h-5 w-5" />}
              delay={0.05}
            />
            <MetricCard
              title="Assignment Completion"
              value={`${cohort.avg_completion}%`}
              subtitle="Homework & coursework submission follow-through"
              badgeText="Milestone Execution"
              badgeColor="bg-teal-500/15 text-teal-300 border-teal-400/30"
              accentColor="#5DBB8A"
              icon={<CheckCircle2 className="h-5 w-5" />}
              delay={0.1}
            />
            <MetricCard
              title="Mean Late Submissions"
              value={cohort.avg_late_assignments}
              subtitle="Average delayed coursework items per period"
              badgeText="Workload Friction"
              badgeColor="bg-amber-500/15 text-amber-300 border-amber-400/30"
              accentColor="#F4C76B"
              icon={<Clock className="h-5 w-5" />}
              delay={0.15}
            />
            <MetricCard
              title="Mean Assessment Score"
              value={`${cohort.avg_score} / 100`}
              subtitle="Cohort score baseline vs. 40-point proficiency mark"
              badgeText={`${(100 - cohort.at_risk_rate_pct).toFixed(1)}% Passing Rate`}
              badgeColor="bg-emerald-500/15 text-emerald-200 border-emerald-400/30"
              accentColor="#10B981"
              icon={<Award className="h-5 w-5" />}
              delay={0.2}
            />
          </div>

          {/* Tier Risk Charts */}
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-6 backdrop-blur-xl">
              <h2 className="font-fantasy text-lg font-bold text-emerald-200">
                At-Risk Rate (%) Across Attendance Tiers
              </h2>
              <p className="text-xs text-slate-400">
                Low attendance correlates strongly with scoring below 40/100 in the subsequent cycle
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={cohort.attendance_bands} margin={{ top: 18, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="band" stroke="#94A3B8" interval={0} tick={{ fill: '#CBD5E1', fontSize: 10 }} />
                  <YAxis stroke="#94A3B8" tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(7, 21, 16, 0.96)',
                      border: '1px solid #2D9978',
                      borderRadius: '10px',
                    }}
                    formatter={(value: unknown, _name: unknown, props: { payload?: { count?: number } }) => [
                      `${value}% (${props?.payload?.count?.toLocaleString() ?? 0} snapshots)`,
                      'At-Risk Rate',
                    ]}
                  />
                  <Bar dataKey="risk_rate" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                    {cohort.attendance_bands.map((_, idx) => (
                      <Cell key={idx} fill={bandColors[idx % bandColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-6 backdrop-blur-xl">
              <h2 className="font-fantasy text-lg font-bold text-emerald-200">
                At-Risk Rate (%) Across Completion Tiers
              </h2>
              <p className="text-xs text-slate-400">
                Sustained assignment submission correlates with higher next-assessment resilience
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={cohort.completion_bands} margin={{ top: 18, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="band" stroke="#94A3B8" interval={0} tick={{ fill: '#CBD5E1', fontSize: 10 }} />
                  <YAxis stroke="#94A3B8" tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(7, 21, 16, 0.96)',
                      border: '1px solid #2D9978',
                      borderRadius: '10px',
                    }}
                    formatter={(value: unknown, _name: unknown, props: { payload?: { count?: number } }) => [
                      `${value}% (${props?.payload?.count?.toLocaleString() ?? 0} snapshots)`,
                      'At-Risk Rate',
                    ]}
                  />
                  <Bar dataKey="risk_rate" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                    {cohort.completion_bands.map((_, idx) => (
                      <Cell key={idx} fill={bandColors[idx % bandColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* River of Progress Longitudinal Chart */}
          <div className="mt-6 rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-6 backdrop-blur-xl">
            <h2 className="font-fantasy text-lg font-bold text-emerald-200">
              River of Progress — Longitudinal Vitality Across Cycles
            </h2>
            <p className="text-xs text-slate-400">
              Period-by-period evolution of attendance, coursework completion, and cohort at-risk percentage
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={cohort.period_trends} margin={{ top: 16, right: 20, left: 0, bottom: 6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="period" tickFormatter={(p: number) => `Period ${p}`} stroke="#94A3B8" />
                <YAxis stroke="#94A3B8" tickFormatter={(v: number) => `${v}%`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(7, 21, 16, 0.96)',
                    border: '1px solid #2D9978',
                    borderRadius: '10px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="avg_attendance" name="Avg Attendance (%)" stroke="#5DE6E0" strokeWidth={3} dot={{ r: 5 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="avg_completion" name="Avg Completion (%)" stroke="#5DBB8A" strokeWidth={3} dot={{ r: 5 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="at_risk_rate" name="At-Risk Rate (%)" stroke="#F43F5E" strokeWidth={3} dot={{ r: 5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Explainable AI Cohort Attribution & Rule-Based Separator */}
          {shapBars.length > 0 && (
            <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-6 backdrop-blur-xl lg:col-span-7">
                <div className="mb-2 flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald-300" />
                  <h2 className="font-fantasy text-lg font-bold text-emerald-200">
                    Global SHAP Attribution ({analytics?.model_name || 'Champion Model'})
                  </h2>
                </div>
                <p className="mb-2 text-xs text-slate-400">
                  Average absolute probability impact across N={analytics?.sample_size ?? 400} sampled snapshots (non-causal statistical explanation)
                </p>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={shapBars} layout="vertical" margin={{ top: 4, right: 16, left: 10, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                    <XAxis type="number" stroke="#94A3B8" />
                    <YAxis type="category" dataKey="feature" width={160} stroke="#94A3B8" tick={{ fill: '#E2E8F0', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(7, 21, 16, 0.96)',
                        border: '1px solid #2D9978',
                        borderRadius: '10px',
                      }}
                    />
                    <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                      {shapBars.map((_, idx) => (
                        <Cell key={idx} fill={idx < 3 ? '#5DBB8A' : '#2D9978'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="flex flex-col justify-between rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-6 backdrop-blur-xl lg:col-span-5">
                <div>
                  <div className="mb-2 flex items-center gap-2">
                    <Compass className="h-4 w-4 text-amber-300" />
                    <h2 className="font-fantasy text-lg font-bold text-amber-200">
                      Explainability vs. Teacher Support
                    </h2>
                  </div>
                  <div className="mt-3 space-y-3 text-xs leading-relaxed text-slate-300">
                    <div className="rounded-xl border border-emerald-400/20 bg-emerald-950/30 p-3.5">
                      <p className="font-bold text-emerald-300">1. Statistical Model Attribution (SHAP)</p>
                      <p className="mt-1">
                        Explains how prior scores, momentum, attendance, and coursework contributed mathematically to the calibrated risk probability. It identifies predictive signals, <b>not root causes</b>.
                      </p>
                    </div>
                    <div className="rounded-xl border border-amber-400/20 bg-amber-950/25 p-3.5">
                      <p className="font-bold text-amber-300">2. Rule-Based Support Guidelines</p>
                      <p className="mt-1">
                        Recommended mentoring outreach, homework check-ins, and tutoring suggestions come from deterministic teacher rules applied separately after human review.
                      </p>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => navigate('/academy')}
                  className="mt-4 inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-400 to-teal-500 px-4 py-2.5 font-fantasy text-xs font-bold uppercase tracking-wider text-slate-950 transition hover:brightness-110"
                >
                  Open Student SHAP Dossiers <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </WorldTransition>
  );
};
