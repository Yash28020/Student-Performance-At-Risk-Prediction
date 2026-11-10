import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Users,
  AlertTriangle,
  ShieldCheck,
  Target,
  ArrowRight,
  Castle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  PieChart,
  Pie,
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
import { WORLDS } from '../constants/worlds';
import { extractApiError, fetchCohortSummary, fetchModelInfo } from '../api/client';
import type { CohortSummary, ModelInfo } from '../types';

export const World1Kingdom: React.FC = () => {
  const navigate = useNavigate();
  const [cohort, setCohort] = useState<CohortSummary | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [c, m] = await Promise.all([fetchCohortSummary(), fetchModelInfo()]);
      setCohort(c);
      setModelInfo(m);
    } catch (err) {
      setError(extractApiError(err, 'Unable to reach EduGuard backend telemetry service.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="relative min-h-screen bg-[#0B1022]">
        <WorldBackdrop variant="kingdom" />
        <LoadingSpell message="Illuminating the Enchanted Kingdom..." accentColor="#F4C76B" />
      </div>
    );
  }

  if (error || !cohort || !modelInfo) {
    return (
      <div className="relative min-h-screen bg-[#0B1022] p-8">
        <WorldBackdrop variant="kingdom" />
        <ErrorPortal message={error || 'Kingdom telemetry unavailable.'} onRetry={loadData} />
      </div>
    );
  }

  const pieData = [
    { name: 'Passing Benchmark (≥ 40)', value: cohort.passing_count, color: '#2D9978' },
    { name: 'At-Risk Events (< 40)', value: cohort.at_risk_count, color: '#F43F5E' },
  ];

  const temporalTest = modelInfo.held_out_test_metrics_temporal;
  const valMetrics = modelInfo.validation_metrics;
  const nextWorlds = WORLDS.filter((w) => w.id >= 2);

  return (
    <WorldTransition worldKey="world-1">
      <div className="relative min-h-screen bg-[#0B1022] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="kingdom" />
        <ParticleField theme="kingdom" density={45} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Royal Header */}
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-amber-300">
                <Castle className="h-3.5 w-3.5" />
                World 1 · Executive Overview
              </div>
              <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-[#FFE599] via-[#F4C76B] to-amber-200 bg-clip-text sm:text-4xl">
                The Enchanted Kingdom
              </h1>
              <p className="mt-1 text-sm text-slate-300/85">
                Royal Observatory of longitudinal student readiness, risk rates, and champion model performance.
              </p>
            </div>

            <button
              onClick={() => navigate('/academy')}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#F4C76B] to-amber-400 px-5 py-2.5 font-fantasy text-xs font-bold uppercase tracking-wider text-slate-950 shadow-[0_0_25px_rgba(244,199,107,0.4)] transition hover:scale-105"
            >
              Inspect Students
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          <AdvisoryBanner accentColor="#F4C76B" />

          {/* KPI Metric Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="Total Simulated Students"
              value={cohort.total_unique_students.toLocaleString()}
              subtitle={`${cohort.total_snapshots.toLocaleString()} longitudinal pre-assessment snapshots`}
              badgeText={`Periods 1–4 · Avg Score: ${cohort.avg_score}`}
              badgeColor="bg-amber-500/15 text-amber-300 border-amber-400/30"
              accentColor="#F4C76B"
              icon={<Users className="h-5 w-5" />}
              delay={0.05}
            />
            <MetricCard
              title="Historical Risk Rate"
              value={`${cohort.at_risk_rate_pct}%`}
              subtitle={`${cohort.at_risk_count.toLocaleString()} snapshots scored < 40/100 next cycle`}
              badgeText={`${cohort.passing_count.toLocaleString()} Passing Snapshots`}
              badgeColor="bg-rose-500/15 text-rose-300 border-rose-400/30"
              accentColor="#F43F5E"
              icon={<AlertTriangle className="h-5 w-5" />}
              delay={0.1}
            />
            <MetricCard
              title="Period 4 Support Queue"
              value={cohort.latest_period_at_risk_students.toLocaleString()}
              subtitle={`Attendance ${cohort.avg_attendance}% · Completion ${cohort.avg_completion}%`}
              badgeText={`Avg Late Assignments: ${cohort.avg_late_assignments}`}
              badgeColor="bg-purple-500/15 text-purple-300 border-purple-400/30"
              accentColor="#7555C7"
              icon={<Target className="h-5 w-5" />}
              delay={0.15}
            />
            <MetricCard
              title="Model Safety-Net Recall"
              value={`${(valMetrics.recall * 100).toFixed(1)}%`}
              subtitle={`Temporal Test AP: ${(temporalTest.average_precision ?? temporalTest.pr_auc).toFixed(3)} · F1: ${temporalTest.f1_score.toFixed(3)}`}
              badgeText={`${modelInfo.model_name ?? modelInfo.champion_architecture} · τ*=${modelInfo.selected_threshold}`}
              badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-400/30"
              accentColor="#2D9978"
              icon={<ShieldCheck className="h-5 w-5" />}
              delay={0.2}
            />
          </div>

          {/* Visualizations Row */}
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Trajectory Line Chart */}
            <div className="rounded-2xl border border-amber-300/20 bg-slate-900/65 p-6 backdrop-blur-xl lg:col-span-7">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="font-fantasy text-lg font-bold text-amber-200">
                    Cohort-Level Assessment Trajectory
                  </h2>
                  <p className="text-xs text-slate-400">
                    Average prior assessment score by period comparing Passing vs. At-Risk trajectories
                  </p>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={cohort.period_trends} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis
                    dataKey="period"
                    stroke="#94A3B8"
                    tickFormatter={(p: number) => `Period ${p}`}
                    tick={{ fill: '#CBD5E1', fontSize: 12 }}
                  />
                  <YAxis
                    domain={[20, 90]}
                    stroke="#94A3B8"
                    tick={{ fill: '#CBD5E1', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 24, 46, 0.96)',
                      border: '1px solid rgba(244, 199, 107, 0.45)',
                      borderRadius: '12px',
                      color: '#F8FAFC',
                    }}
                  />
                  <Legend />
                  <ReferenceLine
                    y={40}
                    stroke="#F43F5E"
                    strokeDasharray="5 5"
                    label={{ value: 'Benchmark (40)', fill: '#FDA4AF', fontSize: 11, position: 'insideTopRight' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="passing_avg_score"
                    name="Passing Cohort (≥ 40)"
                    stroke="#2D9978"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#2D9978' }}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_score"
                    name="Overall Cohort Average"
                    stroke="#F4C76B"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={{ r: 4, fill: '#F4C76B' }}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="at_risk_avg_score"
                    name="At-Risk Cohort (< 40)"
                    stroke="#F43F5E"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#F43F5E' }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Risk Distribution Donut */}
            <div className="rounded-2xl border border-amber-300/20 bg-slate-900/65 p-6 backdrop-blur-xl lg:col-span-5">
              <h2 className="font-fantasy text-lg font-bold text-amber-200">
                Academic Risk Distribution
              </h2>
              <p className="text-xs text-slate-400">
                Proportion of {cohort.total_snapshots.toLocaleString()} longitudinal prediction snapshots
              </p>

              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={62}
                    outerRadius={95}
                    paddingAngle={4}
                    dataKey="value"
                    isAnimationActive={false}
                  >
                    {pieData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} stroke="rgba(0,0,0,0.4)" />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 24, 46, 0.96)',
                      border: '1px solid rgba(244, 199, 107, 0.45)',
                      borderRadius: '12px',
                      color: '#F8FAFC',
                    }}
                    formatter={(val: unknown) => [`${Number(val).toLocaleString()} snapshots`, 'Count']}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>

              <div className="mt-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs text-slate-300">
                <span className="font-semibold text-amber-300">Champion Engine: </span>
                {modelInfo.model_name ?? modelInfo.champion_architecture}
                {modelInfo.is_calibrated ? ' (Calibrated)' : ''} · Disjoint Test Recall{' '}
                {(modelInfo.held_out_test_metrics_disjoint.recall * 100).toFixed(1)}% · Temporal AP{' '}
                {(temporalTest.average_precision ?? temporalTest.pr_auc).toFixed(3)}
              </div>
            </div>
          </div>

          {/* Interconnected World Gateways */}
          <div className="mt-8">
            <h2 className="mb-3 font-fantasy text-lg font-bold text-slate-200">
              Traverse the Five Realms Beyond the Castle
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
              {nextWorlds.map((w) => (
                <button
                  key={w.id}
                  onClick={() => navigate(w.route)}
                  className="group rounded-xl border border-white/10 bg-slate-900/60 p-4 text-left backdrop-blur-md transition hover:-translate-y-1 hover:border-white/30"
                >
                  <p
                    className="text-[10px] font-bold uppercase tracking-widest"
                    style={{ color: w.accentColor }}
                  >
                    World {w.id}
                  </p>
                  <p className="mt-1 font-fantasy text-sm font-bold text-white">{w.name}</p>
                  <p className="mt-1 text-xs text-slate-400">{w.moduleName}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </WorldTransition>
  );
};
