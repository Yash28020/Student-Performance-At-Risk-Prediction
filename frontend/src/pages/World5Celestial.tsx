import React, { useEffect, useState } from 'react';
import { Orbit, Sparkles, Compass } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
  Legend,
} from 'recharts';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WorldTransition } from '../components/WorldTransition';
import { LoadingSpell } from '../components/LoadingSpell';
import { ErrorPortal } from '../components/ErrorPortal';
import { AdvisoryBanner } from '../components/AdvisoryBanner';
import { extractApiError, fetchGlobalAnalytics } from '../api/client';
import { formatFeatureName } from '../constants/worlds';
import type { GlobalAnalyticsResponse } from '../types';

export const World5Celestial: React.FC = () => {
  const [analytics, setAnalytics] = useState<GlobalAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchGlobalAnalytics();
      setAnalytics(data);
    } catch (err) {
      setError(extractApiError(err, 'Unable to chart Celestial Multiverse analytics.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="relative min-h-screen bg-[#040614]">
        <WorldBackdrop variant="celestial" />
        <LoadingSpell message="Charting Global SHAP Constellations..." accentColor="#7555C7" />
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="relative min-h-screen bg-[#040614] p-8">
        <WorldBackdrop variant="celestial" />
        <ErrorPortal message={error || 'Global analytics unavailable.'} onRetry={loadAnalytics} />
      </div>
    );
  }

  const passingStars = analytics.constellation_sample.filter((p) => p.at_risk === 0);
  const riskStars = analytics.constellation_sample.filter((p) => p.at_risk === 1);

  const shapBars = analytics.global_shap_importance.map((item) => ({
    feature: formatFeatureName(item.feature),
    importance: item.mean_absolute_shap,
  }));

  const corrBars = analytics.feature_correlations.map((item) => ({
    feature: formatFeatureName(item.feature),
    corr: item.correlation_with_risk,
  }));

  return (
    <WorldTransition worldKey="world-5">
      <div className="relative min-h-screen bg-[#040614] px-4 py-8 pt-20 sm:px-8 md:pt-8">
        <WorldBackdrop variant="celestial" />
        <ParticleField theme="celestial" density={85} />

        <div className="relative z-10 mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-indigo-400/35 bg-indigo-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-widest text-indigo-200">
              <Orbit className="h-3.5 w-3.5" />
              World 5 · Global SHAP & Risk Constellations
            </div>
            <h1 className="mt-2 font-fantasy text-3xl font-bold text-transparent bg-gradient-to-r from-[#A5B4FC] via-[#56A8F5] to-[#FFE566] bg-clip-text sm:text-4xl">
              The Celestial Multiverse
            </h1>
            <p className="mt-1 text-sm text-indigo-100/75">
              Macro-level TreeExplainer SHAP rankings, signed risk correlations, and school student risk clusters.
            </p>
          </div>

          <AdvisoryBanner
            text={`Cosmic Analytics Notice: Constellations plot authentic synthetic student records (N=${analytics.sample_size} TreeExplainer evaluation sample) using the ${analytics.model_name ?? analytics.classifier_architecture ?? 'Champion'} model${analytics.is_calibrated ? ' (Sigmoid Calibrated)' : ''}. SHAP values measure statistical predictive associations, not causal proof.`}
            accentColor="#56A8F5"
          />

          {/* Top Row: Constellation Scatter */}
          <div className="rounded-2xl border border-indigo-400/25 bg-slate-950/75 p-6 backdrop-blur-xl">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="font-fantasy text-lg font-bold text-sky-200">
                  Student Risk Constellation Map (Attendance vs. Current Score)
                </h2>
                <p className="text-xs text-slate-400">
                  Hover any star to inspect scholar coordinates, completion rate, and calibrated model probability
                </p>
              </div>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300/30 bg-amber-500/10 px-3 py-1 font-mono text-[11px] text-amber-200">
                <Sparkles className="h-3 w-3" />
                {analytics.constellation_sample.length} Mapped Stars
              </span>
            </div>

            <ResponsiveContainer width="100%" height={330}>
              <ScatterChart margin={{ top: 14, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                <XAxis
                  type="number"
                  dataKey="attendance_pct"
                  name="Attendance"
                  unit="%"
                  domain={[0, 100]}
                  stroke="#94A3B8"
                />
                <YAxis
                  type="number"
                  dataKey="previous_score"
                  name="Current Score"
                  domain={[0, 100]}
                  stroke="#94A3B8"
                />
                <ZAxis type="number" dataKey="risk_probability" range={[45, 220]} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={(props: { active?: boolean; payload?: ReadonlyArray<{ payload?: { student_id: string; period: number; attendance_pct: number; previous_score: number; score_trend: number; assignment_completion_pct: number; risk_probability: number } }> }) => {
                    const { active, payload } = props;
                    if (!active || !payload || !payload.length || !payload[0].payload) return null;
                    const pt = payload[0].payload;
                    return (
                      <div className="rounded-xl border border-indigo-400/50 bg-[#070B1E]/95 p-3 text-xs text-white shadow-2xl">
                        <p className="font-mono font-bold text-amber-300">
                          {pt.student_id} (Period {pt.period})
                        </p>
                        <p className="mt-1">Attendance: <b>{pt.attendance_pct}%</b></p>
                        <p>Score: <b>{pt.previous_score}/100</b> (Trend {pt.score_trend >= 0 ? '+' : ''}{pt.score_trend})</p>
                        <p>Completion: <b>{pt.assignment_completion_pct}%</b></p>
                        <p className="mt-1 font-mono text-sky-300">
                          Model Risk Prob: {(pt.risk_probability * 100).toFixed(1)}%
                        </p>
                      </div>
                    );
                  }}
                />
                <Legend />
                <ReferenceLine y={40} stroke="#F43F5E" strokeDasharray="4 4" />
                <Scatter name="Passing Benchmark (≥ 40)" data={passingStars} fill="#56A8F5" fillOpacity={0.75} isAnimationActive={false} />
                <Scatter name="At-Risk (< 40)" data={riskStars} fill="#F43F5E" fillOpacity={0.9} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Bottom Row: Global SHAP + Feature Correlations */}
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-purple-400/25 bg-slate-950/75 p-6 backdrop-blur-xl">
              <div className="mb-3 flex items-center gap-2">
                <Compass className="h-4 w-4 text-amber-300" />
                <h2 className="font-fantasy text-lg font-bold text-amber-200">
                  Global Mean |SHAP| Feature Gravity
                </h2>
              </div>
              <p className="mb-2 text-xs text-slate-400">
                Average predictive magnitude of each pre-assessment signal across the cohort
              </p>

              <ResponsiveContainer width="100%" height={290}>
                <BarChart data={shapBars} layout="vertical" margin={{ top: 6, right: 20, left: 12, bottom: 6 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                  <XAxis type="number" stroke="#94A3B8" />
                  <YAxis type="category" dataKey="feature" width={165} stroke="#94A3B8" tick={{ fill: '#E2E8F0', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(5, 8, 21, 0.96)',
                      border: '1px solid #7555C7',
                      borderRadius: '10px',
                    }}
                  />
                  <Bar dataKey="importance" name="Mean |SHAP| (Probability Space)" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                    {shapBars.map((_, i) => (
                      <Cell key={i} fill={i < 3 ? '#FFE566' : '#7555C7'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="mt-2 rounded-lg border border-indigo-400/20 bg-indigo-950/30 px-3 py-2 text-[11px] text-indigo-200/85">
                <b>Non-Causal Interpretation:</b> SHAP attributions explain how much each input shifted the model&apos;s predicted risk probability across the cohort, not direct cause-and-effect.
              </p>
            </div>

            <div className="rounded-2xl border border-sky-400/25 bg-slate-950/75 p-6 backdrop-blur-xl">
              <h2 className="font-fantasy text-lg font-bold text-sky-200">
                Signed Feature Correlations with Risk
              </h2>
              <p className="mb-2 text-xs text-slate-400">
                Negative correlations reduce risk (higher score/attendance); positive elevate risk (late submissions)
              </p>

              <ResponsiveContainer width="100%" height={290}>
                <BarChart data={corrBars} layout="vertical" margin={{ top: 6, right: 20, left: 12, bottom: 6 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                  <XAxis type="number" domain={[-0.75, 0.75]} stroke="#94A3B8" />
                  <YAxis type="category" dataKey="feature" width={165} stroke="#94A3B8" tick={{ fill: '#E2E8F0', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(5, 8, 21, 0.96)',
                      border: '1px solid #56A8F5',
                      borderRadius: '10px',
                    }}
                  />
                  <ReferenceLine x={0} stroke="rgba(255,255,255,0.35)" />
                  <Bar dataKey="corr" name="Pearson Correlation" radius={[4, 4, 4, 4]} isAnimationActive={false}>
                    {corrBars.map((entry, i) => (
                      <Cell key={i} fill={entry.corr >= 0 ? '#F43F5E' : '#56A8F5'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </WorldTransition>
  );
};
