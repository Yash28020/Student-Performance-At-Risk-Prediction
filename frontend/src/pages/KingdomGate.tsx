import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight, ShieldCheck, Castle, BookOpen, Trees, Cpu, Orbit, Gem, RefreshCw } from 'lucide-react';
import { ParticleField } from '../components/ParticleField';
import { WorldBackdrop } from '../components/WorldBackdrop';
import { WORLDS } from '../constants/worlds';
import { fetchCohortSummary, fetchModelInfo } from '../api/client';
import type { CohortSummary, ModelInfo } from '../types';

const PORTAL_ICONS: Record<number, React.ReactNode> = {
  1: <Castle className="h-5 w-5" />,
  2: <BookOpen className="h-5 w-5" />,
  3: <Trees className="h-5 w-5" />,
  4: <Cpu className="h-5 w-5" />,
  5: <Orbit className="h-5 w-5" />,
  6: <Gem className="h-5 w-5" />,
};

export const KingdomGate: React.FC = () => {
  const navigate = useNavigate();
  const [cohort, setCohort] = useState<CohortSummary | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loadingTelemetry, setLoadingTelemetry] = useState(true);
  const [telemetryError, setTelemetryError] = useState(false);

  const loadTelemetry = useCallback(async () => {
    setLoadingTelemetry(true);
    setTelemetryError(false);
    try {
      const [c, m] = await Promise.all([fetchCohortSummary(), fetchModelInfo()]);
      setCohort(c);
      setModelInfo(m);
    } catch {
      setTelemetryError(true);
    } finally {
      setLoadingTelemetry(false);
    }
  }, []);

  useEffect(() => {
    loadTelemetry();
  }, [loadTelemetry]);

  const realms = WORLDS.filter((w) => w.id !== 0);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070B19] text-white">
      <WorldBackdrop variant="gate" />
      <ParticleField theme="kingdom" density={75} />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-6 py-16 text-center">
        {/* Crown / Advisory Crest */}
        <motion.div
          initial={{ opacity: 0, y: -18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="mb-6 inline-flex items-center gap-2 rounded-full border border-amber-300/35 bg-amber-500/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-amber-200 backdrop-blur-md"
        >
          <Sparkles className="h-3.5 w-3.5 text-amber-300" />
          Detect Early · Understand Deeply · Support Intelligently
        </motion.div>

        {/* Hero Heading */}
        <motion.h1
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.85, delay: 0.1 }}
          className="font-fantasy text-4xl font-black tracking-wide text-transparent bg-gradient-to-r from-[#FFE599] via-[#F4C76B] to-[#FFE0A3] bg-clip-text sm:text-6xl md:text-7xl drop-shadow-[0_6px_32px_rgba(244,199,107,0.3)]"
        >
          Welcome to EduGuard Kingdom
        </motion.h1>

        {/* Hero Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          className="mt-5 max-w-2xl text-lg font-light tracking-wide text-slate-200/90 sm:text-xl"
        >
          Where Intelligence Meets the Future of Every Student.
        </motion.p>

        {/* Primary CTA */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-4"
        >
          <button
            onClick={() => navigate('/kingdom')}
            className="group relative inline-flex items-center gap-3 overflow-hidden rounded-xl bg-gradient-to-r from-[#F4C76B] via-[#FFE08A] to-[#E3AD47] px-9 py-4 font-fantasy text-sm font-black uppercase tracking-[0.2em] text-[#10182E] shadow-[0_0_45px_rgba(244,199,107,0.55)] transition-all duration-300 hover:scale-105 hover:shadow-[0_0_70px_rgba(244,199,107,0.85)]"
          >
            <span>ENTER THE KINGDOM</span>
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </button>

          <button
            onClick={() => navigate('/academy')}
            className="inline-flex items-center gap-2 rounded-xl border border-white/25 bg-white/5 px-6 py-4 text-xs font-semibold uppercase tracking-widest text-slate-100 backdrop-blur-md transition hover:border-amber-300/60 hover:bg-white/10"
          >
            Consult Wizarding Scrolls
          </button>
        </motion.div>

        {/* Live Telemetry Pill Row */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.55 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-6 rounded-2xl border border-white/10 bg-slate-950/60 px-6 py-3.5 text-xs backdrop-blur-xl"
        >
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Cohort:</span>
            <span className="font-mono font-bold text-amber-300">
              {cohort
                ? `${cohort.total_unique_students.toLocaleString()} Students`
                : loadingTelemetry
                  ? 'Syncing...'
                  : 'Offline'}
            </span>
          </div>
          <span className="hidden h-3.5 w-px bg-white/15 sm:block" />
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Snapshots:</span>
            <span className="font-mono font-bold text-cyan-300">
              {cohort
                ? cohort.total_snapshots.toLocaleString()
                : loadingTelemetry
                  ? 'Syncing...'
                  : 'Offline'}
            </span>
          </div>
          <span className="hidden h-3.5 w-px bg-white/15 sm:block" />
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Safety-Net Recall:</span>
            <span className="font-mono font-bold text-emerald-400">
              {modelInfo?.validation_metrics
                ? `${(modelInfo.validation_metrics.recall * 100).toFixed(1)}%`
                : loadingTelemetry
                  ? 'Syncing...'
                  : 'Offline'}
            </span>
          </div>
          <span className="hidden h-3.5 w-px bg-white/15 sm:block" />
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-3.5 w-3.5 text-purple-300" />
            <span className="text-slate-300">
              {modelInfo?.model_name ? `${modelInfo.model_name} · ` : ''}Advisory Only · SHAP Explainable
            </span>
          </div>
          {telemetryError && (
            <button
              onClick={loadTelemetry}
              className="inline-flex items-center gap-1 rounded-full border border-rose-400/40 bg-rose-500/15 px-2.5 py-0.5 text-[11px] font-semibold text-rose-200 transition hover:bg-rose-500/30"
            >
              <RefreshCw className="h-3 w-3" />
              Retry API
            </button>
          )}
        </motion.div>

        {/* Six World Gateways Preview Grid */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mt-12 grid w-full grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3"
        >
          {realms.map((realm) => (
            <button
              key={realm.id}
              onClick={() => navigate(realm.route)}
              className="group flex items-start gap-3.5 rounded-2xl border border-white/10 bg-slate-900/55 p-4 text-left backdrop-blur-lg transition-all duration-300 hover:-translate-y-1 hover:border-white/30 hover:bg-slate-900/80"
            >
              <div
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/15 transition-transform group-hover:scale-110"
                style={{
                  color: realm.accentColor,
                  backgroundColor: `${realm.accentColor}1E`,
                  boxShadow: `0 0 18px ${realm.glowColor}`,
                }}
              >
                {PORTAL_ICONS[realm.id]}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
                  World {realm.id} · {realm.moduleName}
                </p>
                <h3 className="mt-0.5 truncate font-fantasy text-sm font-bold text-white group-hover:text-amber-200">
                  {realm.name}
                </h3>
                <p className="mt-1 line-clamp-1 text-xs text-slate-400">{realm.subtitle}</p>
              </div>
            </button>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

