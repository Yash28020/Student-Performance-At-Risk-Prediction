import React, { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Castle,
  BookOpen,
  Trees,
  Cpu,
  Orbit,
  Gem,
  Compass,
  Shield,
  ChevronRight,
  ChevronLeft,
  Menu,
  X,
  Activity,
} from 'lucide-react';
import { WORLDS } from '../constants/worlds';
import { fetchHealth } from '../api/client';
import type { HealthResponse, WorldId } from '../types';

const WORLD_ICONS: Record<WorldId, React.ReactNode> = {
  0: <Compass className="h-5 w-5" />,
  1: <Castle className="h-5 w-5" />,
  2: <BookOpen className="h-5 w-5" />,
  3: <Trees className="h-5 w-5" />,
  4: <Cpu className="h-5 w-5" />,
  5: <Orbit className="h-5 w-5" />,
  6: <Gem className="h-5 w-5" />,
};

export const Navigation: React.FC = () => {
  const [expanded, setExpanded] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const location = useLocation();

  // Hide persistent sidebar on the full-screen Kingdom Gate (`/`) unless user opens mobile menu
  const isGate = location.pathname === '/';

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((res) => {
        if (active && res?.status === 'healthy') {
          setApiOnline(true);
          setHealth(res);
        }
      })
      .catch(() => {
        if (active) setApiOnline(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const modelLabel = health?.model_name || health?.champion_architecture || 'ML Pipeline';
  const thresholdLabel =
    health?.selected_threshold !== undefined
      ? `τ*=${health.selected_threshold.toFixed(2)}`
      : 'τ*=active';
  const versionLabel = health?.model_version ? `v${health.model_version}` : 'live';

  if (isGate) {
    return (
      <div className="fixed right-5 top-5 z-50 flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-full border border-amber-300/30 bg-slate-950/75 px-3.5 py-1.5 text-xs backdrop-blur-md">
          <span
            className={`h-2 w-2 rounded-full ${
              apiOnline === true
                ? 'bg-emerald-400 shadow-[0_0_8px_#34D399]'
                : apiOnline === false
                  ? 'bg-rose-400'
                  : 'animate-pulse bg-amber-300'
            }`}
          />
          <span className="font-mono text-[11px] text-amber-100/90">
            {apiOnline === true
              ? `${modelLabel} ${versionLabel} Ready`
              : apiOnline === false
                ? 'Backend Offline'
                : 'Linking Core...'}
          </span>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Desktop Left Multiverse Sidebar */}
      <aside
        className={`sticky top-0 z-40 hidden h-screen shrink-0 flex-col border-r border-white/10 bg-[#080C18]/90 backdrop-blur-2xl transition-all duration-300 md:flex ${
          expanded ? 'w-64' : 'w-20'
        }`}
      >
        {/* Crest Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-4">
          <NavLink to="/" className="flex items-center gap-3 overflow-hidden">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-amber-400/40 bg-gradient-to-br from-amber-500/20 to-purple-600/20 text-amber-300 shadow-[0_0_15px_rgba(244,199,107,0.25)]">
              <Shield className="h-5 w-5" />
            </div>
            {expanded && (
              <div className="min-w-0">
                <h1 className="truncate font-fantasy text-sm font-bold tracking-wider text-amber-200">
                  EDUGUARD AI
                </h1>
                <p className="truncate text-[10px] uppercase tracking-widest text-slate-400">
                  Academic Multiverse
                </p>
              </div>
            )}
          </NavLink>

          <button
            onClick={() => setExpanded((e) => !e)}
            className="rounded-lg border border-white/10 bg-white/5 p-1.5 text-slate-300 transition hover:border-amber-400/40 hover:text-amber-300"
            title={expanded ? 'Collapse Portal Rail' : 'Expand Portal Rail'}
          >
            {expanded ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </button>
        </div>

        {/* World Portals */}
        <nav className="flex-1 space-y-1.5 overflow-y-auto px-3 py-4">
          {expanded && (
            <p className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              Six Realms
            </p>
          )}
          {WORLDS.map((world) => (
            <NavLink
              key={world.id}
              to={world.route}
              className={({ isActive }) =>
                `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-200 ${
                  isActive
                    ? 'border border-white/20 bg-white/10 text-white shadow-lg'
                    : 'border border-transparent text-slate-400 hover:bg-white/5 hover:text-white'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.div
                      layoutId="activeWorldIndicator"
                      className="absolute inset-y-1.5 left-0 w-1 rounded-r-full"
                      style={{
                        backgroundColor: world.accentColor,
                        boxShadow: `0 0 12px ${world.accentColor}`,
                      }}
                    />
                  )}
                  <div
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-transform group-hover:scale-110"
                    style={{
                      color: isActive ? world.accentColor : undefined,
                      backgroundColor: isActive ? `${world.accentColor}22` : 'rgba(255,255,255,0.04)',
                    }}
                  >
                    {WORLD_ICONS[world.id]}
                  </div>

                  {expanded && (
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-semibold">
                        {world.id === 0 ? 'Kingdom Gate' : `${world.id}. ${world.shortName}`}
                      </p>
                      <p className="truncate text-[10px] text-slate-400">{world.moduleName}</p>
                    </div>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* System Status Footer */}
        <div className="border-t border-white/10 p-3">
          <div className="flex items-center gap-2.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5">
            <Activity
              className={`h-4 w-4 shrink-0 ${
                apiOnline ? 'text-emerald-400' : 'text-amber-400'
              }`}
            />
            {expanded && (
              <div className="min-w-0 flex-1 text-[11px]">
                <p className="font-semibold text-slate-200">
                  {apiOnline === true
                    ? 'FastAPI Core Active'
                    : apiOnline === false
                      ? 'Backend Offline'
                      : 'Connecting Engine'}
                </p>
                <p className="truncate font-mono text-[10px] text-slate-400">
                  {apiOnline === true && health
                    ? `${modelLabel} · ${thresholdLabel} · ${versionLabel}`
                    : apiOnline === false
                      ? 'Awaiting API heartbeat'
                      : 'Loading model metadata...'}
                </p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile Top Bar & Drawer */}
      <div className="fixed inset-x-0 top-0 z-50 flex items-center justify-between border-b border-white/10 bg-[#080C18]/90 px-4 py-3 backdrop-blur-xl md:hidden">
        <NavLink to="/" className="flex items-center gap-2.5">
          <Shield className="h-5 w-5 text-amber-300" />
          <span className="font-fantasy text-sm font-bold tracking-wider text-amber-200">
            EDUGUARD KINGDOM
          </span>
        </NavLink>
        <button
          onClick={() => setMobileOpen((o) => !o)}
          className="rounded-lg border border-white/15 bg-white/5 p-2 text-slate-200"
          aria-label="Toggle Menu"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, x: -240 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -240 }}
            className="fixed inset-y-0 left-0 z-50 w-72 border-r border-white/15 bg-[#080C18] p-4 pt-16 shadow-2xl md:hidden"
          >
            <div className="space-y-2">
              {WORLDS.map((world) => (
                <NavLink
                  key={world.id}
                  to={world.route}
                  className={({ isActive }) =>
                    `flex items-center gap-3 rounded-xl px-3 py-3 text-sm ${
                      isActive ? 'bg-white/10 text-white' : 'text-slate-400'
                    }`
                  }
                >
                  <span style={{ color: world.accentColor }}>{WORLD_ICONS[world.id]}</span>
                  <div>
                    <p className="font-semibold">
                      {world.id === 0 ? 'Kingdom Gate' : `${world.id}. ${world.name}`}
                    </p>
                    <p className="text-[11px] text-slate-400">{world.moduleName}</p>
                  </div>
                </NavLink>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

