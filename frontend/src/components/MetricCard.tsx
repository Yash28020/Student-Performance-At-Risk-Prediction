import React from 'react';
import { motion } from 'framer-motion';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badgeText?: string;
  badgeColor?: string;
  accentColor?: string;
  icon?: React.ReactNode;
  delay?: number;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badgeText,
  badgeColor = 'bg-white/10 text-white/80 border-white/20',
  accentColor = '#F4C76B',
  icon,
  delay = 0,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay }}
      whileHover={{ y: -3, transition: { duration: 0.2 } }}
      className="relative overflow-hidden rounded-2xl border border-white/15 bg-slate-900/65 p-5 backdrop-blur-xl shadow-xl transition-colors hover:border-white/30"
    >
      <div
        className="pointer-events-none absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-20 blur-2xl"
        style={{ backgroundColor: accentColor }}
      />
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-300/80">
            {title}
          </p>
          <p className="mt-2 font-fantasy text-3xl font-bold tracking-tight text-white">
            {value}
          </p>
        </div>
        {icon && (
          <div
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/15 bg-white/5"
            style={{ color: accentColor }}
          >
            {icon}
          </div>
        )}
      </div>

      {subtitle && <p className="mt-2 text-xs text-slate-300/75">{subtitle}</p>}

      {badgeText && (
        <span
          className={`mt-3 inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${badgeColor}`}
        >
          {badgeText}
        </span>
      )}
    </motion.div>
  );
};

