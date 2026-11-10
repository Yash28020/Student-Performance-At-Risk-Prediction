import React from 'react';
import { ShieldCheck } from 'lucide-react';

interface AdvisoryBannerProps {
  text?: string;
  accentColor?: string;
}

export const AdvisoryBanner: React.FC<AdvisoryBannerProps> = ({
  text = 'Advisory Intelligence Notice: EduGuard AI surfaces statistical risk correlations to empower proactive support by school teachers and academic coordinators. Outputs never trigger automated grading, discipline, or punitive action.',
  accentColor = '#F4C76B',
}) => {
  return (
    <div
      className="mb-6 flex items-start gap-3 rounded-xl border bg-slate-900/60 px-4 py-3 text-xs leading-relaxed text-slate-200/90 backdrop-blur-md"
      style={{ borderColor: `${accentColor}45` }}
    >
      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" style={{ color: accentColor }} />
      <span>{text}</span>
    </div>
  );
};

