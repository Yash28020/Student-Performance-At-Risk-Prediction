import type { WorldConfig } from '../types';

export const WORLDS: WorldConfig[] = [
  {
    id: 0,
    shortName: 'Gate',
    name: 'Kingdom Gate',
    moduleName: 'Portal Entrance',
    subtitle: 'Threshold of the Academic Multiverse',
    route: '/',
    accentColor: '#F4C76B',
    glowColor: 'rgba(244, 199, 107, 0.5)',
    themeGradient: 'from-[#10182E] via-[#1B2246] to-[#10182E]',
  },
  {
    id: 1,
    shortName: 'Kingdom',
    name: 'The Enchanted Kingdom',
    moduleName: 'Executive Overview',
    subtitle: 'School Cohort Telemetry & Risk Horizon',
    route: '/kingdom',
    accentColor: '#F4C76B',
    glowColor: 'rgba(244, 199, 107, 0.45)',
    themeGradient: 'from-[#10182E] via-[#211C42] to-[#0D1326]',
  },
  {
    id: 2,
    shortName: 'Academy',
    name: 'The Wizarding Academy',
    moduleName: 'Single & Batch Prediction',
    subtitle: 'Student Profiles, Batch Scorer, SHAP & Teacher Plans',
    route: '/academy',
    accentColor: '#C9A84C',
    glowColor: 'rgba(201, 168, 76, 0.45)',
    themeGradient: 'from-[#1A1423] via-[#2A1B32] to-[#120E1A]',
  },
  {
    id: 3,
    shortName: 'Nature',
    name: 'The Mythical Nature Realm',
    moduleName: 'Cohort & Explainability',
    subtitle: 'Attendance, Coursework & Feature Attribution',
    route: '/nature',
    accentColor: '#2D9978',
    glowColor: 'rgba(45, 153, 120, 0.45)',
    themeGradient: 'from-[#091E16] via-[#123225] to-[#071510]',
  },
  {
    id: 4,
    shortName: 'Cyberpunk',
    name: 'The Cyberpunk Future',
    moduleName: 'Model Lab & Interventions',
    subtitle: 'Benchmarks, What-If Simulator & Teacher Follow-Ups',
    route: '/cyberpunk',
    accentColor: '#56A8F5',
    glowColor: 'rgba(0, 245, 255, 0.45)',
    themeGradient: 'from-[#050A14] via-[#131130] to-[#060916]',
  },
  {
    id: 5,
    shortName: 'Celestial',
    name: 'The Celestial Multiverse',
    moduleName: 'Global SHAP Constellations',
    subtitle: 'Risk Clusters & Predictive Feature Gravity',
    route: '/celestial',
    accentColor: '#7555C7',
    glowColor: 'rgba(117, 85, 199, 0.5)',
    themeGradient: 'from-[#050815] via-[#160F33] to-[#06091A]',
  },
  {
    id: 6,
    shortName: 'Oceanic',
    name: 'The Oceanic Crystal Realm',
    moduleName: 'Governance & Reports',
    subtitle: 'Responsible AI, Audit Ledger & CSV Report Exports',
    route: '/ocean',
    accentColor: '#5DE6E0',
    glowColor: 'rgba(93, 230, 224, 0.45)',
    themeGradient: 'from-[#071B2E] via-[#0B2A3E] to-[#04111E]',
  },
];

export const FEATURE_LABELS: Record<string, string> = {
  attendance_pct: 'Classroom Attendance (%)',
  assignment_completion_pct: 'Assignment Completion (%)',
  late_assignment_count: 'Late Assignment Count',
  previous_score: 'Current Score (t)',
  previous_score_2: 'Prior Score (t-1)',
  previous_score_avg: 'Rolling 2-Score Mean',
  score_trend: 'Score Momentum (Δ)',
  assessment_period: 'Assessment Period',
};

export function formatFeatureName(key: string): string {
  return (
    FEATURE_LABELS[key] ||
    key
      .split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')
  );
}
