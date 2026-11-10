import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Cell,
  ResponsiveContainer,
} from 'recharts';
import type { SHAPContribution } from '../types';
import { formatFeatureName } from '../constants/worlds';

interface SHAPChartProps {
  contributions: SHAPContribution[];
  baseValue?: number;
  height?: number;
}

export const SHAPChart: React.FC<SHAPChartProps> = ({
  contributions,
  baseValue,
  height = 290,
}) => {
  const data = [...contributions]
    .sort((a, b) => a.shap_value - b.shap_value)
    .map((c) => ({
      name: formatFeatureName(c.feature),
      shap: c.shap_value,
      rawValue: c.value ?? 'Imputed',
      direction: c.direction,
    }));

  return (
    <div className="w-full">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-300/80">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
            Elevates Risk (+)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-400" />
            Protective Factor (−)
          </span>
        </div>
        {baseValue !== undefined && (
          <span className="font-mono text-[11px] text-amber-300/90">
            Base Cohort Expected Value: {baseValue.toFixed(3)}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 6, right: 24, left: 16, bottom: 6 }}
        >
          <XAxis
            type="number"
            stroke="#94A3B8"
            tick={{ fill: '#CBD5E1', fontSize: 11 }}
            tickFormatter={(v: number) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={160}
            stroke="#94A3B8"
            tick={{ fill: '#F8FAFC', fontSize: 11 }}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.06)' }}
            contentStyle={{
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(244, 199, 107, 0.4)',
              borderRadius: '10px',
              color: '#F8FAFC',
              fontSize: '12px',
            }}
            formatter={(value: unknown, _name: unknown, props: { payload?: { rawValue?: string | number } }) => {
              const num = Number(value);
              const valStr = num >= 0 ? `+${num.toFixed(4)}` : num.toFixed(4);
              const featVal = props?.payload?.rawValue ?? '';
              return [`${valStr} (Observed: ${featVal})`, 'SHAP Impact'];
            }}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.35)" strokeDasharray="3 3" />
          <Bar dataKey="shap" radius={[4, 4, 4, 4]} isAnimationActive={false}>
            {data.map((entry, idx) => (
              <Cell
                key={`cell-${idx}`}
                fill={entry.shap >= 0 ? '#F43F5E' : '#10B981'}
                fillOpacity={0.9}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
