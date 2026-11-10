import React from 'react';
import { motion } from 'framer-motion';

interface RiskGaugeProps {
  probability: number; // 0 to 1
  threshold?: number; // dynamic decision threshold from backend
  riskLabel?: string;
  tier?: string;
  accentColor?: string;
  size?: number;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  probability,
  threshold = 0.5,
  riskLabel = 'Assessed Risk',
  tier = 'LOW_RISK',
  size = 200,
}) => {
  const pct = Math.min(100, Math.max(0, Math.round(probability * 1000) / 10));
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2 - 8;
  const circumference = Math.PI * radius; // semi-circle
  const progressOffset = circumference - (pct / 100) * circumference;

  const gaugeColor =
    tier === 'HIGH_RISK'
      ? '#EF4444'
      : tier === 'MODERATE_RISK'
        ? '#F59E0B'
        : '#10B981';

  // Threshold tick angle (-180 to 0 deg)
  const thresholdAngle = -180 + threshold * 180;
  const rad = (thresholdAngle * Math.PI) / 180;
  const cx = size / 2;
  const cy = size * 0.58;
  const tickInner = radius - 10;
  const tickOuter = radius + 10;
  const x1 = cx + tickInner * Math.cos(rad);
  const y1 = cy + tickInner * Math.sin(rad);
  const x2 = cx + tickOuter * Math.cos(rad);
  const y2 = cy + tickOuter * Math.sin(rad);

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width={size} height={size * 0.68} className="overflow-visible">
        {/* Track arc */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Animated Value arc */}
        <motion.path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={gaugeColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: progressOffset }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
          style={{ filter: `drop-shadow(0 0 10px ${gaugeColor}88)` }}
        />

        {/* Decision threshold tick */}
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="#F4C76B"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <text
          x={x2 + (threshold > 0.5 ? 6 : -6)}
          y={y2 - 6}
          textAnchor={threshold > 0.5 ? 'start' : 'end'}
          fill="#F4C76B"
          fontSize="10"
          fontFamily="JetBrains Mono, monospace"
        >
          τ={Math.round(threshold * 100)}%
        </text>

        {/* Center Probability Text */}
        <text
          x={cx}
          y={cy - 18}
          textAnchor="middle"
          fill={gaugeColor}
          fontSize="30"
          fontWeight="bold"
          fontFamily="Cinzel, serif"
        >
          {pct.toFixed(1)}%
        </text>
        <text
          x={cx}
          y={cy + 6}
          textAnchor="middle"
          fill="#CBD5E1"
          fontSize="11"
          fontFamily="Inter, sans-serif"
        >
          {riskLabel}
        </text>
      </svg>
    </div>
  );
};

