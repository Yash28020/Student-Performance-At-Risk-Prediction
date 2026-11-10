import React from 'react';
import { motion } from 'framer-motion';

interface WorldBackdropProps {
  variant: 'gate' | 'kingdom' | 'academy' | 'nature' | 'cyberpunk' | 'celestial' | 'ocean';
}

export const WorldBackdrop: React.FC<WorldBackdropProps> = ({ variant }) => {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden select-none">
      {variant === 'gate' || variant === 'kingdom' ? (
        <>
          {/* Atmospheric Twilight Sky & Golden Aura */}
          <div className="absolute inset-0 bg-gradient-to-b from-[#070B19] via-[#131C3A] to-[#0B1022]" />
          <div className="absolute left-1/2 top-[18%] h-[520px] w-[720px] -translate-x-1/2 rounded-full bg-amber-400/15 blur-[140px]" />
          <div className="absolute right-[10%] top-[12%] h-72 w-72 rounded-full bg-purple-600/20 blur-[120px]" />

          {/* Distant Mountain Silhouettes */}
          <svg
            viewBox="0 0 1440 420"
            preserveAspectRatio="none"
            className="absolute bottom-0 left-0 h-[42vh] w-full opacity-40"
          >
            <path
              d="M0,420 L0,280 L180,190 L340,260 L540,140 L740,240 L940,160 L1160,250 L1320,175 L1440,235 L1440,420 Z"
              fill="#090F21"
            />
            <path
              d="M0,420 L0,330 L260,250 L490,320 L710,230 L960,315 L1220,240 L1440,305 L1440,420 Z"
              fill="#050814"
            />
          </svg>

          {/* Floating Enchanted Castle Silhouette */}
          <motion.div
            animate={{ y: [0, -14, 0] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
            className={`absolute left-1/2 -translate-x-1/2 ${
              variant === 'gate'
                ? 'top-[8%] w-[560px] md:w-[700px] opacity-45'
                : 'right-12 left-auto translate-x-0 top-[6%] w-[420px] opacity-20'
            }`}
          >
            <svg viewBox="0 0 600 380" className="h-full w-full drop-shadow-[0_0_40px_rgba(244,199,107,0.3)]">
              <defs>
                <linearGradient id="castleGold" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#F4C76B" stopOpacity="0.85" />
                  <stop offset="55%" stopColor="#3B2D63" stopOpacity="0.9" />
                  <stop offset="100%" stopColor="#10182E" stopOpacity="0.95" />
                </linearGradient>
              </defs>
              {/* Floating Rock Island Base */}
              <polygon points="120,260 480,260 420,315 310,365 210,315" fill="#182040" stroke="#F4C76B44" strokeWidth="1.5" />
              <polygon points="40,210 115,210 95,245 60,245" fill="#182040" opacity="0.7" />
              <polygon points="495,200 570,200 550,238 515,238" fill="#182040" opacity="0.7" />
              {/* Main Fortress Walls & Towers */}
              <rect x="200" y="150" width="200" height="110" fill="url(#castleGold)" />
              <rect x="270" y="90" width="60" height="170" fill="url(#castleGold)" />
              <polygon points="300,25 262,92 338,92" fill="#F4C76B" />
              {/* Side Spires */}
              <rect x="155" y="125" width="38" height="135" fill="url(#castleGold)" />
              <polygon points="174,68 150,126 198,126" fill="#F4C76B" />
              <rect x="407" y="125" width="38" height="135" fill="url(#castleGold)" />
              <polygon points="426,68 402,126 450,126" fill="#F4C76B" />
              {/* Glowing Archway Portal */}
              <path d="M275,260 L275,205 A25,25 0 0 1 325,205 L325,260 Z" fill="#F4C76B" opacity="0.9" />
              {/* Illuminated Windows */}
              <rect x="292" y="115" width="16" height="26" rx="8" fill="#FFE599" />
              <rect x="168" y="150" width="10" height="18" rx="5" fill="#FFE599" />
              <rect x="422" y="150" width="10" height="18" rx="5" fill="#FFE599" />
            </svg>
          </motion.div>
        </>
      ) : null}

      {variant === 'academy' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-b from-[#170F1E] via-[#24162C] to-[#0D0914]" />
          <div className="absolute left-[15%] top-[15%] h-96 w-96 rounded-full bg-amber-500/12 blur-[130px]" />
          <div className="absolute right-[10%] bottom-[10%] h-96 w-96 rounded-full bg-purple-700/20 blur-[140px]" />
          {/* Grand Gothic Arches Silhouette */}
          <svg viewBox="0 0 1200 600" preserveAspectRatio="none" className="absolute inset-0 h-full w-full opacity-15">
            {[150, 450, 750, 1050].map((cx) => (
              <path
                key={cx}
                d={`M${cx - 90},600 L${cx - 90},220 Q${cx},80 ${cx + 90},220 L${cx + 90},600`}
                fill="none"
                stroke="#C9A84C"
                strokeWidth="3"
              />
            ))}
          </svg>
        </>
      )}

      {variant === 'nature' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-b from-[#071B15] via-[#0F2F23] to-[#06130E]" />
          <div className="absolute left-[20%] top-[10%] h-[460px] w-[460px] rounded-full bg-emerald-500/15 blur-[140px]" />
          <div className="absolute right-[15%] bottom-[15%] h-80 w-80 rounded-full bg-teal-400/15 blur-[130px]" />
          {/* Ancient Canopy & Terraced Falls */}
          <svg viewBox="0 0 1440 420" preserveAspectRatio="none" className="absolute bottom-0 left-0 h-[36vh] w-full opacity-35">
            <path d="M0,420 C240,280 440,340 720,250 C980,170 1220,290 1440,220 L1440,420 Z" fill="#082419" />
            <path d="M0,420 C300,320 620,380 960,290 C1180,230 1320,330 1440,300 L1440,420 Z" fill="#04150E" />
          </svg>
        </>
      )}

      {variant === 'cyberpunk' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-b from-[#050814] via-[#0F1030] to-[#040710]" />
          <div className="absolute left-[10%] top-[15%] h-96 w-96 rounded-full bg-cyan-500/15 blur-[130px]" />
          <div className="absolute right-[10%] top-[25%] h-96 w-96 rounded-full bg-violet-600/20 blur-[140px]" />
          {/* Holographic Perspective Grid */}
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage:
                'linear-gradient(to right, rgba(0,245,255,0.18) 1px, transparent 1px), linear-gradient(to bottom, rgba(138,43,226,0.18) 1px, transparent 1px)',
              backgroundSize: '48px 48px',
            }}
          />
        </>
      )}

      {variant === 'celestial' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-b from-[#040614] via-[#120B2B] to-[#040714]" />
          <div className="absolute left-[18%] top-[18%] h-[420px] w-[420px] rounded-full bg-indigo-500/20 blur-[150px]" />
          <div className="absolute right-[12%] bottom-[12%] h-[380px] w-[380px] rounded-full bg-purple-600/20 blur-[150px]" />
          {/* Rotating Orbital Rings */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 90, repeat: Infinity, ease: 'linear' }}
            className="absolute -right-40 -top-40 h-[560px] w-[560px] rounded-full border border-indigo-400/15"
          />
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 65, repeat: Infinity, ease: 'linear' }}
            className="absolute -right-24 -top-24 h-[430px] w-[430px] rounded-full border border-dashed border-amber-300/15"
          />
        </>
      )}

      {variant === 'ocean' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-b from-[#041423] via-[#07263B] to-[#030D17]" />
          <div className="absolute left-[25%] top-[10%] h-[460px] w-[460px] rounded-full bg-cyan-400/15 blur-[140px]" />
          <div className="absolute right-[15%] bottom-[10%] h-96 w-96 rounded-full bg-teal-400/15 blur-[130px]" />
          {/* Underwater Crystal Spires Silhouette */}
          <svg viewBox="0 0 1200 320" preserveAspectRatio="none" className="absolute bottom-0 left-0 h-[28vh] w-full opacity-25">
            <polygon points="80,320 130,110 185,320" fill="#5DE6E0" opacity="0.25" />
            <polygon points="155,320 205,175 245,320" fill="#00FFD4" opacity="0.2" />
            <polygon points="940,320 1005,95 1065,320" fill="#5DE6E0" opacity="0.25" />
            <polygon points="1030,320 1085,165 1135,320" fill="#00FFD4" opacity="0.2" />
          </svg>
        </>
      )}
    </div>
  );
};

