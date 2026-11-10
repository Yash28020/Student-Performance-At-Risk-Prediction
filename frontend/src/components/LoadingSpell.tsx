import React from 'react';
import { motion } from 'framer-motion';

interface LoadingSpellProps {
  message?: string;
  accentColor?: string;
}

export const LoadingSpell: React.FC<LoadingSpellProps> = ({
  message = 'Channeling EduGuard Intelligence...',
  accentColor = '#F4C76B',
}) => {
  return (
    <div className="flex min-h-[320px] flex-col items-center justify-center gap-5 p-8">
      <div className="relative flex h-20 w-20 items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          className="absolute inset-0 rounded-full border-2 border-dashed"
          style={{ borderColor: accentColor }}
        />
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
          className="absolute inset-2.5 rounded-full border-2 border-t-transparent"
          style={{ borderColor: `${accentColor} transparent ${accentColor} ${accentColor}` }}
        />
        <div
          className="h-3 w-3 rounded-full shadow-lg"
          style={{ backgroundColor: accentColor, boxShadow: `0 0 18px ${accentColor}` }}
        />
      </div>
      <p className="font-fantasy text-sm tracking-widest text-slate-200">{message}</p>
    </div>
  );
};

