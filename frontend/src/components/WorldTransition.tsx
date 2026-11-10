import React from 'react';
import { motion } from 'framer-motion';

interface WorldTransitionProps {
  children: React.ReactNode;
  worldKey: string;
}

export const WorldTransition: React.FC<WorldTransitionProps> = ({ children, worldKey }) => {
  return (
    <motion.div
      key={worldKey}
      initial={{ opacity: 0, y: 14, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="relative z-10 min-h-screen w-full overflow-x-clip"
    >
      {children}
    </motion.div>
  );
};

