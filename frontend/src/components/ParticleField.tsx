import React, { useEffect, useRef } from 'react';

export type ParticleTheme = 'kingdom' | 'academy' | 'nature' | 'cyberpunk' | 'celestial' | 'ocean';

interface ParticleFieldProps {
  theme: ParticleTheme;
  density?: number;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  pulseSpeed: number;
  phase: number;
  color: string;
}

const THEME_PALETTES: Record<ParticleTheme, string[]> = {
  kingdom: ['#F4C76B', '#FFE599', '#7555C7', '#FFFFFF'],
  academy: ['#F4C76B', '#C9A84C', '#F5E6C8', '#A855F7'],
  nature: ['#5DBB8A', '#2D9978', '#A7F3D0', '#FDE68A'],
  cyberpunk: ['#00F5FF', '#56A8F5', '#8A2BE2', '#E0F2FE'],
  celestial: ['#FFFFFF', '#56A8F5', '#FFE566', '#A855F7'],
  ocean: ['#5DE6E0', '#00FFD4', '#A8E6E3', '#38BDF8'],
};

export const ParticleField: React.FC<ParticleFieldProps> = ({ theme, density = 55 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    const palette = THEME_PALETTES[theme];
    const particles: Particle[] = Array.from({ length: density }, () => {
      const color = palette[Math.floor(Math.random() * palette.length)];
      const isCyber = theme === 'cyberpunk';
      const isCelestial = theme === 'celestial';
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: isCyber ? 0 : (Math.random() - 0.5) * (isCelestial ? 0.12 : 0.35),
        vy: isCyber
          ? Math.random() * 1.4 + 0.5
          : isCelestial
            ? (Math.random() - 0.5) * 0.12
            : -(Math.random() * 0.45 + 0.15),
        size: isCelestial ? Math.random() * 1.8 + 0.5 : Math.random() * 2.8 + 1.0,
        alpha: Math.random() * 0.6 + 0.2,
        pulseSpeed: Math.random() * 0.03 + 0.01,
        phase: Math.random() * Math.PI * 2,
        color,
      };
    });

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.phase += p.pulseSpeed;

        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;
        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;

        const currentAlpha = Math.max(0.08, Math.min(0.9, p.alpha + Math.sin(p.phase) * 0.25));

        ctx.save();
        ctx.globalAlpha = currentAlpha;
        ctx.fillStyle = p.color;

        if (theme === 'ocean') {
          // Rising bioluminescent bubbles
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 1.5, 0, Math.PI * 2);
          ctx.strokeStyle = p.color;
          ctx.lineWidth = 0.9;
          ctx.stroke();
          ctx.globalAlpha = currentAlpha * 0.25;
          ctx.fill();
        } else if (theme === 'cyberpunk') {
          // Vertical neon data streaks
          ctx.fillRect(p.x, p.y, 1.5, p.size * 6);
        } else {
          // Soft glowing orb / ember / star
          ctx.shadowBlur = theme === 'celestial' ? 6 : 12;
          ctx.shadowColor = p.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [theme, density]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0 h-full w-full opacity-75"
      aria-hidden="true"
    />
  );
};

