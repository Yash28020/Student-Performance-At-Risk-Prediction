/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // World 1 - Enchanted Kingdom
        kingdom: {
          gold: '#F4C76B',
          dark: '#10182E',
          purple: '#7555C7',
          light: '#E8D5A3',
        },
        // World 2 - Wizarding Academy
        academy: {
          parchment: '#F5E6C8',
          ink: '#2C1A0E',
          crimson: '#8B1A1A',
          gold: '#C9A84C',
        },
        // World 3 - Nature Realm
        nature: {
          emerald: '#2D9978',
          forest: '#1A4A2E',
          leaf: '#5DBB8A',
          moss: '#3D7A5A',
        },
        // World 4 - Cyberpunk
        cyber: {
          neon: '#00F5FF',
          violet: '#8A2BE2',
          dark: '#050A14',
          electric: '#56A8F5',
        },
        // World 5 - Celestial
        celestial: {
          blue: '#56A8F5',
          nebula: '#7B2FBE',
          star: '#FFE566',
          void: '#050815',
        },
        // World 6 - Ocean
        ocean: {
          cyan: '#5DE6E0',
          deep: '#071B2E',
          biolum: '#00FFD4',
          crystal: '#A8E6E3',
        },
      },
      fontFamily: {
        fantasy: ['Cinzel', 'serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'float-slow': 'float 10s ease-in-out infinite',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'twinkle': 'twinkle 3s ease-in-out infinite',
        'drift': 'drift 20s linear infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'portal': 'portal 4s ease-in-out infinite',
        'bubble': 'bubble 8s ease-in-out infinite',
        'scan': 'scan 3s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(244,199,107,0.4)' },
          '50%': { boxShadow: '0 0 50px rgba(244,199,107,0.8), 0 0 80px rgba(244,199,107,0.3)' },
        },
        twinkle: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.3', transform: 'scale(0.8)' },
        },
        drift: {
          '0%': { transform: 'translateX(-100vw)' },
          '100%': { transform: 'translateX(100vw)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        portal: {
          '0%, 100%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.1)' },
        },
        bubble: {
          '0%': { transform: 'translateY(100vh) scale(0)', opacity: '0' },
          '10%': { opacity: '0.6' },
          '90%': { opacity: '0.6' },
          '100%': { transform: 'translateY(-100px) scale(1)', opacity: '0' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

