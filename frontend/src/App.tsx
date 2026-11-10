import React from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Navigation } from './components/Navigation';
import { KingdomGate } from './pages/KingdomGate';
import { World1Kingdom } from './pages/World1Kingdom';
import { World2Academy } from './pages/World2Academy';
import { World3Nature } from './pages/World3Nature';
import { World4Cyberpunk } from './pages/World4Cyberpunk';
import { World5Celestial } from './pages/World5Celestial';
import { World6Ocean } from './pages/World6Ocean';

const AnimatedRoutes: React.FC = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<KingdomGate />} />
        <Route path="/kingdom" element={<World1Kingdom />} />
        <Route path="/academy" element={<World2Academy />} />
        <Route path="/nature" element={<World3Nature />} />
        <Route path="/cyberpunk" element={<World4Cyberpunk />} />
        <Route path="/celestial" element={<World5Celestial />} />
        <Route path="/ocean" element={<World6Ocean />} />
        <Route path="*" element={<KingdomGate />} />
      </Routes>
    </AnimatePresence>
  );
};

export function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen w-full overflow-x-clip bg-[#070B19] text-white">
        <Navigation />
        <main className="min-w-0 flex-1 overflow-x-clip">
          <AnimatedRoutes />
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
