import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import Home from './pages/Home';
import DocPage from './pages/DocPage';
import Sidebar from './components/Sidebar';
import { Menu } from 'lucide-react';

const App = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <Router>
      <div className="flex h-screen bg-gray-950 text-white relative">
        <div
          className={`
            fixed top-0 left-0 z-50 h-full w-64 bg-gary-950 border-r border-slate-700
            transform transition-transform duration-300 ease-in-out
            md:static md:translate-x-0 md:z-auto
            ${drawerOpen ? 'translate-x-0' : '-translate-x-full'}
          `}
        >
          <Sidebar onClose={() => setDrawerOpen(false)} />
        </div>

        <div
          onClick={() => setDrawerOpen(false)}
          className={`
            fixed inset-0 bg-black/50 backdrop-blur-sm z-40
            transition-opacity duration-200
            md:hidden ${drawerOpen ? 'block' : 'hidden'}
          `}
        />

        <div className="flex-1 overflow-y-auto">
          <div className="md:hidden p-4 flex items-center">
            <button onClick={() => setDrawerOpen(true)}>
              <Menu className="w-6 h-6 text-white" />
            </button>
            <span className="ml-4 text-lg font-semibold">Docs Hub</span>
          </div>

          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/docs/:docId" element={<DocPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
