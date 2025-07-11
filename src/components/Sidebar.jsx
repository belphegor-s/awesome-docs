import { NavLink } from 'react-router-dom';
import { listDocs } from '../docs';
import { FaPython, FaHome } from 'react-icons/fa';

const ICONS = {
  Python: FaPython,
};

const Sidebar = ({ onClose }) => {
  const docs = listDocs();

  const handleClick = () => {
    if (onClose) onClose();
  };

  return (
    <div className="p-4 min-h-screen">
      <h1 className="text-2xl font-bold mb-6">Awesome Docs</h1>
      <nav className="space-y-2">
        <NavLink to="/" onClick={handleClick} className="grid grid-cols-[auto_1fr] gap-2 items-center px-3 py-2 text-slate-300 hover:bg-slate-700 rounded">
          <FaHome /> Home
        </NavLink>
        {docs.map((doc) => {
          const Icon = ICONS[doc];
          return (
            <NavLink
              key={doc}
              to={`/docs/${doc}`}
              onClick={handleClick}
              className={({ isActive }) =>
                `grid grid-cols-[auto_1fr] gap-2 items-center px-3 py-2 rounded text-sm font-medium ${isActive ? 'bg-slate-700 text-white' : 'text-slate-300 hover:bg-slate-700'}`
              }
            >
              <Icon /> {doc}
            </NavLink>
          );
        })}
      </nav>
    </div>
  );
};

export default Sidebar;
