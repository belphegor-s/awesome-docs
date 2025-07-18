import { Link } from 'react-router-dom';
import { listDocs } from '../docs';

const docNames = listDocs();

const Home = () => {
  return (
    <div className="p-10">
      <h1 className="text-3xl font-bold mb-6">Welcome to the Awesome Docs 🧠</h1>
      <p className="text-slate-300 mb-8">Select a documentation file from the sidebar or jump right into one below:</p>
      <ul className="space-y-4">
        {docNames.map((doc) => (
          <li key={doc}>
            <Link to={`/docs/${doc}`} className="group text-blue-400 text-lg">
              → <span className="group-hover:underline">{doc}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Home;
