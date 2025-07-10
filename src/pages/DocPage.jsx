import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import MarkdownPreview from '@uiw/react-markdown-preview';
import { loadDoc } from '../docs';

const DocPage = () => {
  const { docId } = useParams();
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    loadDoc(docId).then((res) => {
      setContent(res);
      setLoading(false);
    });
  }, [docId]);

  if (loading) return <div className="p-10 text-slate-400">Loading...</div>;

  if (!content) {
    return <div className="p-10 text-red-400 text-xl">❌ No document found for "{docId}"</div>;
  }

  return (
    <div className="p-4 sm:p-10">
      <MarkdownPreview source={content} className="p-4 sm:p-8 rounded-lg border border-gray-800" />
    </div>
  );
};

export default DocPage;
