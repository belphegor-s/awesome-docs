const modules = import.meta.glob('./*.js');

export const loadDoc = async (docId) => {
  const matchKey = `./${docId}.js`;
  const loader = modules[matchKey];

  if (!loader) return null;

  const mod = await loader();

  if (mod[docId]) return mod[docId];

  if (mod.default) return mod.default;

  return null;
};

export const listDocs = () => {
  return Object.keys(modules).map((path) => path.replace('./', '').replace('.js', ''));
};
