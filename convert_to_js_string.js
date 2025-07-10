import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const docsDir = path.resolve(__dirname, 'docs');
const docsSrcDir = path.resolve(__dirname, 'src/docs');

const files = readdirSync(docsDir);

files.forEach((file) => {
  const fullPath = path.join(docsDir, file);

  // Skip non-files
  if (!statSync(fullPath).isFile()) return;

  // Only process .md files
  if (path.extname(file) !== '.md') return;

  const content = readFileSync(fullPath, 'utf-8');

  let baseName = path.basename(file, '.md');
  baseName = baseName.replace(/_/g, ' ').replace(/\w\S*/g, (w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase());
  const outputPath = path.join(docsSrcDir, `${baseName}.js`);

  const jsContent = `export const ${baseName} = ${JSON.stringify(content, null, 2)};\n`;

  writeFileSync(outputPath, jsContent, 'utf-8');
  console.log(`✅ Converted ${file} → ${baseName}.js`);
});
